from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx
import json
import uuid
import os
import base64
import asyncio
from pathlib import Path
from datetime import datetime

app = FastAPI(title="Kartu Lebaran AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

TEMPLATES = json.loads(Path("prompt_templates.json").read_text())
jobs = {}
VIDEOS_DIR = Path("videos")
IMAGES_DIR = Path("images")
VIDEOS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

ARK_BASE = "https://ark.ap-southeast.bytepluses.com/api/v3"
SEEDREAM_MODEL = os.environ.get("SEEDREAM_MODEL_ID", "seedream-5-0-260128")
SEEDANCE_MODEL = os.environ.get("SEEDANCE_MODEL_ID", "seedance-1-5-pro-251215")


def build_image_prompt(location: str, vibe: str, user_message: str) -> str:
    t = TEMPLATES
    loc = t["locations"][location]["scene"]
    emotion = t["vibes"][vibe]["emotion"]
    base_style = t["base_style"]
    return (
        f"Portrait of the exact person from the reference image placed in {loc}. "
        "CRITICAL: Retain the exact facial features, age, gender, and identity of the input image. "
        f"Emotion: {emotion}. Wearing traditional Eid outfit. "
        f"Text overlay: '{user_message}'. {base_style}. "
        "High resolution, photorealistic, soft golden lighting, festive Eid Al-Fitr atmosphere."
    )


def build_video_prompt(location: str, vibe: str, music: str) -> str:
    t = TEMPLATES
    cinematics = t["vibes"][vibe]["cinematics"]
    atmosphere = t["music_moods"][music]["atmosphere"]
    return (
        "CRITICAL: Maintain 100% facial consistency with the reference image throughout. "
        "The face must not morph, change, or transform. "
        f"Animate environment with gentle motion. {cinematics}. Mood: {atmosphere}. "
        "Only animate background: particles floating, fabric swaying softly. Smooth ending."
    )


async def call_seedream(image_prompt: str, face_b64: str, job_id: str) -> str:
    jobs[job_id]["status"] = "step1_generating_scene"
    headers = {
        "Authorization": f"Bearer {os.environ['BYTEPLUS_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SEEDREAM_MODEL,
        "prompt": image_prompt,
        "image_urls": [f"data:image/jpeg;base64,{face_b64}"],
        "response_format": "url",
        "size": "2K",
        "watermark": False,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{ARK_BASE}/images/generations", headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Seedream error {r.status_code}: {r.text}")
        result = r.json()
        data = result.get("data", [])
        if not data:
            raise RuntimeError(f"Seedream: no data — {result}")
        item = data[0]
        if "b64_json" in item:
            image_b64 = item["b64_json"]
        elif "url" in item:
            img_r = await client.get(item["url"])
            image_b64 = base64.b64encode(img_r.content).decode("utf-8")
        else:
            raise RuntimeError(f"Seedream: unknown response format — {result}")
        img_path = IMAGES_DIR / f"{job_id}_scene.jpg"
        img_path.write_bytes(base64.b64decode(image_b64))
        jobs[job_id]["scene_image_path"] = str(img_path)
        return image_b64


async def call_seedance(video_prompt: str, scene_b64: str, job_id: str) -> None:
    jobs[job_id]["status"] = "step2_animating_video"
    headers = {
        "Authorization": f"Bearer {os.environ['BYTEPLUS_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SEEDANCE_MODEL,
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{scene_b64}"}},
            {"type": "text", "text": video_prompt + " --duration 5 --resolution 1080p --ratio 9:16 --camerafixed true"},
        ],
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(f"{ARK_BASE}/contents/generations/tasks", headers=headers, json=payload)
        r.raise_for_status()
        task_id = r.json().get("id")
        if not task_id:
            raise RuntimeError(f"Seedance: no task ID — {r.json()}")

        video_url = None
        for _ in range(72):
            await asyncio.sleep(5)
            pr = await client.get(f"{ARK_BASE}/contents/generations/tasks/{task_id}", headers=headers)
            poll = pr.json()
            jobs[job_id]["step2_last_poll"] = str(poll)
            status = poll.get("status", "")
            if status in ("succeeded", "success"):
                content = poll.get("content", {})
                if isinstance(content, dict):
                    video_url = content.get("video_url")
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("video_url"):
                            video_url = item["video_url"]
                            break
                break
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"Seedance failed — {poll}")

        if not video_url:
            raise RuntimeError(f"Seedance: no video URL. Last: {jobs[job_id].get('step2_last_poll')}")

        vr = await client.get(video_url, timeout=120.0)
        video_path = VIDEOS_DIR / f"{job_id}.mp4"
        video_path.write_bytes(vr.content)

    jobs[job_id]["status"] = "done"
    jobs[job_id]["video_path"] = str(video_path)
    jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


async def run_pipeline(face_b64: str, image_prompt: str, video_prompt: str, job_id: str) -> None:
    try:
        scene_b64 = await call_seedream(image_prompt, face_b64, job_id)
        await call_seedance(video_prompt, scene_b64, job_id)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.get("/")
def root():
    return {"message": "Kartu Lebaran AI — 2-step pipeline", "status": "running"}


@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    photo: UploadFile = File(...),
    location: str = Form(...),
    vibe: str = Form(...),
    music: str = Form(...),
    user_message: str = Form(default="Minal Aidin Wal Faizin"),
):
    photo_bytes = await photo.read()
    face_b64 = base64.b64encode(photo_bytes).decode("utf-8")
    image_prompt = build_image_prompt(location, vibe, user_message)
    video_prompt = build_video_prompt(location, vibe, music)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id": job_id, "status": "queued", "created_at": datetime.utcnow().isoformat()}
    background_tasks.add_task(run_pipeline, face_b64, image_prompt, video_prompt, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
        "completed_at": job.get("completed_at"),
        "scene_ready": "scene_image_path" in job,
    }


@app.get("/download/{job_id}")
def get_download(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"Not ready: {job['status']}")
    return FileResponse(job["video_path"], media_type="video/mp4", filename="kartu-lebaran.mp4")


@app.get("/preview/{job_id}")
def get_preview(job_id: str):
    job = jobs.get(job_id)
    if not job or "scene_image_path" not in job:
        raise HTTPException(404, "Scene image not ready")
    return FileResponse(job["scene_image_path"], media_type="image/jpeg", filename="scene.jpg")
