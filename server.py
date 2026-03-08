import os
import tempfile

from fastapi import Body, FastAPI, Query
import requests

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

app = FastAPI(title="Door Face Verify API")

# In-memory enrollment map: face_id -> reference image URL
enrolled_faces: dict[str, str] = {}
DEFAULT_FACE_ID = os.getenv("DEFAULT_FACE_ID", "owner")
DEFAULT_OWNER_IMAGE_URL = os.getenv("DEFAULT_OWNER_IMAGE_URL", "")

if DEFAULT_OWNER_IMAGE_URL:
    enrolled_faces[DEFAULT_FACE_ID] = DEFAULT_OWNER_IMAGE_URL

DEEPFACE_MODEL_NAME = os.getenv("DEEPFACE_MODEL_NAME", "Facenet512")
DEEPFACE_DISTANCE_THRESHOLD = float(os.getenv("DEEPFACE_DISTANCE_THRESHOLD", "0.35"))

pending_verify = {
    "active": False,
    "taken": False,
    "command_id": 0,
    "face_id": "",
}

last_verify_result = {
    "command_id": 0,
    "matched": False,
    "score": 0.0,
    "user_id": "",
    "reason": "none",
}


def fetch_cam_capture_bytes(cam_ip: str) -> bytes:
    url = f"http://{cam_ip}/capture"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.content


def fetch_url_bytes(image_url: str) -> bytes:
    resp = requests.get(image_url, timeout=10)
    resp.raise_for_status()
    return resp.content


def deepface_compare(cam_image_bytes: bytes, ref_image_url: str) -> dict:
    if DeepFace is None:
        raise RuntimeError("deepface_not_installed")

    ref_image_bytes = fetch_url_bytes(ref_image_url)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as cam_file, tempfile.NamedTemporaryFile(
        suffix=".jpg", delete=True
    ) as ref_file:
        cam_file.write(cam_image_bytes)
        cam_file.flush()
        ref_file.write(ref_image_bytes)
        ref_file.flush()

        result = DeepFace.verify(
            img1_path=cam_file.name,
            img2_path=ref_file.name,
            model_name=DEEPFACE_MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=True,
        )

    distance = float(result.get("distance", 1.0))
    threshold = float(result.get("threshold", DEEPFACE_DISTANCE_THRESHOLD))
    matched = bool(result.get("verified", False)) and distance <= DEEPFACE_DISTANCE_THRESHOLD

    return {
        "matched": matched,
        "distance": distance,
        "threshold": threshold,
        "model": result.get("model", DEEPFACE_MODEL_NAME),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "known_faces": len(enrolled_faces),
        "mode": "deepface",
        "deepface_available": DeepFace is not None,
        "deepface_model": DEEPFACE_MODEL_NAME,
        "distance_threshold": DEEPFACE_DISTANCE_THRESHOLD,
    }


@app.get("/verify_from_cam")
def verify_from_cam(
    cam_ip: str = Query(..., description="ESP32-CAM LAN IP"),
    face_id: str = Query(..., description="Enrolled face_id to verify against"),
):
    if face_id not in enrolled_faces:
        return {
            "matched": False,
            "score": 0.0,
            "user_id": "",
            "reason": "face_id_not_enrolled",
        }

    try:
        cam_bytes = fetch_cam_capture_bytes(cam_ip)
        ref_url = enrolled_faces[face_id]
        result = deepface_compare(cam_bytes, ref_url)
        distance = float(result.get("distance", 1.0))
        matched = bool(result.get("matched", False))
        score = max(0.0, min(1.0, 1.0 - distance / max(DEEPFACE_DISTANCE_THRESHOLD, 0.0001)))

        return {
            "matched": matched,
            "score": round(score, 4),
            "user_id": face_id if matched else "",
            "reason": "ok" if matched else "not_match",
            "distance": distance,
            "threshold": result.get("threshold", DEEPFACE_DISTANCE_THRESHOLD),
            "model": result.get("model", DEEPFACE_MODEL_NAME),
        }

    except Exception as exc:
        return {
            "matched": False,
            "score": 0.0,
            "user_id": "",
            "reason": f"error:{exc}",
        }


@app.post("/trigger_verify")
def trigger_verify(face_id: str = Query("owner")):
    pending_verify["active"] = True
    pending_verify["taken"] = False
    pending_verify["command_id"] += 1
    pending_verify["face_id"] = face_id
    return {
        "ok": True,
        "command_id": pending_verify["command_id"],
        "face_id": face_id,
    }


@app.get("/next_verify")
def next_verify():
    if pending_verify["active"] and not pending_verify["taken"]:
        pending_verify["taken"] = True
        return {
            "verify": True,
            "command_id": pending_verify["command_id"],
            "face_id": pending_verify["face_id"],
        }
    return {"verify": False}


@app.post("/report_verify")
def report_verify(
    command_id: int = Query(...),
    matched: bool = Query(False),
    score: float = Query(0.0),
    user_id: str = Query(""),
    reason: str = Query(""),
):
    last_verify_result["command_id"] = command_id
    last_verify_result["matched"] = matched
    last_verify_result["score"] = score
    last_verify_result["user_id"] = user_id
    last_verify_result["reason"] = reason

    if command_id == pending_verify["command_id"]:
        pending_verify["active"] = False
        pending_verify["taken"] = False

    return {"ok": True}


@app.get("/last_verify")
def last_verify():
    return {"ok": True, **last_verify_result}


@app.post("/verify_upload")
def verify_upload(
    face_id: str = Query(..., description="Enrolled face_id to verify against"),
    image_bytes: bytes = Body(..., media_type="application/octet-stream"),
):
    if face_id not in enrolled_faces:
        return {
            "matched": False,
            "score": 0.0,
            "user_id": "",
            "reason": "face_id_not_enrolled",
        }

    if not image_bytes:
        return {
            "matched": False,
            "score": 0.0,
            "user_id": "",
            "reason": "empty_image",
        }

    try:
        ref_url = enrolled_faces[face_id]
        result = deepface_compare(image_bytes, ref_url)
        distance = float(result.get("distance", 1.0))
        matched = bool(result.get("matched", False))
        score = max(0.0, min(1.0, 1.0 - distance / max(DEEPFACE_DISTANCE_THRESHOLD, 0.0001)))

        return {
            "matched": matched,
            "score": round(score, 4),
            "user_id": face_id if matched else "",
            "reason": "ok" if matched else "not_match",
            "distance": distance,
            "threshold": result.get("threshold", DEEPFACE_DISTANCE_THRESHOLD),
            "model": result.get("model", DEEPFACE_MODEL_NAME),
        }

    except Exception as exc:
        return {
            "matched": False,
            "score": 0.0,
            "user_id": "",
            "reason": f"error:{exc}",
        }


@app.post("/enroll_demo")
def enroll_demo(face_id: str = Query(...), image_url: str = Query(...)):
    """
    Quick demo enroll endpoint:
    - Saves one reference image URL for a face_id
    - Verification compares live CAM capture with this URL in DeepFace
    """
    enrolled_faces[face_id] = image_url
    return {"ok": True, "face_id": face_id, "image_url": image_url, "total": len(enrolled_faces)}
