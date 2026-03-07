from fastapi import FastAPI, Query
import requests
import os

app = FastAPI(title="Door Face Verify API")

# In-memory enrollment map: face_id -> reference image URL
enrolled_faces: dict[str, str] = {}

FACEPP_API_URL = os.getenv("FACEPP_API_URL", "https://api-us.faceplusplus.com/facepp/v3/compare")
FACEPP_API_KEY = os.getenv("FACEPP_API_KEY", "")
FACEPP_API_SECRET = os.getenv("FACEPP_API_SECRET", "")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "75"))


def fetch_cam_capture_bytes(cam_ip: str) -> bytes:
    url = f"http://{cam_ip}/capture"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.content


def facepp_compare(cam_image_bytes: bytes, ref_image_url: str) -> dict:
    if not FACEPP_API_KEY or not FACEPP_API_SECRET:
        raise RuntimeError("facepp_credentials_missing")

    files = {
        "image_file1": ("cam.jpg", cam_image_bytes, "image/jpeg"),
    }
    data = {
        "api_key": FACEPP_API_KEY,
        "api_secret": FACEPP_API_SECRET,
        "image_url2": ref_image_url,
    }
    resp = requests.post(FACEPP_API_URL, data=data, files=files, timeout=10)
    resp.raise_for_status()
    return resp.json()


@app.get("/health")
def health():
    return {
        "ok": True,
        "known_faces": len(enrolled_faces),
        "mode": "facepp",
        "facepp_configured": bool(FACEPP_API_KEY and FACEPP_API_SECRET),
        "match_threshold": MATCH_THRESHOLD,
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
        result = facepp_compare(cam_bytes, ref_url)

        confidence = float(result.get("confidence", 0.0))
        matched = confidence >= MATCH_THRESHOLD

        return {
            "matched": matched,
            "score": round(confidence / 100.0, 4),
            "user_id": face_id if matched else "",
            "reason": "ok" if matched else "not_match",
            "confidence": confidence,
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
    - Verification compares live CAM capture with this URL in Face++
    """
    enrolled_faces[face_id] = image_url
    return {"ok": True, "face_id": face_id, "image_url": image_url, "total": len(enrolled_faces)}
