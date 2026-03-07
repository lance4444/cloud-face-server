from fastapi import FastAPI, Query
import cv2
import numpy as np
import requests

try:
    import face_recognition  # requires dlib
except Exception:
    face_recognition = None

app = FastAPI(title="Door Face Verify API")

# TODO: Replace with your enrolled data loader.
# For now this is a placeholder in-memory store.
known_face_encodings: list[np.ndarray] = []
known_face_ids: list[str] = []

# Typical threshold for face_recognition distance comparison
MATCH_THRESHOLD = 0.50


def fetch_cam_capture(cam_ip: str) -> np.ndarray:
    url = f"http://{cam_ip}/capture"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image from /capture")
    return frame


@app.get("/health")
def health():
    return {
        "ok": True,
        "known_faces": len(known_face_ids),
        "face_recognition_available": face_recognition is not None,
    }


@app.get("/verify_from_cam")
def verify_from_cam(cam_ip: str = Query(..., description="ESP32-CAM LAN IP")):
    if face_recognition is None:
        return {
            "matched": False,
            "score": 0.0,
            "user_id": "",
            "reason": "face_recognition_not_installed",
        }

    if not known_face_encodings:
        return {
            "matched": False,
            "score": 0.0,
            "user_id": "",
            "reason": "no_enrolled_faces",
        }

    try:
        frame_bgr = fetch_cam_capture(cam_ip)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(frame_rgb, model="hog")
        if not locations:
            return {
                "matched": False,
                "score": 0.0,
                "user_id": "",
                "reason": "no_face_detected",
            }

        encodings = face_recognition.face_encodings(frame_rgb, locations)
        if not encodings:
            return {
                "matched": False,
                "score": 0.0,
                "user_id": "",
                "reason": "encoding_failed",
            }

        candidate = encodings[0]
        distances = face_recognition.face_distance(known_face_encodings, candidate)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        matched = best_dist <= MATCH_THRESHOLD
        score = max(0.0, min(1.0, 1.0 - best_dist))

        return {
            "matched": matched,
            "score": round(score, 4),
            "user_id": known_face_ids[best_idx] if matched else "",
            "reason": "ok" if matched else "not_match",
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
    - Downloads one image from URL
    - Extracts first face encoding
    - Stores in memory
    """
    if face_recognition is None:
        return {"ok": False, "reason": "face_recognition_not_installed"}

    try:
        resp = requests.get(image_url, timeout=8)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(img_rgb, model="hog")
        if not locs:
            return {"ok": False, "reason": "no_face"}

        encs = face_recognition.face_encodings(img_rgb, locs)
        if not encs:
            return {"ok": False, "reason": "encode_failed"}

        known_face_ids.append(face_id)
        known_face_encodings.append(encs[0])
        return {"ok": True, "face_id": face_id, "total": len(known_face_ids)}

    except Exception as exc:
        return {"ok": False, "reason": str(exc)}
