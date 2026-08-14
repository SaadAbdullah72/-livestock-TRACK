import hashlib
import json
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import numpy as np
from ultralytics import YOLO

from core.arcface_extractor import ArcFaceEngine
from core.matcher import BiometricMatcher

app = FastAPI(title="Livestock Biometric Passport API")

# Initialize models
detector = YOLO("yolov8n.pt")
extractor = ArcFaceEngine()
matcher = BiometricMatcher(threshold=0.75)

# In-Memory Database (Vector Storage)
PASSPORT_DB = {}


@app.post("/api/passport/register")
async def register_cattle_passport(
    tag_id: str = Form(...),
    breed: str = Form(...),
    owner_wallet: str = Form(...),
    file: UploadFile = File(...),
):
    # Read Image Bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image upload.")

    # 1. YOLO Muzzle Crop
    results = detector(img, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].cpu().numpy())
        cropped = img[y1:y2, x1:x2]
    else:
        h, w, _ = img.shape
        cropped = img[
            int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)
        ]

    # 2. Extract ArcFace Vector
    vector = extractor.extract_vector(cropped)

    # 3. Generate Primary Biometric Hash
    binary_bits = (vector > 0).astype(int)
    bit_str = "".join(map(str, binary_bits))
    bio_hash = "0x" + hashlib.sha256(bit_str.encode()).hexdigest()

    # 4. Save to Database
    passport_data = {
        "tag_id": tag_id,
        "breed": breed,
        "owner_wallet": owner_wallet,
        "biometric_hash": bio_hash,
        "vector": vector.tolist(),
    }
    PASSPORT_DB[tag_id] = passport_data

    return {
        "status": "SUCCESS",
        "message": "Digital Cattle Passport Registered!",
        "tag_id": tag_id,
        "biometric_hash": bio_hash,
    }


@app.post("/api/passport/verify")
async def verify_cattle(tag_id: str = Form(...), file: UploadFile = File(...)):
    if tag_id not in PASSPORT_DB:
        raise HTTPException(
            status_code=404, detail="Passport Tag ID not found in database."
        )

    # Read New Scan Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO Crop & Feature Extraction
    results = detector(img, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].cpu().numpy())
        cropped = img[y1:y2, x1:x2]
    else:
        h, w, _ = img.shape
        cropped = img[
            int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)
        ]

    new_vector = extractor.extract_vector(cropped)
    registered_vector = np.array(PASSPORT_DB[tag_id]["vector"])

    # 3. Match Biometrics
    match_result = matcher.compute_similarity(registered_vector, new_vector)

    return {
        "status": "PROCESSED",
        "tag_id": tag_id,
        "owner_wallet": PASSPORT_DB[tag_id]["owner_wallet"],
        "similarity_score": f"{match_result['match_percentage']}%",
        "is_authenticated": match_result["is_matched"],
        "verdict": match_result["verdict"],
    }