import hashlib
import json
import os
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

app = FastAPI(title="Livestock Biometric Passport & Registration API")

# 1. Storage Paths
STORAGE_DIR = "passports_data"
os.makedirs(STORAGE_DIR, exist_ok=True)

# 2. Models Setup
print("[INFO] Initializing Vision & Biometric Pipeline...")
detector = YOLO("yolov8n.pt")

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# Image Preprocessing (Micro-texture contrast enhancement)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# 3. Vector Database (Qdrant In-Memory for Instant Search)
qdrant = QdrantClient(":memory:")
COLLECTION_NAME = "cattle_passports"
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

point_id_counter = 1


def extract_biometrics(img_bgr):
    """Detects Muzzle, extracts 512-D Vector & Generates Biometric Hash."""
    # Step A: YOLO Muzzle Crop / Fallback
    results = detector(img_bgr, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].cpu().numpy())
        cropped = img_bgr[y1:y2, x1:x2]
    else:
        h, w, _ = img_bgr.shape
        cropped = img_bgr[
            int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)
        ]

    # Step B: Contrast Enhance (Pores/Ridges)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Step C: Deep Vector Extraction
    tensor = preprocess(enhanced_bgr).unsqueeze(0)
    with torch.no_grad():
        raw_vector = feature_extractor(tensor).flatten().numpy()

    # Normalized Vector (Unit Length for Cosine Matching)
    norm_vector = raw_vector / np.linalg.norm(raw_vector)

    # Step D: Cryptographic Anchor Hash
    binary_bits = (raw_vector > np.median(raw_vector)).astype(int)
    bit_str = "".join(map(str, binary_bits))
    bio_hash = "0x" + hashlib.sha256(bit_str.encode()).hexdigest()

    return norm_vector, bio_hash, cropped


# ----------------------------------------------------
# REGISTRATION ENDPOINT (Digital Passport Creation)
# ----------------------------------------------------
@app.post("/api/register")
async def register_cattle(
    tag_id: str = Form(...),
    breed: str = Form(...),
    age_months: int = Form(...),
    weight_kg: float = Form(...),
    owner_wallet: str = Form(...),
    file: UploadFile = File(...),
):
    global point_id_counter

    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=400, detail="Invalid image file uploaded."
        )

    # 1. Extract Biometric DNA
    vector, bio_hash, cropped_img = extract_biometrics(img)

    # 2. Check for Duplicate Animal (Deduplication Check)
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=vector.tolist(), limit=1
    ).points

    if search_result and search_result[0].score > 0.85:
        existing_tag = search_result[0].payload["tag_id"]
        raise HTTPException(
            status_code=409,
            detail=f"Registration Rejected: This animal is already registered under Tag ID: {existing_tag} (Similarity: {search_result[0].score*100:.2f}%)",
        )

    # 3. Store Vector in Qdrant DB
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id_counter,
                vector=vector.tolist(),
                payload={
                    "tag_id": tag_id,
                    "breed": breed,
                    "age_months": age_months,
                    "weight_kg": weight_kg,
                    "owner_wallet": owner_wallet,
                    "biometric_hash": bio_hash,
                },
            )
        ],
    )
    point_id_counter += 1

    # 4. Save Passport JSON on Local Storage (IPFS Ready)
    passport_data = {
        "passport_id": f"PASSPORT-{tag_id}",
        "metadata": {
            "tag_id": tag_id,
            "breed": breed,
            "age_months": age_months,
            "weight_kg": weight_kg,
            "owner_wallet": owner_wallet,
            "status": "ACTIVE_REGISTERED",
        },
        "biometrics": {
            "biometric_hash": bio_hash,
        },
    }

    json_path = os.path.join(STORAGE_DIR, f"passport_{tag_id}.json")
    with open(json_path, "w") as f:
        json.dump(passport_data, f, indent=4)

    # Save cropped image preview
    cv2.imwrite(
        os.path.join(STORAGE_DIR, f"muzzle_{tag_id}.jpg"), cropped_img
    )

    return {
        "status": "SUCCESS",
        "message": "Digital Cattle Passport Successfully Registered!",
        "passport_id": f"PASSPORT-{tag_id}",
        "biometric_hash": bio_hash,
        "owner_wallet": owner_wallet,
        "json_storage_path": json_path,
    }
    