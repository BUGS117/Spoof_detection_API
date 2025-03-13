from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import os
import io
import uvicorn
import cv2

# Import YOLOv8 model from Ultralytics
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Initialize YOLOv8 model for face detection
yolo_model = YOLO("yolov8x.pt")  # Use the YOLOv8x model

# Directory to store face embeddings
db_dir = "db_faces/"

def checkCollection(collectionId, readOnly=True):
    try:
        if not collectionId:
            return False, None
        db_collection = f"{db_dir}{collectionId}"
        exist = os.path.exists(db_collection)
        if not readOnly:
            if not exist:
                os.makedirs(db_collection)
                exist = os.path.exists(db_collection)
        return exist, db_collection
    except Exception as e:
        return False, None

def preprocess_image_yolo(image_bytes):
    """
    Preprocess image using YOLOv8 to detect faces.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)

        # Perform YOLOv8 inference
        results = yolo_model(np_image)
        if not results or len(results[0].boxes) == 0:
            return None  # No faces detected

        # Crop the first detected face
        face_box = results[0].boxes[0].xyxy[0].numpy().astype(int)  # Get bounding box
        x1, y1, x2, y2 = face_box
        cropped_face = np_image[y1:y2, x1:x2]
        return cropped_face  # Return cropped face image
    except Exception as e:
        print("Error in preprocess_image_yolo:", e)
        return None

def embedding_image(face):
    """
    Generate embeddings using InsightFace for a detected face.
    """
    try:
        if face is None:
            return False, None

        # Initialize InsightFace model
        face_analysis = FaceAnalysis()
        face_analysis.prepare(ctx_id=0)

        # Perform face analysis to get embedding
        results = face_analysis.get(np.array(face))
        if not results:
            return False, None

        embedding = results[0]["embedding"]
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return True, embedding
    except Exception as e:
        print("Error generating embedding:", e)
        return False, None

@app.post("/register")
async def register_face(collectionId: str = Form(None), name: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()

        # Step 1: Check collection and preprocess face with YOLOv8
        exist, db_collection = checkCollection(collectionId, readOnly=False)
        if not exist:
            return JSONResponse(content={"status": False, "message": "Collection not found", "data": None})

        face = preprocess_image_yolo(image_bytes)
        if face is None:
            return JSONResponse(content={"status": False, "message": "No face detected in the image", "data": None})

        valid, embedding = embedding_image(face)
        if not valid:
            return JSONResponse(content={"status": False, "message": "Failed to generate face embeddings", "data": None})

        # Save embedding to database
        np.save(os.path.join(db_collection, f"{name}.npy"), embedding)
        return JSONResponse(content={"status": True, "message": "Success", "data": {"id": name}})
    except Exception as e:
        return JSONResponse(content={"status": False, "message": str(e), "data": None})

@app.post("/recognize")
async def recognize_face(collectionId: str = Form(None), image: UploadFile = File(...), accuracy: float = Form(0.8)):
    try:
        image_bytes = await image.read()

        # Step 1: Check collection and preprocess face with YOLOv8
        exist, db_collection = checkCollection(collectionId)
        if not exist:
            return JSONResponse(content={"status": False, "message": "Collection not found", "data": None})

        face = preprocess_image_yolo(image_bytes)
        if face is None:
            return JSONResponse(content={"status": False, "message": "No face detected in the image", "data": None})

        valid, embedding = embedding_image(face)
        if not valid:
            return JSONResponse(content={"status": False, "message": "Failed to generate face embeddings", "data": None})

        # Step 2: Compare with database
        max_similarity = -1
        identity = None
        filelist = os.listdir(db_collection)
        for file in filelist:
            saved_embedding = np.load(os.path.join(db_collection, file))
            similarity = np.dot(saved_embedding, embedding)  # Cosine similarity
            if similarity > max_similarity and similarity > accuracy:
                max_similarity = similarity
                identity = os.path.splitext(file)[0]

        if identity:
            return JSONResponse(content={"status": True, "message": "Success", "data": {"id": identity, "accuracy": f"{max_similarity:.2f}"}})
        else:
            return JSONResponse(content={"status": False, "message": "Not registered", "data": None})
    except Exception as e:
        return JSONResponse(content={"status": False, "message": str(e), "data": None})

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="172.20.10.2", port=1200)  # Replace with your IP
