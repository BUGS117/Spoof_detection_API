from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import os
import io
import uvicorn
import cv2

# Import SpoofDetector class
from spoof_detector import SpoofDetector

# Initialize FastAPI app
app = FastAPI()

# Initialize InsightFace model
app_insightface = FaceAnalysis(name="buffalo_l") # use any one of the model antelopev2 or buffalo_l
app_insightface.prepare(ctx_id=-1, det_size=(640, 640)) 

# Initialize SpoofDetector
spoof_detector = SpoofDetector(model_dir="./resources/anti_spoof_models", device_id=0) # Use any one of the model 2.7_80x80_MiniFASNetV2.pth or 4_0_0_80x80_MiniFASNetV1SE.pth

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

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)
        faces = app_insightface.get(np_image)
        if not faces:
            return None
        return faces[0]  # Return the first detected face
    except Exception as e:
        return None

def embedding_image(face):
    try:
        if face is None:
            return False, None
        embedding = face["embedding"]
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return True, embedding
    except Exception as e:
        return False, None

@app.post("/register")
async def register_face(collectionId: str = Form(None), name: str = Form(...), image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()

        # Step 1: Spoof detection
        np_image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        spoof_result = spoof_detector.predict(img)
        if not spoof_result["status"] or spoof_result["label"] == "Fake Face":
            return JSONResponse(content={"status": False, "message": "Image detected as spoof", "data": None})

        # Step 2: Check collection and preprocess face
        exist, db_collection = checkCollection(collectionId, readOnly=False)
        if not exist:
            return JSONResponse(content={"status": False, "message": "Collection not found", "data": None})

        face = preprocess_image(image_bytes)
        valid, embedding = embedding_image(face)
        if not valid:
            return JSONResponse(content={"status": False, "message": "No face detected in the image", "data": None})

        # Save embedding to database
        np.save(os.path.join(db_collection, f"{name}.npy"), embedding)
        return JSONResponse(content={"status": True, "message": "Success", "data": {"id": name}})
    except Exception as e:
        return JSONResponse(content={"status": False, "message": str(e), "data": None})

@app.post("/recognize")
async def recognize_face(collectionId: str = Form(None), image: UploadFile = File(...), accuracy: float = Form(0.8)):
    try:
        image_bytes = await image.read()

        # Step 1: Spoof detection
        np_image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        spoof_result = spoof_detector.predict(img)
        if not spoof_result["status"] or spoof_result["label"] == "Fake Face":
            return JSONResponse(content={"status": False, "message": "Image detected as spoof", "data": None})

        # Step 2: Check collection and preprocess face
        exist, db_collection = checkCollection(collectionId)
        if not exist:
            return JSONResponse(content={"status": False, "message": "Collection not found", "data": None})

        face = preprocess_image(image_bytes)
        valid, embedding = embedding_image(face)
        if not valid:
            return JSONResponse(content={"status": False, "message": "No face detected in the image", "data": None})

        # Step 3: Compare with database
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
    uvicorn.run(app, host="192.168.0.160", port=1200) #put your own ip 

 