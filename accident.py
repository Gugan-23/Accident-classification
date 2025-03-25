import cv2
import torch
import cloudinary
import cloudinary.uploader
import threading
import os
from flask import Flask, render_template, Response, request, jsonify
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pymongo import MongoClient
from datetime import datetime, timezone
from waitress import serve

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

class_names = ["Phone", "Laptop", "Tablet", "Computer", "Keyboard", "Washing machine", "Fridge"]

cloudinary.config(
    cloud_name="dlyohvur6",
    api_key="114323931176428",
    api_secret="KqlJkk3LdZL3MHzzt2FSiSkA-f0"
)

client = MongoClient("mongodb+srv://vgugan16:gugan2@cluster5.qyh1fuo.mongodb.net/?retryWrites=truew=majority&appName=Cluster0")
db = client["dL"]
video_collection = db["videos"]

cap = cv2.VideoCapture(0) 

def process_frame(frame):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_inputs = processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs).half()
        text_inputs = processor(text=class_names, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs).half()
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    sorted_indices = similarity[0].argsort(descending=True)
    detected_objects = [(class_names[idx], similarity[0][idx].item()) for idx in sorted_indices[:5]]

    return detected_objects

def draw_labels(frame, detected_objects):
    y_offset = 30
    for obj, confidence in detected_objects:
        label = f"{obj}: {confidence:.2f}"
        cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        detected_objects = process_frame(frame)
        frame = draw_labels(frame, detected_objects)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting.")
    serve(app, host="0.0.0.0", port=5000)
