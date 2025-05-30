from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import re
import mysql.connector

app = Flask(__name__)

# Create uploads folder if not exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class Detector:
    def __init__(self, model_type="FT"):
        self.cfg = get_cfg()
        self.model_type = model_type

        if model_type == "FT":  # Fine-Tuned model
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = "model_final.pth"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
            MetadataCatalog.get("person_train").thing_classes = ["person"]
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Detect only the 'person' class

        self.cfg.MODEL.DEVICE = "cpu"  # Set to "cpu" or "cuda"
        self.predictor = DefaultPredictor(self.cfg)

    def extract_gps_from_srt(self, srt_path):
        gps_data = []
        with open(srt_path, 'r') as file:
            content = file.read()
            pattern = r"\[latitude: ([\-0-9.]+)\] \[longitude: ([\-0-9.]+)\]"
            matches = re.findall(pattern, content)

            for match in matches:
                latitude = float(match[0])
                longitude = float(match[1])
                gps_data.append((latitude, longitude))

        return gps_data

    def onVideo(self, video_path, srt_path):
        gps_data = self.extract_gps_from_srt(srt_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening the video file.")
            return []

        detected_locations = []
        frame_count = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                outputs = self.predictor(frame)
                instances = outputs["instances"]

                if len(instances) > 0 and gps_data:
                    detected_locations.append(gps_data[min(frame_count // frame_interval, len(gps_data) - 1)])

            frame_count += 1

        cap.release()
        return detected_locations

@app.route('/')
def coba():
    return render_template('coba.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'video' not in request.files or 'subtitle' not in request.files:
        return jsonify({"error": "Files are missing"}), 400

    video = request.files['video']
    subtitle = request.files['subtitle']

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    subtitle_path = os.path.join(app.config['UPLOAD_FOLDER'], subtitle.filename)
    video.save(video_path)
    subtitle.save(subtitle_path)

    detector = Detector(model_type="FT")
    detected_locations = detector.onVideo(video_path, subtitle_path)

    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Adjust your MySQL password
            database="mask_rcnn_database"
        )
        cursor = connection.cursor()
        for lat, lng in detected_locations:
            cursor.execute(
                "INSERT INTO detected_locations (latitude, longitude) VALUES (%s, %s)",
                (lat, lng)
            )
        connection.commit()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return jsonify({"message": "Files processed successfully", "locations": detected_locations})

@app.route('/detected_locations', methods=['GET'])
def get_detected_locations():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Adjust your MySQL password
            database="mask_rcnn_database"
        )
        cursor = connection.cursor()
        cursor.execute("SELECT latitude, longitude FROM detected_locations")
        locations = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        locations = []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return jsonify(locations)

if __name__ == '__main__':
    app.run(debug=True)
