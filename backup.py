from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os
import re
import mysql.connector
from getpass import getuser  # Untuk mendapatkan default username perangkat

app = Flask(__name__)

# Create an uploads folder if not exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        # Load model config and pretrained model  
        if model_type == "PS":  # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
            
        elif model_type == "FT":  # Fine-Tuned
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = "model_final.pth"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
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

    def onVideo(self, videoPath, srtPath):
        gps_data = self.extract_gps_from_srt(srtPath)
        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error opening the file...")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps
        frame_count = 0
        last_location = None

        while True:
            success, image = cap.read()
            if not success:
                print("End of video or error reading frame.")
                break

            if frame_count % frame_interval == 0:
                if self.model_type != "PS":
                    predictions = self.predictor(image)
                    instances = predictions["instances"]
                    high_confidence_instances = instances[instances.scores >= 0.9]

                    if len(high_confidence_instances) > 0 and gps_data:
                        last_location = gps_data[min(frame_count, len(gps_data)-1)]
                        print(f"Person and Location Detected at frame {frame_count}: Latitude = {last_location[0]}, Longitude = {last_location[1]}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Save last location to MySQL database
        if last_location:
            try:
                connection = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",  # Sesuaikan dengan password MySQL Anda
                    database="activity_log_mask-rcnn"
                )
                cursor = connection.cursor()

                default_user = getuser()
                query = "INSERT INTO activitylog (lokasi_awal, user) VALUES (%s, %s)"
                cursor.execute(query, (f"{last_location[0]}, {last_location[1]}", default_user))
                connection.commit()

                print("Last location and user saved to the database.")
            except mysql.connector.Error as err:
                print(f"Error: {err}")
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'video' not in request.files or 'subtitle' not in request.files:
        return redirect(request.url)

    video = request.files['video']
    subtitle = request.files['subtitle']
    
    if video.filename == '' or subtitle.filename == '':
        return redirect(request.url)

    if video and subtitle:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        subtitle_path = os.path.join(app.config['UPLOAD_FOLDER'], subtitle.filename)
        video.save(video_path)
        subtitle.save(subtitle_path)

        # Initialize detector and run on video
        detector = Detector(model_type="FT")  # Specify model type as needed
        detector.onVideo(video_path, subtitle_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)