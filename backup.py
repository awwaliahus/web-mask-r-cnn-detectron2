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
from PIL import Image
import piexif

# Setup Flask application
app = Flask(__name__)

# Create an uploads folder if not exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Detector Class for handling the object detection
class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        # Load model config and pretrained model
        if model_type == "PS":  # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        elif model_type == "FT":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = "model_final_50m.pth"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # hanya 1 class: person

            # Penting! Pastikan dataset dan label classnya tepat
            self.cfg.DATASETS.TRAIN = ("person_train",)
            MetadataCatalog.get("person_train").thing_classes = ["person", "boat"]

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

        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",  # Adjust to your MySQL password
                database="mask_rcnn_database"
            )
            cursor = connection.cursor()

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

                        # Visualize detections on the frame
                        if len(high_confidence_instances) > 0:
                            metadata = MetadataCatalog.get("person_train")
                            v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
                            v = v.draw_instance_predictions(high_confidence_instances.to("cpu"))
                            result_image = v.get_image()[:, :, ::-1]

                            # Resize the result image to a smaller size
                            resized_result_image = cv2.resize(result_image, (600, 400))  # Adjust the dimensions as needed

                            # Show the resized frame with detections
                            cv2.imshow("Detection Results", resized_result_image)

                            if gps_data:
                                last_location = gps_data[min(frame_count, len(gps_data) - 1)]
                                latitude_match = last_location[0]

                                # Check for matching latitude in the database (5 decimal places)
                                cursor.execute("SELECT arah_angin FROM data_parameter WHERE ROUND(latitude, 5) = %s LIMIT 1",
                                               (round(latitude_match, 5),))
                                result = cursor.fetchone()

                                if result:
                                    arah_angin = result[0]
                                    print(f"Person and Location Detected at frame {frame_count}: Latitude = {last_location[0]}, Longitude = {last_location[1]}, arah_angin = {arah_angin}")
                                else:
                                    print(f"Person and Location Detected at frame {frame_count}: Latitude = {last_location[0]}, Longitude = {last_location[1]} (no matching arah_angin found)")

                # Increment frame counter
                frame_count += 1

                # Exit when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except mysql.connector.Error as err:
            print(f"Error: {err}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

        cap.release()
        cv2.destroyAllWindows()

        # Save last location to MySQL database
        if last_location:
            try:
                connection = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",  # Adjust to your MySQL password
                    database="mask_rcnn_database"
                )
                cursor = connection.cursor()

                # Get the IP address of the device that accessed the website
                ip_address = request.remote_addr

                # Add arah_angin to the database
                query = "INSERT INTO activitylog (latitude, longitude, user, arah_angin) VALUES (%s, %s, %s, %s)"
                cursor.execute(query, (last_location[0], last_location[1], ip_address, arah_angin if result else None))
                connection.commit()

                print("Last location, arah_angin, and user saved to the database.")
            except mysql.connector.Error as err:
                print(f"Error: {err}")
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()

# Function to extract GPS from EXIF data
def get_exif_gps(image_path):
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info["exif"])

        gps_data = exif_dict.get("GPS")
        if not gps_data:
            return None

        def convert_to_degrees(value):
            d, m, s = value
            return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

        lat = convert_to_degrees(gps_data[piexif.GPSIFD.GPSLatitude])
        lon = convert_to_degrees(gps_data[piexif.GPSIFD.GPSLongitude])
        if gps_data[piexif.GPSIFD.GPSLatitudeRef] == b'S':
            lat = -lat
        if gps_data[piexif.GPSIFD.GPSLongitudeRef] == b'W':
            lon = -lon

        return lat, lon
    except Exception as e:
        print(f"EXIF GPS error: {e}")
        return None

# Flask routes
@app.route('/detected_locations', methods=['GET'])
def detected_locations():
    try:
        # Get the IP address of the user
        ip_address = request.remote_addr

        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Your MySQL password
            database="mask_rcnn_database"
        )
        cursor = connection.cursor()

        # Query to get latitude, longitude, and arah_angin based on user's IP address
        cursor.execute("SELECT latitude, longitude, arah_angin FROM activitylog WHERE user = %s ORDER BY created_at DESC LIMIT 1", (ip_address,))
        data = cursor.fetchone()

        if data:
            latitude, longitude, arah_angin = data
            location = {
                "latitude": latitude,
                "longitude": longitude,
                "arah_angin": arah_angin if arah_angin else "Tidak Diketahui"
            }
            return jsonify({"locations": [location]})

        return jsonify({"locations": []})

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return jsonify({"error": str(err)}), 500
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

    return jsonify({"status": "done"})  # Return status to frontend

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({"error": "No selected image."}), 400

    # Save image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Initialize Detector
    detector = Detector(model_type="FT")

    # Run prediction
    img_cv = cv2.imread(image_path)
    predictions = detector.predictor(img_cv)
    instances = predictions["instances"]
    person_detected = False

    # Filter for person class (usually class 0 in COCO, check if FT model follows this)
    if len(instances) > 0:
        pred_classes = instances.pred_classes.tolist()
        if 0 in pred_classes:  # Assuming class 0 is 'person'
            person_detected = True

    if person_detected:
        gps = get_exif_gps(image_path)
        if gps:
            lat, lon = gps

            try:
                connection = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",  # Adjust as needed
                    database="mask_rcnn_database"
                )
                cursor = connection.cursor()

                # Cari arah_angin dari data_parameter
                cursor.execute("SELECT arah_angin FROM data_parameter WHERE ROUND(latitude, 5) = %s LIMIT 1", (round(lat, 5),))
                result = cursor.fetchone()
                arah_angin = result[0] if result else None

                ip_address = request.remote_addr
                query = "INSERT INTO activitylog (latitude, longitude, user, arah_angin) VALUES (%s, %s, %s, %s)"
                cursor.execute(query, (lat, lon, ip_address, arah_angin))
                connection.commit()

                print(f"Person detected at ({lat}, {lon})")

                return jsonify({
                    "status": "Person detected",
                    "latitude": lat,
                    "longitude": lon,
                    "arah_angin": arah_angin if arah_angin else "Tidak Diketahui"
                })

            except mysql.connector.Error as err:
                print(f"MySQL Error: {err}")
                return jsonify({"error": str(err)}), 500
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()

        else:
            print("Person detected, but no GPS data found.")
            return jsonify({"status": "Person detected", "gps": "not found"})
    else:
        print("No person detected.")
        return jsonify({"status": "No person detected"})

if __name__ == '__main__':
    app.run(debug=True)