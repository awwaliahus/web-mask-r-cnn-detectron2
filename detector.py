from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        # Load model config and pretrained model
        if model_type == "OD":  # Object Detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
            
        elif model_type == "IS":  # Instance Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            
        elif model_type == "KP":  # Keypoint Detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
            
        elif model_type == "LVIS":  # LVIS Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
            
        elif model_type == "PS":  # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
            
        elif model_type == "FT":  # Fine-Tuned
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = "model_final.pth"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"  # Set to "cpu" or "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        if self.model_type!= "PS":
            predictions = self.predictor(image)
        
            viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get("person_train"),
            instance_mode = ColorMode.IMAGE)
            
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get("person_train"))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
        
        # Resize the image to fit the window while keeping the aspect ratio
        output_image = output.get_image()[:,:,::-1]

        # Set desired window size
        window_width = 800
        window_height = 600

        # Calculate the aspect ratio of the image
        aspect_ratio = output_image.shape[1] / output_image.shape[0]

        # Resize image to fit within the window while maintaining aspect ratio
        if aspect_ratio > 1:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)
        else:
            new_height = window_height
            new_width = int(window_height * aspect_ratio)

        resized_image = cv2.resize(output_image, (new_width, new_height))

        # Show the resized image in the window
        cv2.imshow("Result", resized_image)

        # Resize the window accordingly
        cv2.resizeWindow("Result", new_width, new_height)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
    
        if not cap.isOpened():
            print("Error opening the file...")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        frame_interval = fps  # Number of frames to skip (1 second)
        
        frame_count = 0  # Track current frame count
        
        while True:
            # Read the current frame
            success, image = cap.read()
            
            # Break the loop if the video has ended
            if not success:
                print("End of video or error reading frame.")
                break
            
            # Process the frame only at each 1-second interval
            if frame_count % frame_interval == 0:
                if self.model_type != "PS":
                    predictions = self.predictor(image)
                    
                    viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get("person_train"),
                                    instance_mode=ColorMode.IMAGE)
                    
                    output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
                else:
                    predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                    viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get("person_train"))
                    output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
            
                # Display the processed frame
                cv2.imshow("Result", output.get_image()[:, :, ::-1])
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            # Increment the frame count
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

