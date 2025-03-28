"""import cv2
import os
import time
from ultralytics import YOLO

class ThreatDetector:
    def __init__(self, callback):
        
        
        
        
    
        self.model = YOLO('best(7).pt')
        self.frames_folder = "frames"
        self.violence_folder = "violence_frames"
        os.makedirs(self.frames_folder, exist_ok=True)
        os.makedirs(self.violence_folder, exist_ok=True)
        self.callback = callback  # Kivy callback function to update UI

    def process_video(self, video_path):
    
      

        
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_count = 0
        last_save_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if time.time() - last_save_time >= 0.2:  # Process frame every 0.2 sec
                last_save_time = time.time()
                frame_count += 1

                # Run YOLOv8 inference
                results = self.model(frame, device="cpu")

                detected_violence = False
                image_path = None

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = box.conf[0].item()

                        if class_id == 1 and confidence >= 0.8:  # Violence detected
                            detected_violence = True
                            image_path = os.path.join(self.violence_folder, f"violence_{frame_count}.jpg")
                            cv2.imwrite(image_path, frame)
                            print(f"📸 Threat detected: {image_path}")

                            if self.callback:
                                self.callback(image_path)  # Send image path to Kivy UI

                if detected_violence:
                    continue  # Skip non-violence cases

            # Show real-time detection
            cv2.imshow('YOLOv8 Threat Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(" Video Processing Complete")"""

import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

class ThreatDetector:
    def __init__(self, callback):
        """Initialize YOLOv8 model with optimized settings for camera input."""
        self.model = YOLO('best(7).pt')  # Your trained model
        self.violence_folder = "violence_frames"
        os.makedirs(self.violence_folder, exist_ok=True)
        self.callback = callback
        self.avg_frame = None
        self.frame_count = 0
        self.last_detection_time = 0
        self.detection_cooldown = 2  # seconds between detections

    def process_camera(self):
        """Process camera feed with optimized detection parameters."""
        print(" Starting camera threat detection...")
        
        # Camera setup with optimal parameters
        cap = cv2.VideoCapture("fighttest.mp4")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
        
        # CLAHE for lighting normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(" Camera frame read error")
                    break
                
                self.frame_count += 1
                
                # Skip frames for performance (process every 3rd frame)
                if self.frame_count % 3 != 0:
                    continue
                
                # Pre-processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = clahe.apply(gray)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                # Motion detection
                if self.avg_frame is None:
                    self.avg_frame = gray.copy().astype("float")
                    continue
                
                cv2.accumulateWeighted(gray, self.avg_frame, 0.5)
                frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_frame))
                
                # Skip static frames
                if cv2.countNonZero(frame_delta) < 500:
                    continue
                
                # Run detection with higher confidence threshold
                results = self.model(frame, 
                                   device="cpu",
                                   conf=0.75,  # Higher confidence threshold
                                   iou=0.5,    # NMS IoU threshold
                                   imgsz=640)  # Optimal input size
                
                annotated_frame = results[0].plot()
                
                # Process detections with cooldown
                current_time = time.time()
                for result in results:
                    for box in result.boxes:
                        if (int(box.cls[0]) == 1 and  # Violence class
                           box.conf[0] >= 0.85 and    # Confidence threshold
                           (current_time - self.last_detection_time) > self.detection_cooldown):
                            
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(
                                self.violence_folder, 
                                f"violence_{timestamp}_{self.frame_count}.jpg"
                            )
                            cv2.imwrite(img_path, annotated_frame)
                            print(f"📸 Valid threat detected: {img_path}")
                            self.last_detection_time = current_time
                            
                            if self.callback:
                                self.callback(img_path)
                
                # Display output
                cv2.imshow('Threat Detection (Press Q to quit)', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera processing stopped")

    def process_video(self, video_path):
        """Alternative method for video files with different parameters."""
        print(f"🔍 Processing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f" Error: Could not open video {video_path}")
            return

        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * 0.2))  # Process 5 frames/sec

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval != 0:
                frame_count += 1
                continue

            # Use different (lower) thresholds for video files
            results = self.model(frame, device="cpu", conf=0.65, iou=0.45)
            annotated_frame = results[0].plot()
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 1 and box.conf[0] >= 0.76:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(
                            self.violence_folder, 
                            f"violence_{timestamp}_{frame_count}.jpg"
                        )
                        cv2.imwrite(img_path, annotated_frame)
                        
                        if self.callback:
                            self.callback(img_path)

            cv2.imshow('Video Threat Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print(" Video processing complete")