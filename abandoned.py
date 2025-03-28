import os
import time
import json
from datetime import datetime
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
import threading
from collections import defaultdict
import cv2
from ultralytics import YOLO

# Configure window size
Window.size = (800, 600)

# Load KV files
Builder.load_file("home_screen.kv")
Builder.load_file("notifications_screen.kv")
Builder.load_file("logs_screen.kv")

class DetectionScreen(Screen):
    """Screen for live detection viewing"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.live_img = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.live_img)
        
        btn_layout = BoxLayout(size_hint_y=0.1, spacing=10, padding=10)
        back_btn = Button(
            text="BACK TO HOME",
            background_normal='',
            background_color=(0.2, 0.3, 0.4, 1),
            color=(1, 1, 1, 1))
        back_btn.bind(on_press=lambda x: App.get_running_app().back_to_home())
        btn_layout.add_widget(back_btn)
        
        layout.add_widget(btn_layout)
        self.add_widget(layout)
    
    def update_frame(self, frame_path):
        """Update the displayed frame"""
        self.live_img.source = frame_path
        self.live_img.reload()

class HomeScreen(Screen):
    pass

class NotificationsScreen(Screen):
    pass

class LogsScreen(Screen):
    LOGS_FILE = "threat_logs.json"
    IMAGE_DIR = "detected_frames"
    ALERT_SOUND = "alert.wav"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.IMAGE_DIR, exist_ok=True)
        self.alert_sound = self._load_sound()
        Clock.schedule_once(self._load_logs)

    def _load_sound(self):
        """Load the alert sound file"""
        try:
            if os.path.exists(self.ALERT_SOUND):
                sound = SoundLoader.load(self.ALERT_SOUND)
                if sound:
                    sound.volume = 0.8
                    return sound
        except Exception as e:
            print(f"Error loading sound: {e}")
        return None

    def play_alert(self):
        """Play the alert sound"""
        if self.alert_sound:
            try:
                self.alert_sound.play()
            except Exception as e:
                print(f"Error playing sound: {e}")

    def _load_logs(self, dt=None):
        """Load saved logs from file"""
        try:
            if os.path.exists(self.LOGS_FILE):
                with open(self.LOGS_FILE, 'r') as f:
                    logs = json.load(f)
                    for log in reversed(logs):
                        if os.path.exists(log['image_path']):
                            self._create_log_entry(log['image_path'], log['timestamp'], log.get('threat_type', 'VIOLENCE'))
        except Exception as e:
            print(f"Error loading logs: {e}")

    def _save_logs(self):
        """Save current logs to file"""
        logs = []
        for child in self.ids.logs_list.children:
            if hasattr(child, 'log_data'):
                logs.append(child.log_data)
        
        try:
            with open(self.LOGS_FILE, 'w') as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            print(f"Error saving logs: {e}")

    def add_log_entry(self, image_path, threat_type="VIOLENCE"):
        """Add a new threat entry to the logs"""
        if not os.path.exists(image_path):
            return
            
        self.play_alert()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._create_log_entry(image_path, timestamp, threat_type)
        self._show_alert(image_path, threat_type)
        self._save_logs()

    def _create_log_entry(self, image_path, timestamp, threat_type):
        """Create UI elements for a threat entry"""
        logs_list = self.ids.logs_list
        
        entry = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=100,
            spacing=10,
            padding=10
        )
        entry.log_data = {
            'image_path': image_path,
            'timestamp': timestamp,
            'threat_type': threat_type
        }
        
        img = Image(
            source=image_path,
            size_hint_x=0.3,
            allow_stretch=True,
            keep_ratio=True
        )
        img.bind(on_touch_down=lambda instance, touch: 
                self._show_large_image(image_path) if img.collide_point(*touch.pos) else False)
        
        details = Label(
            text=f"[b]{threat_type} DETECTED[/b]\n{timestamp}",
            markup=True,
            size_hint_x=0.7,
            halign="left",
            valign="middle"
        )
        
        entry.add_widget(img)
        entry.add_widget(details)
        logs_list.add_widget(entry)

    def _show_alert(self, image_path, threat_type):
        """Show initial alert popup"""
        content = BoxLayout(orientation="vertical", spacing=10)
        
        alert_img = Image(
            source=image_path,
            size_hint=(1, 0.7),
            allow_stretch=True,
            keep_ratio=True
        )
        
        alert_text = Label(
            text=f"[color=ff0000][b]🚨 {threat_type} DETECTED![/b][/color]",
            markup=True,
            size_hint=(1, 0.3)
        )
        
        content.add_widget(alert_img)
        content.add_widget(alert_text)
        
        Popup(
            title="SECURITY ALERT",
            content=content,
            size_hint=(0.8, 0.8),
            separator_color=(1, 0, 0, 1)
        ).open()

    def _show_large_image(self, image_path):
        """Show enlarged image in popup when clicked"""
        content = BoxLayout(orientation="vertical", spacing=10)
        
        img = Image(
            source=image_path,
            size_hint=(1, 0.9),
            allow_stretch=True,
            keep_ratio=True
        )
        
        btn = Button(
            text="CLOSE",
            size_hint=(1, 0.1),
            background_color=(0.2, 0.2, 0.2, 1)
        )
        
        popup = Popup(
            title="Threat Details",
            content=content,
            size_hint=(0.95, 0.95))
        
        btn.bind(on_press=popup.dismiss)
        content.add_widget(img)
        content.add_widget(btn)
        popup.open()

    def clear_logs(self):
        """Clear all logs from screen and storage"""
        confirm_popup = Popup(
            title="Confirm Clear Logs",
            size_hint=(0.6, 0.4))
        
        content = BoxLayout(orientation="vertical", spacing=10)
        content.add_widget(Label(text="Are you sure you want to clear all logs?"))
        
        btn_layout = BoxLayout(spacing=10)
        yes_btn = Button(text="YES")
        no_btn = Button(text="NO")
        
        def clear_and_close(instance):
            self.ids.logs_list.clear_widgets()
            try:
                if os.path.exists(self.LOGS_FILE):
                    os.remove(self.LOGS_FILE)
                for file in os.listdir(self.IMAGE_DIR):
                    os.remove(os.path.join(self.IMAGE_DIR, file))
            except Exception as e:
                print(f"Error clearing logs: {e}")
            confirm_popup.dismiss()
        
        yes_btn.bind(on_press=clear_and_close)
        no_btn.bind(on_press=confirm_popup.dismiss)
        
        btn_layout.add_widget(yes_btn)
        btn_layout.add_widget(no_btn)
        content.add_widget(btn_layout)
        confirm_popup.content = content
        confirm_popup.open()

class ThreatDetector:
    def __init__(self, callback):
        """Initialize YOLOv8 model for violence detection"""
        self.model = YOLO('best(7).pt')  # Your trained violence detection model
        self.violence_folder = "violence_frames"
        os.makedirs(self.violence_folder, exist_ok=True)
        self.callback = callback
        self.running = False
        self.cap = None

    def process_video(self, video_source=0):
        """
        Process video source (camera or file) with optimized parameters
        Args:
            video_source: Can be 0 for camera or file path for video file
        """
        self.running = True
        self.cap = cv2.VideoCapture(video_source)
        
        # Set camera properties if using webcam
        if video_source == 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        temp_dir = "temp_violence_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        frame_count = 0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * 0.2)) if fps > 0 else 1  # Process 5 frames/sec for video files
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Skip frames for performance (only for video files)
                if video_source != 0 and frame_count % frame_interval != 0:
                    frame_count += 1
                    continue
                
                # Adjust confidence based on input source
                conf_threshold = 0.76 if video_source != 0 else 0.7  # Higher confidence for camera
                results = self.model(frame, device="cpu", conf=conf_threshold)
                annotated_frame = results[0].plot()
                
                # Save frame for display
                display_path = os.path.join(temp_dir, f"live_{frame_count}.jpg")
                cv2.imwrite(display_path, annotated_frame)
                
                # Update UI display
                if hasattr(self, 'update_display'):
                    self.update_display(display_path)
                
                # Check for threats
                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) == 1 and box.conf[0] >= conf_threshold:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(
                                self.violence_folder, 
                                f"violence_{timestamp}_{frame_count}.jpg"
                            )
                            cv2.imwrite(img_path, annotated_frame)
                            self.callback(img_path)
                
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            self.running = False

    def stop_detection(self):
        """Stop any ongoing detection"""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

class AbandonedObjectDetector:
    def __init__(self, detection_callback):
        self.config = {
            "model": "yolov5m.pt",
            "classes": [26, 27, 25, 28, 39, 41],
            "abandoned_threshold": 30,
            "stationary_threshold": 5,
            "output_dir": "abandoned_objects",
            "tracker_config": "bytetrack.yaml"
        }
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.model = YOLO(self.config['model'])
        self.track_history = defaultdict(list)
        self.callback = detection_callback
        self.running = False

    def process_camera(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        temp_dir = "temp_abandoned_frames"
        os.makedirs(temp_dir, exist_ok=True)
        frame_count = 0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=self.config['classes'],
                    tracker=self.config['tracker_config']
                )
                
                annotated_frame = results[0].plot()
                display_path = os.path.join(temp_dir, f"live_{frame_count}.jpg")
                cv2.imwrite(display_path, annotated_frame)
                
                if hasattr(self, 'update_display'):
                    self.update_display(display_path)
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
                class_ids = results[0].boxes.cls.cpu().numpy()
                
                current_objects = set()
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box[:4])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    current_objects.add(track_id)
                    
                    if track_id not in self.track_history:
                        self.track_history[track_id] = {
                            'positions': [],
                            'class_id': class_id,
                            'first_seen': frame_count
                        }
                    
                    self.track_history[track_id]['positions'].append(center)
                    
                    if len(self.track_history[track_id]['positions']) > 10:
                        movements = [
                            ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                            for p1, p2 in zip(self.track_history[track_id]['positions'][-10:], 
                                            self.track_history[track_id]['positions'][-9:])
                        ]
                        
                        if all(m < self.config['stationary_threshold'] for m in movements):
                            stationary_frames = frame_count - self.track_history[track_id]['first_seen']
                            
                            if stationary_frames == self.config['abandoned_threshold']:
                                class_name = self.model.names[int(class_id)]
                                timestamp = int(time.time())
                                filename = f"abandoned_{class_name}_{track_id}_{timestamp}.jpg"
                                output_path = os.path.join(self.config['output_dir'], filename)
                                cv2.imwrite(output_path, annotated_frame)
                                self.callback(output_path)
                
                for track_id in list(self.track_history.keys()):
                    if track_id not in current_objects:
                        del self.track_history[track_id]
                
                frame_count += 1
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False

class MainApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.violence_detector = None
        self.abandoned_detector = None

    def build(self):
        """Initialize screen manager"""
        self.sm = ScreenManager()
        self.sm.add_widget(HomeScreen(name='home'))
        self.sm.add_widget(NotificationsScreen(name='notifications'))
        self.sm.add_widget(LogsScreen(name='logs'))
        
        # Add detection screens
        self.violence_screen = DetectionScreen(name='violence_live')
        self.abandoned_screen = DetectionScreen(name='abandoned_live')
        self.sm.add_widget(self.violence_screen)
        self.sm.add_widget(self.abandoned_screen)
        
        return self.sm

    def view_live_detections(self, video_source=0):
        """Start violence detection with live view or video file"""
        if self.violence_detector and self.violence_detector.running:
            return
            
        def detection_callback(img_path):
            Clock.schedule_once(lambda dt: self.sm.get_screen('logs').add_log_entry(img_path, "VIOLENCE"))
            
        self.violence_detector = ThreatDetector(detection_callback)
        self.violence_detector.update_display = lambda path: Clock.schedule_once(
            lambda dt: self.violence_screen.update_frame(path))
        
        self.sm.current = "violence_live"
        threading.Thread(
            target=lambda: self.violence_detector.process_video(video_source),
            daemon=True
        ).start()

    def process_video_file(self, video_path):
        """Process a video file for violence detection"""
        self.view_live_detections(video_path)

    def start_abandoned_detection(self):
        """Start abandoned object detection with live view"""
        if self.abandoned_detector and self.abandoned_detector.running:
            return
            
        def detection_callback(img_path):
            Clock.schedule_once(lambda dt: self.sm.get_screen('logs').add_log_entry(img_path, "ABANDONED OBJECT"))
        
        self.abandoned_detector = AbandonedObjectDetector(detection_callback)
        self.abandoned_detector.update_display = lambda path: Clock.schedule_once(
            lambda dt: self.abandoned_screen.update_frame(path))
        
        self.sm.current = "abandoned_live"
        threading.Thread(target=self.abandoned_detector.process_camera, daemon=True).start()

    def stop_abandoned_detection(self):
        """Stop abandoned object detection"""
        if self.abandoned_detector and self.abandoned_detector.running:
            self.abandoned_detector.running = False

    def stop_violence_detection(self):
        """Stop violence detection"""
        if self.violence_detector and self.violence_detector.running:
            self.violence_detector.running = False

    def show_notifications(self):
        self.sm.current = "notifications"

    def show_logs(self):
        self.sm.current = "logs"

    def show_settings(self):
        # Implement settings screen if needed
        pass

    def back_to_home(self):
        self.sm.current = "home"
        self.sm.transition.direction = 'right'

if __name__ == "__main__":
    MainApp().run()