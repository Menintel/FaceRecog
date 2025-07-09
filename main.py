import cv2
from deepface import DeepFace
import os
import numpy as np
from scipy.spatial.distance import cosine
import time
import json
import threading
from datetime import datetime
import logging

# ********************
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceVerificationSystem:
    def __init__(self):
        self.enrollment_dir = "captured_faces"
        self.config_file = "system_config.json"
        self.log_file = "verification_log.txt"
        
        # ********************
        # Enhanced configuration
        self.config = {
            "enrollment_images_required": 5,
            "verification_threshold": 0.25,
            "face_detection_model": "opencv",  # or "mtcnn", "retinaface"
            "face_recognition_model": "VGG-Face",  # or "Facenet", "OpenFace", "DeepFace"
            "min_face_size": (80, 80),
            "max_face_size": (400, 400),
            "quality_threshold": 0.7
        }
        
        # ********************
        # Initialize directories and files
        self.setup_directories()
        self.load_config()
        
        # ********************
        # System state
        self.enrollment_embeddings = []
        self.captured_enrollment_images = []
        self.average_embedding = None
        self.phase = "enrollment"
        self.last_verification_time = 0
        self.verification_cooldown = 1.0  # seconds
        
        # ********************
        # Enhanced face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def setup_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.enrollment_dir):
            os.makedirs(self.enrollment_dir)
            
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading config: {e}")
                
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            
    def log_verification_attempt(self, result, distance, success=True):
        """Log verification attempts"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - Result: {result}, Distance: {distance:.4f}, Success: {success}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Error writing to log: {e}")
            
    def assess_face_quality(self, face_region):
        """********************
        Assess the quality of detected face"""
        try:
            # Check for blur using Laplacian variance
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Check brightness
            brightness = np.mean(gray_face)
            
            # Check contrast
            contrast = gray_face.std()
            
            # Quality score (normalized)
            quality_score = min(1.0, (laplacian_var / 100) * (contrast / 50) * (brightness / 128))
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing face quality: {e}")
            return 0.0
            
    def detect_faces_enhanced(self, frame):
        """********************
        Enhanced face detection with quality assessment"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.config["min_face_size"],
            maxSize=self.config["max_face_size"],
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # ********************
        # Filter faces by quality and eye detection
        quality_faces = []
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            
            # Assess quality
            quality = self.assess_face_quality(face_region)
            
            # Check for eyes (indicates proper face orientation)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                maxSize=(100, 100)
            )
            
            if quality >= self.config["quality_threshold"] and len(eyes) >= 2:
                quality_faces.append((x, y, w, h, quality))
                
        return quality_faces
        
    def draw_enhanced_face_info(self, frame, faces):
        """********************
        Draw enhanced face information"""
        for i, (x, y, w, h, quality) in enumerate(faces):
            # Color based on quality
            if quality >= 0.8:
                color = (0, 255, 0)  # Green - excellent
            elif quality >= 0.6:
                color = (0, 255, 255)  # Yellow - good
            else:
                color = (0, 128, 255)  # Orange - fair
                
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Face info
            cv2.putText(frame, f"Face {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Quality: {quality:.2f}", (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, f"{w}x{h}", (x, y+h+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                       
        return frame
        
    def process_enrollment_image(self, frame, img_count):
        """********************
        Process enrollment image with enhanced error handling"""
        img_name = f"{self.enrollment_dir}/enrollment_face_{img_count}.jpg"
        
        try:
            # Save image
            cv2.imwrite(img_name, frame)
            
            # Get embedding with multiple attempts
            embedding = None
            for model in ["VGG-Face", "Facenet", "OpenFace"]:
                try:
                    result = DeepFace.represent(
                        img_name, 
                        model_name=model,
                        enforce_detection=True,
                        detector_backend="opencv"
                    )
                    embedding = result[0]["embedding"]
                    break
                except Exception as e:
                    logger.warning(f"Failed with {model}: {e}")
                    continue
                    
            if embedding is None:
                raise Exception("All face recognition models failed")
                
            self.captured_enrollment_images.append(img_name)
            self.enrollment_embeddings.append(embedding)
            
            logger.info(f"Enrollment image {len(self.captured_enrollment_images)} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing enrollment image: {e}")
            if os.path.exists(img_name):
                os.remove(img_name)
            return False
            
    def verify_face(self, frame):
        """********************
        Enhanced face verification with multiple models"""
        if self.average_embedding is None:
            return False, "No enrollment data available"
            
        current_time = time.time()
        if current_time - self.last_verification_time < self.verification_cooldown:
            return False, "Verification cooldown active"
            
        self.last_verification_time = current_time
        
        verification_img_name = f"{self.enrollment_dir}/verification_face.jpg"
        cv2.imwrite(verification_img_name, frame)
        
        try:
            # Try multiple models for verification
            verification_embedding = None
            for model in ["VGG-Face", "Facenet", "OpenFace"]:
                try:
                    result = DeepFace.represent(
                        verification_img_name,
                        model_name=model,
                        enforce_detection=True,
                        detector_backend="opencv"
                    )
                    verification_embedding = result[0]["embedding"]
                    break
                except Exception as e:
                    logger.warning(f"Verification failed with {model}: {e}")
                    continue
                    
            if verification_embedding is None:
                return False, "Face recognition failed"
                
            # Calculate distance
            distance = cosine(self.average_embedding, verification_embedding)
            threshold = self.config["verification_threshold"]
            
            is_same_person = distance < threshold
            result_text = "SAME PERSON" if is_same_person else "DIFFERENT PERSON"
            
            # Log the attempt
            self.log_verification_attempt(result_text, distance, True)
            
            return True, f"{result_text} (Score: {distance:.4f})"
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            self.log_verification_attempt("ERROR", 0.0, False)
            return False, f"Verification error: {str(e)}"
            
    def draw_ui(self, frame):
        """********************
        Enhanced UI with better information display"""
        # Create overlay
        overlay = frame.copy()
        
        # Status panel
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 30
        if self.phase == "enrollment":
            cv2.putText(frame, "ENROLLMENT PHASE", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
            status = f"Images captured: {len(self.captured_enrollment_images)}/{self.config['enrollment_images_required']}"
            cv2.putText(frame, status, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(frame, "Press 'S' to capture | 'V' to verify | 'Q' to quit", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(frame, "Press 'R' to reset | 'C' to configure", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                       
        elif self.phase == "verification":
            cv2.putText(frame, "VERIFICATION PHASE", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            
            threshold_text = f"Threshold: {self.config['verification_threshold']:.3f}"
            cv2.putText(frame, threshold_text, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(frame, "Press 'V' to verify | 'E' to re-enroll | 'Q' to quit", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                       
        return frame
        
    def reset_enrollment(self):
        """********************
        Reset enrollment data"""
        self.enrollment_embeddings = []
        self.captured_enrollment_images = []
        self.average_embedding = None
        self.phase = "enrollment"
        
        # Clean up old enrollment images
        for file in os.listdir(self.enrollment_dir):
            if file.startswith("enrollment_face_"):
                os.remove(os.path.join(self.enrollment_dir, file))
                
        logger.info("Enrollment data reset")
        
    def run(self):
        """********************
        Main application loop with enhanced error handling"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open webcam.")
            return
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create window
        cv2.namedWindow('Enhanced Face Verification System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Enhanced Face Verification System', 1200, 800)
        
        img_count = 0
        verification_result = ""
        result_display_time = 0
        
        logger.info("Face Verification System started")
        print("=== ENHANCED FACE VERIFICATION SYSTEM ===")
        print("Controls:")
        print("  S - Capture enrollment image")
        print("  V - Verify face")
        print("  R - Reset enrollment")
        print("  E - Switch to enrollment")
        print("  C - Configure settings")
        print("  Q - Quit")
        print("=" * 40)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            try:
                # Enhanced face detection
                faces = self.detect_faces_enhanced(frame)
                display_frame = self.draw_enhanced_face_info(display_frame, faces)
                
                # Face count
                cv2.putText(display_frame, f"High-quality faces: {len(faces)}", 
                           (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                           
            except Exception as e:
                logger.error(f"Face detection error: {e}")
                
            # Draw UI
            display_frame = self.draw_ui(display_frame)
            
            # Display verification result
            if verification_result and time.time() - result_display_time < 3:
                cv2.putText(display_frame, verification_result, (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif time.time() - result_display_time >= 3:
                verification_result = ""
                
            cv2.imshow('Enhanced Face Verification System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('s') or key == ord('S'):
                if self.phase == "enrollment":
                    if len(self.captured_enrollment_images) >= self.config["enrollment_images_required"]:
                        print("Enrollment complete! Press 'V' to verify.")
                        continue
                        
                    if len(faces) > 0:
                        if self.process_enrollment_image(frame, img_count):
                            img_count += 1
                            print(f"Captured {len(self.captured_enrollment_images)}/{self.config['enrollment_images_required']} images")
                            
                            if len(self.captured_enrollment_images) == self.config["enrollment_images_required"]:
                                self.average_embedding = np.mean(self.enrollment_embeddings, axis=0)
                                print("Enrollment complete! Press 'V' to start verification.")
                        else:
                            print("Failed to process image. Please try again.")
                    else:
                        print("No high-quality face detected. Please position yourself properly.")
                        
            elif key == ord('v') or key == ord('V'):
                if self.phase == "enrollment":
                    if len(self.captured_enrollment_images) >= self.config["enrollment_images_required"]:
                        self.average_embedding = np.mean(self.enrollment_embeddings, axis=0)
                        self.phase = "verification"
                        print("Switched to verification mode.")
                    else:
                        print(f"Need {self.config['enrollment_images_required'] - len(self.captured_enrollment_images)} more images.")
                        
                elif self.phase == "verification":
                    if len(faces) > 0:
                        success, result = self.verify_face(frame)
                        if success:
                            verification_result = result
                            result_display_time = time.time()
                            print(f"Verification: {result}")
                        else:
                            print(f"Verification failed: {result}")
                    else:
                        print("No high-quality face detected for verification.")
                        
            elif key == ord('r') or key == ord('R'):
                self.reset_enrollment()
                print("Enrollment data reset.")
                
            elif key == ord('e') or key == ord('E'):
                self.phase = "enrollment"
                print("Switched to enrollment mode.")
                
            elif key == ord('c') or key == ord('C'):
                print("Current configuration:")
                for key, value in self.config.items():
                    print(f"  {key}: {value}")
                    
            elif key == ord('q') or key == ord('Q'):
                print("Quitting...")
                break
                
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.save_config()
        logger.info("Application closed")

if __name__ == "__main__":
    try:
        system = FaceVerificationSystem()
        system.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        cv2.destroyAllWindows()