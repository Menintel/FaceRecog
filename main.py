import cv2
from deepface import DeepFace
import os
import numpy as np
from scipy.spatial.distance import cosine
import time

# Create a directory to store captured images if it doesn't exist
if not os.path.exists("captured_faces"):
    os.makedirs("captured_faces")

def detect_faces_opencv(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces

def draw_face_rectangles(frame, faces):
    # Draw rectangles around detected faces
    for i, (x, y, w, h) in enumerate(faces):
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add face ID
        cv2.putText(frame, f"Face {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add face size
        cv2.putText(frame, f"{w}x{h}", (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return frame

def capture_and_verify():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set a larger window size
    cv2.namedWindow('Face Verification System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Verification System', 1000, 700)

    img_count = 0
    captured_enrollment_images = []
    enrollment_embeddings = []
    
    print("Welcome to the Face Verification System!")
    print("--- INSTRUCTIONS ---")
    print("1. Press 's' to capture enrollment images (5 needed)")
    print("2. After enrollment, press 'v' to start verification")
    print("3. Press 'q' to quit at any time")
    print("-" * 30)

    phase = "enrollment"  # Can be 'enrollment' or 'verification'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Detect faces in real-time using OpenCV
        try:
            # Detect faces using Haar Cascade
            faces = detect_faces_opencv(frame)
            
            # Draw face rectangles and info
            display_frame = draw_face_rectangles(display_frame.copy(), faces)
            
            # Add face count
            cv2.putText(display_frame, f"Faces: {len(faces)}", (10, 110), 
                      cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
            
        except Exception as e:
            print(f"Face detection error: {e}")
            # If there's an error, just show the frame without face detection
        
        # Add instructions to the frame
        if phase == "enrollment":
            status_text = f"Enrollment: {len(captured_enrollment_images)}/5 images captured"
            instruction_text = "Press 's' to capture | 'v' to verify | 'q' to quit"
            
            # Draw status with a more professional font
            cv2.putText(display_frame, "ENROLLMENT", (10, 30), 
                       cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, status_text, (10, 55), 
                       cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, instruction_text, (10, 80), 
                       cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
        elif phase == "verification":  
            cv2.putText(display_frame, "VERIFICATION", (10, 30), 
                       cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "Press 'v' to verify | 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Show the frame with instructions
        cv2.imshow('Face Verification System', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if phase == "enrollment" and key == ord('s'):
            if len(captured_enrollment_images) >= 5:
                print("Already captured 5 images. Press 'v' to start verification.")
                continue
                
            img_name = f"captured_faces/enrollment_face_{img_count}.jpg"
            cv2.imwrite(img_name, frame)
            
            try:
                # Try to detect face and get embedding
                embedding = DeepFace.represent(img_name, enforce_detection=True)[0]["embedding"]
                captured_enrollment_images.append(img_name)
                enrollment_embeddings.append(embedding)
                print(f"Image {len(captured_enrollment_images)} captured and processed successfully.")
                img_count += 1
                
                # If we have 5 images, calculate average embedding
                if len(captured_enrollment_images) == 5:
                    average_embedding = np.mean(enrollment_embeddings, axis=0)
                    print("\nEnrollment complete! You can now verify faces.")
                    print("Press 'v' to start verification.")
                    
            except Exception as e:
                print(f"Error processing image: {e}")
                print("Please ensure a face is clearly visible and try again.")
                if os.path.exists(img_name):
                    os.remove(img_name)
        
        elif phase == "enrollment" and key == ord('v'):
            if len(captured_enrollment_images) < 5:
                print(f"Please capture {5 - len(captured_enrollment_images)} more images before verification.")
            else:
                average_embedding = np.mean(enrollment_embeddings, axis=0)
                phase = "verification"
                print("\n--- VERIFICATION MODE ---")
                print("Press 'v' to verify the current frame.")
                
        elif phase == "verification" and key == ord('v'):
            verification_img_name = "captured_faces/verification_face.jpg"
            cv2.imwrite(verification_img_name, frame)
            
            try:
                # Get embedding for verification
                verification_result = DeepFace.represent(verification_img_name, enforce_detection=True)[0]["embedding"]
                
                # Calculate distance
                distance = cosine(average_embedding, verification_result)
                threshold = 0.25  # Adjust threshold as needed
                
                # Display result
                result_text = "SAME PERSON" if distance < threshold else "DIFFERENT PERSON"
                color = (0, 255, 0) if distance < threshold else (0, 0, 255)
                
                # Add result to the frame
                cv2.putText(display_frame, f"Result: {result_text} (Score: {distance:.4f})", 
                           (20, 120), cv2.FONT_HERSHEY_PLAIN, 0.6, color, 1)
                
                print(f"Verification result: {result_text} (Distance: {distance:.4f}, Threshold: {threshold})")
                
                # Show the result for 2 seconds
                cv2.imshow('Face Verification System', display_frame)
                cv2.waitKey(2000)
                
            except Exception as e:
                print(f"Verification error: {e}")
                
        elif key == ord('q'):
            print("Quitting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        capture_and_verify()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()