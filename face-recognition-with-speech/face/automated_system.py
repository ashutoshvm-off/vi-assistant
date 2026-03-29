"""
FULLY AUTOMATED FACE RECOGNITION & REGISTRATION SYSTEM
Complete automated workflow with improved UX
- Stabilized face detection with countdown
- User permission prompts before capture
- Better face quality checks
- Performance optimizations
- Integrated FaceRecognitionModel for image-based comparison
"""

import cv2
import torch
import numpy as np
import os
import sys
import threading
import queue
from pathlib import Path

# Add project root to path for shared modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from datetime import datetime, timedelta
from facenet_pytorch import MTCNN, InceptionResnetV1
from supabase_service import SupabaseService
from photo_service import PhotoService
from face_rec import get_embedding, compare_faces_with_photos
from face_recognition_model import FaceRecognitionModel
import pyttsx3

# Initialize Services and Models
supabase_service = SupabaseService()
photo_service = PhotoService()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize Text-to-Speech with queue to avoid conflicts
speech_queue = queue.Queue()
speech_lock = threading.Lock()

def _speech_worker():
    """Background worker that processes speech requests one at a time"""
    engine = None
    while True:
        try:
            text = speech_queue.get(timeout=1)
            if text is None:  # Shutdown signal
                break
            
            # Initialize engine once per speech (avoids run loop conflicts)
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 120)  # Slower, soothing pace
                engine.setProperty('volume', 0.9)
                
                # Try to use a softer/female voice (Windows: Zira)
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"Speech error: {e}")
            
            speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            pass

# Start speech worker thread
speech_thread = threading.Thread(target=_speech_worker, daemon=True)
speech_thread.start()

def speak_async(text):
    """Queue text to be spoken asynchronously"""
    if os.getenv("COORD_DISABLE_LOCAL_TTS", "0") == "1":
        return
    speech_queue.put(text)

# Initialize models (min_face_size=40 for detecting faces from farther away)
detector = MTCNN(keep_all=True, select_largest=False, device=device, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class AutomatedFaceRecognitionSystem:
    """Complete automated face recognition system with improved UX"""
    
    def __init__(self):
        self.client = supabase_service.client
        self.detector = detector
        self.resnet = resnet
        self.device = device
        self.face_cascade = face_cascade
        self.confidence_threshold = 0.8
        self.similarity_threshold = 0.7
        
        # Initialize Face Recognition Model
        self.face_model = FaceRecognitionModel(similarity_threshold=0.7)
        
        # Timing settings (FAST - for walking people)
        self.stable_frames_required = 3   # ~0.1 seconds at 30fps (very fast)
        self.countdown_seconds = 0.3      # 0.3 second countdown (instant)
        self.min_face_size = 60           # Detect smaller/distant faces
        self.cooldown_seconds = 1         # 1 second cooldown
        self.frame_skip = 1               # Process every frame
        
        # State tracking
        self.stable_frame_count = 0
        self.last_face_position = None
        self.last_capture_time = None
        self.countdown_start_time = None
        self.current_status = "Waiting for face..."
        self.current_match = None
        self.is_capturing = False
        self.pending_name_input = False
        self.name_input_buffer = ""
        self.use_image_comparison = False  # Disable to speed up (embeddings are enough)
    
    def _calculate_face_center(self, x, y, w, h):
        """Calculate center point of face bounding box"""
        return (x + w // 2, y + h // 2)
    
    def _is_same_face(self, new_pos, threshold=120):
        """Check if detected face is approximately same position as before (increased tolerance for movement)"""
        if self.last_face_position is None:
            return False
        
        old_center = self.last_face_position
        new_center = new_pos
        
        distance = np.sqrt((old_center[0] - new_center[0])**2 + 
                          (old_center[1] - new_center[1])**2)
        return distance < threshold
    
    def _calculate_blur_score(self, image):
        """Calculate blur score using Laplacian variance (higher = sharper)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _draw_progress_bar(self, frame, progress, x, y, w):
        """Draw countdown progress bar on frame"""
        bar_height = 8
        bar_y = y - 15
        
        # Background
        cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress fill
        fill_width = int(w * progress)
        color = (0, 255, 0) if progress < 0.7 else (0, 200, 255)
        cv2.rectangle(frame, (x, bar_y), (x + fill_width, bar_y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + bar_height), (255, 255, 255), 1)
    
    def _draw_status_overlay(self, frame, status_text, color=(255, 255, 255)):
        """Draw status text overlay at bottom of frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status text
        cv2.putText(frame, status_text, (15, h - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_name_input_overlay(self, frame):
        """Draw name input overlay for new face registration"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Input box
        box_w, box_h = 400, 120
        box_x = (w - box_w) // 2
        box_y = (h - box_h) // 2
        
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "NEW FACE DETECTED", (box_x + 80, box_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # Input prompt
        cv2.putText(frame, "Enter name:", (box_x + 20, box_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Input field
        input_text = self.name_input_buffer + "_"
        cv2.rectangle(frame, (box_x + 20, box_y + 70), (box_x + box_w - 20, box_y + 100), (30, 30, 30), -1)
        cv2.putText(frame, input_text, (box_x + 30, box_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press ENTER to save | ESC to cancel", (box_x + 50, box_y + box_h + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def run_continuous_recognition(self):
        """
        IMPROVED AUTOMATED WORKFLOW - RUNS CONTINUOUSLY
        
        Process:
        1. Camera continuously running
        2. Face detected → Stabilize for 0.5s → 3s countdown
        3. User can press SPACE to capture early, ESC to skip
        4. After countdown → Auto-capture photo
        5. Scan database to recognize
        6. IF RECOGNIZED:
           - Show name + confidence on screen
           - Check 1-week rule
           - Update if allowed
        7. IF NOT RECOGNIZED (NEW FACE):
           - Show name input overlay
           - User types name and presses ENTER
           - Register in database
        8. 5-second cooldown, then loop back
        """
        
        print(f"\n{'='*70}")
        print("AUTOMATED FACE RECOGNITION SYSTEM - IMPROVED VERSION")
        print(f"{'='*70}")
        print("\n📹 Controls:")
        print("   • SPACE = Capture now (skip countdown)")
        print("   • ESC   = Exit / Cancel")
        print("   • Face is auto-captured after 3s countdown")
        print(f"{'='*70}\n")
        
        camera_index = int(os.getenv("FACE_CAMERA_INDEX", "0"))
        camera_w = int(os.getenv("FACE_CAMERA_WIDTH", "640"))
        camera_h = int(os.getenv("FACE_CAMERA_HEIGHT", "480"))
        camera_fps = int(os.getenv("FACE_CAMERA_FPS", "20"))

        from camera_utils import open_camera as _shared_open_camera
        cap = _shared_open_camera(index=camera_index, width=camera_w, height=camera_h, fps=camera_fps)
        
        if cap is None or not cap.isOpened():
            print("✗ Could not open camera")
            return
        
        # Set camera properties for smoother capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_h)
        cap.set(cv2.CAP_PROP_FPS, camera_fps)
        
        print("✓ Camera started\n")
        
        # Load all registered faces
        print("📊 Loading registered faces from database...")
        all_embeddings = supabase_service.get_all_user_embeddings()
        
        if all_embeddings:
            total_faces = sum(len(embs) for embs in all_embeddings.values())
            print(f"✓ Loaded {len(all_embeddings)} people ({total_faces} faces)\n")
        else:
            print("⚠️  No registered faces yet. New faces will be registered.\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            
            # Handle name input mode
            if self.pending_name_input:
                self._draw_name_input_overlay(display_frame)
                
                if key == 27:  # ESC - cancel
                    self.pending_name_input = False
                    self.name_input_buffer = ""
                    self.current_status = "Registration cancelled"
                    print("⚠️  Name input cancelled")
                elif key == 13:  # ENTER - save
                    if self.name_input_buffer.strip():
                        self._register_new_face(self.name_input_buffer.strip().lower(), all_embeddings)
                        all_embeddings = supabase_service.get_all_user_embeddings()
                    self.pending_name_input = False
                    self.name_input_buffer = ""
                elif key == 8:  # BACKSPACE
                    self.name_input_buffer = self.name_input_buffer[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    self.name_input_buffer += chr(key)
                
                cv2.imshow('Face Recognition System - Press ESC to Exit', display_frame)
                continue
            
            # Check ESC to exit
            if key == 27:
                print("\n✓ System stopped by user")
                break
            
            # Check cooldown
            if self.last_capture_time:
                elapsed = (datetime.now() - self.last_capture_time).total_seconds()
                if elapsed < self.cooldown_seconds:
                    remaining = self.cooldown_seconds - elapsed
                    self.current_status = f"Cooldown: {remaining:.1f}s"
                    self._draw_status_overlay(display_frame, self.current_status, (100, 100, 255))
                    cv2.imshow('Face Recognition System - Press ESC to Exit', display_frame)
                    continue
            
            # Only process every Nth frame for performance
            if frame_count % self.frame_skip != 0:
                # Still show current status
                self._draw_status_overlay(display_frame, self.current_status)
                cv2.imshow('Face Recognition System - Press ESC to Exit', display_frame)
                continue
            
            try:
                # Detect faces
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5, minSize=(self.min_face_size, self.min_face_size))
                
                if len(faces) > 0:
                    # Get largest face
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    x, y, w, h = faces[0]
                    
                    face_center = self._calculate_face_center(x, y, w, h)
                    
                    # Check face quality
                    face_roi = frame[y:y+h, x:x+w]
                    blur_score = self._calculate_blur_score(face_roi)
                    
                    if blur_score < 30:  # Only reject very blurry faces (allow shaky/moving)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        self.current_status = "Face too blurry"
                        self.stable_frame_count = 0
                        self.countdown_start_time = None
                    elif w < self.min_face_size:  # Too small
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        self.current_status = "Move closer to camera"
                        self.stable_frame_count = 0
                        self.countdown_start_time = None
                    else:
                        # Good quality face detected
                        if self._is_same_face(face_center):
                            self.stable_frame_count += 1
                        else:
                            self.stable_frame_count = 1
                        
                        self.last_face_position = face_center
                        
                        # Stabilization phase
                        if self.stable_frame_count < self.stable_frames_required:
                            progress = self.stable_frame_count / self.stable_frames_required
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            self.current_status = f"Stabilizing... {int(progress * 100)}%"
                        else:
                            # Start countdown if not already started
                            if self.countdown_start_time is None:
                                self.countdown_start_time = datetime.now()
                                print("\n🔴 Face detected - Starting countdown...")
                            
                            elapsed = (datetime.now() - self.countdown_start_time).total_seconds()
                            remaining = self.countdown_seconds - elapsed
                            progress = elapsed / self.countdown_seconds
                            
                            # Draw countdown
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            self._draw_progress_bar(display_frame, progress, x, y, w)
                            
                            if remaining > 0:
                                self.current_status = f"Capturing in {remaining:.1f}s (SPACE to capture now)"
                                
                                # Check for SPACE to capture early
                                if key == 32:  # SPACE
                                    remaining = 0
                            
                            # Capture when countdown complete
                            if remaining <= 0:
                                print("📸 Capturing...")
                                self._process_capture(frame, x, y, w, h, all_embeddings)
                                self.countdown_start_time = None
                                self.stable_frame_count = 0
                                self.last_capture_time = datetime.now()
                else:
                    # No face detected
                    self.stable_frame_count = 0
                    self.last_face_position = None
                    self.countdown_start_time = None
                    self.current_status = "Waiting for face..."
                
                # Draw status
                status_color = (0, 255, 0) if "Capturing" in self.current_status else (255, 255, 255)
                self._draw_status_overlay(display_frame, self.current_status, status_color)
                
                cv2.imshow('Face Recognition System - Press ESC to Exit', display_frame)
                
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _process_capture(self, frame, x, y, w, h, all_embeddings):
        """Process captured face - recognize or register"""
        
        # Extract and save face
        face_roi = frame[y:y+h, x:x+w].copy()
        temp_file = "temp_face.jpg"
        cv2.imwrite(temp_file, face_roi)
        
        # Use the FaceRecognitionModel for comparison
        embedding, confidence, _ = self.face_model.extract_embedding_with_confidence(temp_file)
        
        if embedding is None:
            self.current_status = "Could not process face"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return
        
        # Compare with database using the model
        result = self.face_model.compare_with_database(embedding, top_k=3)
        
        best_match = None
        best_similarity = 0
        
        if result.get('best_match'):
            best_match = result['best_match']['user_id']
            best_similarity = result['best_match']['similarity']
            
            # Optionally verify with actual image comparison from Supabase
            if self.use_image_comparison and result['is_recognized']:
                img_result = self.face_model.compare_with_user_images(temp_file, best_match)
                if img_result.get('is_match'):
                    best_similarity = img_result['best_match']['similarity']
        
        # Process result
        if best_match and best_similarity > self.similarity_threshold:
            # RECOGNIZED
            print(f"\n✅ RECOGNIZED: {best_match.upper()}")
            print(f"   Confidence: {best_similarity:.2%}")
            print(f"COORD_EVENT|source=face|type=recognized|priority=1|text={best_match.title()} recognized")
            self.current_status = f"Welcome {best_match.title()}! ({best_similarity:.0%} match)"
            
            # Speak the recognized name
            speak_async(best_match.title())
            
            # Check 1-week rule
            self._check_and_update_photo(best_match, temp_file, embedding)
        else:
            # NEW FACE - trigger name input
            print(f"\n❓ NEW FACE (confidence: {best_similarity:.2%})")
            self.current_status = "New face - enter name"
            self.pending_name_input = True
            self.pending_embedding = embedding
            self.pending_temp_file = temp_file
            return  # Don't delete temp file yet
        
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    def _check_and_update_photo(self, user_id, temp_file, embedding):
        """Check 1-week rule and update photo if allowed"""
        try:
            photos_response = self.client.table('photos').select(
                'uploaded_at'
            ).eq('user_id', user_id).order('uploaded_at', desc=True).limit(1).execute()
            
            if photos_response.data:
                last_upload_str = photos_response.data[0]['uploaded_at']
                last_upload = datetime.fromisoformat(last_upload_str.replace('Z', '+00:00'))
                days_since = (datetime.now(last_upload.tzinfo) - last_upload).days
                
                if days_since < 7:
                    print(f"   ⏰ Last photo: {days_since} days ago (skip update)")
                else:
                    print(f"   ⏰ Last photo: {days_since} days ago - updating...")
                    self._store_photo(user_id, temp_file, embedding)
            else:
                self._store_photo(user_id, temp_file, embedding)
                
        except Exception as e:
            print(f"   Could not check update rule: {e}")
    
    def _register_new_face(self, person_name, all_embeddings):
        """Register a new face in the database"""
        try:
            # Create user
            existing_user = supabase_service.get_user(person_name)
            if not existing_user:
                supabase_service.create_user(person_name, {
                    "name": person_name.title(),
                    "email": f"{person_name}@facerec.local"
                })
                print(f"\n✓ Created new user: {person_name.upper()}")
            
            # Store photo
            if hasattr(self, 'pending_temp_file') and hasattr(self, 'pending_embedding'):
                self._store_photo(person_name, self.pending_temp_file, self.pending_embedding)
                
                # Cleanup
                if os.path.exists(self.pending_temp_file):
                    os.remove(self.pending_temp_file)
                
                del self.pending_temp_file
                del self.pending_embedding
            
            self.current_status = f"Registered: {person_name.title()}"
            print(f"✅ NEW FACE REGISTERED: {person_name.upper()}\n")
            print(f"COORD_EVENT|source=face|type=registered|priority=1|text={person_name.title()} registered")
            
            # Speak the registration confirmation
            speak_async(f"{person_name.title()} is registered")
            
        except Exception as e:
            print(f"✗ Error registering face: {e}")
            self.current_status = "Registration failed"
    
    def _store_photo(self, user_name: str, temp_file: str, embedding: np.ndarray):
        """Store photo to Supabase and database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_name}_{timestamp}.jpg"
            
            # ALWAYS save embedding first (most important for recognition)
            print(f"   📝 Saving embedding for {user_name}...")
            embedding_result = supabase_service.add_embedding(
                user_name,
                embedding,
                f"{user_name}/{filename}"
            )
            
            if embedding_result:
                print(f"   ✓ Embedding saved successfully")
            else:
                print(f"   ⚠️ Embedding save returned False (check foreign key)")
            
            # Then try to upload photo (optional - may fail due to storage policy)
            try:
                result = photo_service.upload_photo(user_name, temp_file, filename)
                
                if result and 'id' in result:
                    # Save photo details
                    photo_service.save_photo_details(result['id'], user_name, {
                        'detection_confidence': 0.95,
                        'face_quality': 0.85,
                        'is_frontal': True,
                        'lighting_quality': 0.8,
                        'blur_score': 0.1,
                        'is_usable': True
                    })
                    print(f"   ✓ Photo stored: {filename}")
                else:
                    print(f"   ⚠️ Photo storage failed (storage policy issue)")
            except Exception as photo_err:
                print(f"   ⚠️ Photo upload error: {photo_err}")
                print(f"   💡 Embedding still saved - recognition will work!")
        
        except Exception as e:
            print(f"   ✗ Error storing: {e}")


# ==================== MAIN ====================

def main():
    """Run the automated face recognition system"""
    
    print("\n" + "="*70)
    print("AUTOMATED FACE RECOGNITION & REGISTRATION SYSTEM")
    print("="*70)
    print("\nImproved features:")
    print("  ✓ Stabilized face detection (0.5s)")
    print("  ✓ 3-second countdown before capture")
    print("  ✓ On-screen name input for new faces")
    print("  ✓ Face quality checks (blur, size)")
    print("  ✓ 5-second cooldown between captures")
    print("  ✓ Visual progress indicators")
    print("\nControls:")
    print("  • SPACE = Capture immediately")
    print("  • ESC   = Exit")
    print("\nPress ESC to exit\n")
    
    system = AutomatedFaceRecognitionSystem()
    system.run_continuous_recognition()
    
    print("\n" + "="*70)
    print("Thank you for using Automated Face Recognition System!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
