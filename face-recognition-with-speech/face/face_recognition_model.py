"""
Face Recognition Model
Dedicated module for face recognition with Supabase image comparison
Uses FaceNet-PyTorch (MTCNN + InceptionResnetV1) for recognition
"""

import torch
import numpy as np
import os
import io
import requests
from PIL import Image
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from supabase_service import SupabaseService
from photo_service import PhotoService


class FaceRecognitionModel:
    """
    Face Recognition Model with Supabase Integration
    
    Features:
    - Face detection using MTCNN
    - Face embedding extraction using InceptionResnetV1 (FaceNet)
    - Image fetching from Supabase Storage
    - Direct image-to-image comparison
    - Embedding-based comparison from database
    """
    
    def __init__(self, similarity_threshold: float = 0.7, min_face_size: int = 80):
        """
        Initialize the face recognition model
        
        Args:
            similarity_threshold: Minimum similarity score to consider a match (0-1)
            min_face_size: Minimum face size in pixels for detection
        """
        self.supabase_service = SupabaseService()
        self.photo_service = PhotoService()
        
        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN for face detection
        self.detector = MTCNN(
            keep_all=False,
            select_largest=True,
            min_face_size=min_face_size,
            device=self.device
        )
        
        # Initialize InceptionResnetV1 for face embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        
        # Cache for embeddings
        self._embedding_cache = {}
        self._cache_loaded = False
    
    # ==================== EMBEDDING EXTRACTION ====================
    
    def extract_embedding(self, image_input) -> np.ndarray:
        """
        Extract face embedding from image
        
        Args:
            image_input: Can be:
                - str: File path to image
                - PIL.Image: PIL Image object
                - np.ndarray: OpenCV BGR image
                
        Returns:
            Numpy array of face embedding (512-dim) or None if no face detected
        """
        try:
            # Convert to PIL Image
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    print(f"✗ File not found: {image_input}")
                    return None
                img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                # OpenCV BGR to RGB
                img = Image.fromarray(image_input[:, :, ::-1])
            elif isinstance(image_input, Image.Image):
                img = image_input.convert('RGB')
            else:
                print(f"✗ Unsupported image type: {type(image_input)}")
                return None
            
            # Detect face and get cropped tensor
            img_cropped = self.detector(img)
            
            if img_cropped is None:
                return None
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device))
            
            return embedding.cpu().detach().numpy().squeeze()
            
        except Exception as e:
            print(f"✗ Error extracting embedding: {e}")
            return None
    
    def extract_embedding_with_confidence(self, image_input) -> tuple:
        """
        Extract face embedding with detection confidence
        
        Returns:
            Tuple of (embedding, confidence, face_box) or (None, 0, None)
        """
        try:
            # Convert to PIL Image
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input[:, :, ::-1])
            elif isinstance(image_input, Image.Image):
                img = image_input.convert('RGB')
            else:
                return None, 0, None
            
            # Detect face with probability
            boxes, probs = self.detector.detect(img)
            
            if boxes is None or len(boxes) == 0:
                return None, 0, None
            
            # Get highest confidence face
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            face_box = boxes[best_idx]
            
            # Get cropped face for embedding
            img_cropped = self.detector(img)
            
            if img_cropped is None:
                return None, 0, None
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device))
            
            return embedding.cpu().detach().numpy().squeeze(), confidence, face_box
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return None, 0, None
    
    # ==================== SUPABASE IMAGE OPERATIONS ====================
    
    def download_image_from_supabase(self, user_id: str, photo_name: str) -> Image.Image:
        """
        Download image from Supabase Storage
        
        Args:
            user_id: User identifier
            photo_name: Name of the photo file
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            # Get public URL
            storage_path = f"{user_id}/{photo_name}"
            url = self.supabase_service.client.storage.from_(
                self.photo_service.storage_bucket
            ).get_public_url(storage_path)
            
            # Download image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            return img
            
        except Exception as e:
            print(f"✗ Error downloading image: {e}")
            return None
    
    def get_user_images_from_supabase(self, user_id: str, limit: int = 5) -> list:
        """
        Get all images for a user from Supabase Storage
        
        Args:
            user_id: User identifier
            limit: Maximum number of images to retrieve
            
        Returns:
            List of PIL Images
        """
        try:
            # Get photo records from database
            photos = self.photo_service.get_user_photos(user_id, limit)
            
            images = []
            for photo in photos:
                img = self.download_image_from_supabase(user_id, photo['photo_name'])
                if img:
                    images.append({
                        'image': img,
                        'photo_id': photo['id'],
                        'photo_name': photo['photo_name'],
                        'uploaded_at': photo.get('uploaded_at')
                    })
            
            return images
            
        except Exception as e:
            print(f"✗ Error getting user images: {e}")
            return []
    
    # ==================== COMPARISON METHODS ====================
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def compare_images(self, image1, image2) -> dict:
        """
        Compare two images directly
        
        Args:
            image1: First image (path, PIL Image, or numpy array)
            image2: Second image (path, PIL Image, or numpy array)
            
        Returns:
            Comparison result dictionary
        """
        print("\n🔍 Comparing two images...")
        
        # Extract embeddings
        emb1 = self.extract_embedding(image1)
        if emb1 is None:
            return {'error': 'No face detected in first image', 'is_match': False}
        
        emb2 = self.extract_embedding(image2)
        if emb2 is None:
            return {'error': 'No face detected in second image', 'is_match': False}
        
        # Calculate similarity
        similarity = self.calculate_similarity(emb1, emb2)
        is_match = similarity > self.similarity_threshold
        
        result = {
            'similarity': similarity,
            'is_match': is_match,
            'threshold': self.similarity_threshold,
            'confidence': f"{similarity:.2%}"
        }
        
        return result
    
    def compare_with_database(self, image_input, top_k: int = 5) -> dict:
        """
        Compare input image with all faces in database
        
        Args:
            image_input: Image to compare (path, PIL Image, numpy array, or embedding)
            top_k: Number of top matches to return
            
        Returns:
            Dictionary with match results
        """
        # Extract embedding if not already
        if isinstance(image_input, np.ndarray) and image_input.shape == (512,):
            query_embedding = image_input
        else:
            query_embedding = self.extract_embedding(image_input)
            if query_embedding is None:
                return {'error': 'No face detected in input image', 'matches': []}
        
        # Load all embeddings from database
        all_embeddings = self.supabase_service.get_all_user_embeddings()
        
        if not all_embeddings:
            return {'error': 'No faces registered in database', 'matches': []}
        
        # Compare with all embeddings
        all_matches = []
        
        for user_id, embeddings in all_embeddings.items():
            for emb_data in embeddings:
                db_embedding = emb_data['embedding']
                similarity = self.calculate_similarity(query_embedding, db_embedding)
                
                all_matches.append({
                    'user_id': user_id,
                    'similarity': similarity,
                    'is_match': similarity > self.similarity_threshold,
                    'embedding_id': emb_data.get('id'),
                    'created_at': emb_data.get('created_at')
                })
        
        # Sort by similarity (descending)
        all_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top matches
        top_matches = all_matches[:top_k]
        
        # Get best match
        best_match = top_matches[0] if top_matches else None
        
        result = {
            'best_match': best_match,
            'is_recognized': best_match and best_match['is_match'],
            'top_matches': top_matches,
            'total_compared': len(all_matches)
        }
        
        return result
    
    def compare_with_user_images(self, image_input, user_id: str) -> dict:
        """
        Compare input image with stored images of a specific user
        Fetches actual images from Supabase Storage for comparison
        
        Args:
            image_input: Image to compare
            user_id: User to compare against
            
        Returns:
            Comparison result dictionary
        """
        print(f"\n🔍 Comparing with images of user: {user_id}")
        
        # Extract query embedding
        query_embedding = self.extract_embedding(image_input)
        if query_embedding is None:
            return {'error': 'No face detected in input image', 'is_match': False}
        
        # Get user's images from Supabase Storage
        user_images = self.get_user_images_from_supabase(user_id, limit=5)
        
        if not user_images:
            print(f"   No images found for user: {user_id}")
            return {'error': f'No images found for user {user_id}', 'is_match': False}
        
        # Compare with each image
        comparisons = []
        for img_data in user_images:
            db_embedding = self.extract_embedding(img_data['image'])
            if db_embedding is not None:
                similarity = self.calculate_similarity(query_embedding, db_embedding)
                comparisons.append({
                    'photo_name': img_data['photo_name'],
                    'photo_id': img_data['photo_id'],
                    'similarity': similarity,
                    'is_match': similarity > self.similarity_threshold
                })
        
        if not comparisons:
            return {'error': 'Could not extract embeddings from user images', 'is_match': False}
        
        # Get best comparison
        best = max(comparisons, key=lambda x: x['similarity'])
        avg_similarity = sum(c['similarity'] for c in comparisons) / len(comparisons)
        
        result = {
            'user_id': user_id,
            'best_match': best,
            'average_similarity': avg_similarity,
            'is_match': best['is_match'],
            'comparisons': comparisons,
            'images_compared': len(comparisons)
        }
        
        return result
    
    # ==================== REGISTRATION ====================
    
    def register_face(self, user_id: str, image_input, user_name: str = None) -> dict:
        """
        Register a new face in the database
        
        Args:
            user_id: Unique user identifier
            image_input: Image containing the face
            user_name: Optional display name
            
        Returns:
            Registration result dictionary
        """
        print(f"\n📝 Registering face for user: {user_id}")
        
        # Extract embedding with confidence
        embedding, confidence, face_box = self.extract_embedding_with_confidence(image_input)
        
        if embedding is None:
            return {'error': 'No face detected in image', 'success': False}
        
        if confidence < 0.9:
            return {'error': f'Low detection confidence: {confidence:.2%}', 'success': False}
        
        # Create user if not exists
        existing_user = self.supabase_service.get_user(user_id)
        if not existing_user:
            self.supabase_service.create_user(user_id, {
                'name': user_name or user_id.title(),
                'email': f'{user_id}@facerec.local'
            })
        
        # Save image temporarily
        temp_file = f"temp_register_{user_id}.jpg"
        if isinstance(image_input, str):
            import shutil
            shutil.copy(image_input, temp_file)
        elif isinstance(image_input, np.ndarray):
            import cv2
            cv2.imwrite(temp_file, image_input)
        elif isinstance(image_input, Image.Image):
            image_input.save(temp_file)
        
        # Upload to Supabase Storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_{timestamp}.jpg"
        photo_result = self.photo_service.upload_photo(user_id, temp_file, filename)
        
        # Store embedding
        if photo_result:
            self.supabase_service.add_embedding(
                user_id,
                embedding,
                f"{user_id}/{filename}"
            )
        
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Invalidate cache
        self._cache_loaded = False
        
        result = {
            'success': True,
            'user_id': user_id,
            'photo_id': photo_result.get('id') if photo_result else None,
            'detection_confidence': confidence,
            'embedding_dimension': len(embedding)
        }
        
        return result
    
    # ==================== CACHE MANAGEMENT ====================
    
    def preload_embeddings(self) -> int:
        """
        Preload all embeddings into memory for faster comparison
        
        Returns:
            Number of embeddings loaded
        """
        print("\n📦 Preloading embeddings...")
        
        all_embeddings = self.supabase_service.get_all_user_embeddings()
        self._embedding_cache = all_embeddings
        self._cache_loaded = True
        
        total = sum(len(embs) for embs in all_embeddings.values())
        return total
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache = {}
        self._cache_loaded = False
        print("   ✓ Cache cleared")
    
    # ==================== UTILITY ====================
    
    def get_model_info(self) -> dict:
        """Get information about the model configuration"""
        return {
            'device': str(self.device),
            'detector': 'MTCNN',
            'encoder': 'InceptionResnetV1 (VGGFace2)',
            'embedding_dimension': 512,
            'similarity_threshold': self.similarity_threshold,
            'min_face_size': self.min_face_size,
            'cache_loaded': self._cache_loaded,
            'cached_users': len(self._embedding_cache) if self._cache_loaded else 0
        }


# ==================== DEMO ====================

def demo():
    """Demonstration of the FaceRecognitionModel"""
    
    print("\n" + "="*70)
    print("FACE RECOGNITION MODEL - DEMO")
    print("="*70)
    
    # Initialize model
    model = FaceRecognitionModel(similarity_threshold=0.7)
    
    # Print model info
    print("\n📋 Model Configuration:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Preload embeddings
    model.preload_embeddings()
    
    # Example usage
    print("\n📖 Example Usage:")
    print("""
    # Compare two images
    result = model.compare_images('image1.jpg', 'image2.jpg')
    
    # Compare with database
    result = model.compare_with_database('query_face.jpg')
    
    # Compare with specific user's stored images
    result = model.compare_with_user_images('query.jpg', 'john_doe')
    
    # Register new face
    result = model.register_face('new_user', 'new_face.jpg', 'New User')
    """)
    
    print("="*70 + "\n")


if __name__ == "__main__":
    demo()
