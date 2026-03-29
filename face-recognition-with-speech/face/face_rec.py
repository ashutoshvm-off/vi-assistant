
import torch
import numpy as np
import os
import json
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from supabase_service import SupabaseService
from photo_service import PhotoService

# Initialize Services
supabase_service = SupabaseService()
photo_service = PhotoService()

# 1. Initialize Face Detector (MTCNN)
# keep_all=False: only return the face with the highest probability (good for 1:1 matching)
# select_largest=True: if multiple faces, pick the largest one
# device: use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

detector = MTCNN(keep_all=False, select_largest=True, device=device)

# 2. Initialize Recognition Model (InceptionResnetV1)
# pretrained='vggface2': use model trained on VGGFace2 dataset
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def get_embedding(img_path):
    """
    Detects face, crops it, and returns the embedding.
    """
    if not os.path.exists(img_path):
        print(f"Error: File not found - {img_path}")
        return None

    try:
        img = Image.open(img_path).convert('RGB')
        
        # MTCNN detects and optionally saves the cropped face. 
        # Here we want the tensor directly for the embedding model.
        # return_prob=False gives just the image tensor.
        img_cropped = detector(img) 

        if img_cropped is None:
            print(f"No face detected in {img_path}")
            return None

        # Calculate embedding
        with torch.no_grad():
            # Add batch dimension: (1, 3, 160, 160)
            img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
        
        # Detach directly returns tensor to CPU if needed
        return img_embedding.cpu().detach().numpy().squeeze()
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def store_face_embedding_with_photo(user_id, img_path, metadata=None):
    """
    Upload photo to storage and extract + store face embedding
    
    Args:
        user_id: Unique user identifier
        img_path: Path to image file
        metadata: Optional metadata dict
    
    Returns:
        Dictionary with photo and embedding info or None if failed
    """
    # Step 1: Upload photo
    photo_record = photo_service.upload_photo(user_id, img_path)
    if photo_record is None:
        return None
    
    photo_id = photo_record['id']
    
    # Step 2: Extract embedding
    embedding = get_embedding(img_path)
    if embedding is None:
        print(f"✗ Could not extract embedding for photo {photo_id}")
        return None
    
    # Step 3: Store embedding in embeddings table
    try:
        embedding_list = embedding.tolist()
        embedding_data = {
            'user_id': user_id,
            'embedding': embedding_list,
            'embedding_dimension': len(embedding_list),
            'image_path': img_path
        }
        response = supabase_service.client.table('face_embeddings').insert(embedding_data).execute()
        embedding_id = response.data[0]['id'] if response.data else None
        
        # Step 4: Update photo record with embedding_id
        if embedding_id:
            supabase_service.client.table('photos').update({
                'embedding_id': embedding_id
            }).eq('id', photo_id).execute()
        
        # Step 5: Save photo details
        face_quality = calculate_face_quality(embedding)
        
        analysis_data = {
            'detection_confidence': 0.95,  # From MTCNN
            'face_quality': face_quality,
            'is_frontal': True,
            'lighting_quality': 0.8,
            'blur_score': 0.1,
            'is_usable': True
        }
        
        photo_service.save_photo_details(photo_id, user_id, analysis_data)
        
        print(f"✓ Photo uploaded and face embedding stored successfully")
        return {
            'photo_id': photo_id,
            'embedding_id': embedding_id,
            'photo_info': photo_record,
            'embedding_dimension': len(embedding_list)
        }
    except Exception as e:
        print(f"✗ Error storing embedding: {e}")
        return None

def calculate_face_quality(embedding: np.ndarray) -> float:
    """
    Calculate face quality score based on embedding
    
    Args:
        embedding: Face embedding vector
    
    Returns:
        Quality score (0-1)
    """
    # Normalize embedding
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return 0.0
    
    # Quality based on embedding magnitude and variance
    normalized = embedding / norm
    variance = np.var(normalized)
    
    # Higher variance = more discriminative features = higher quality
    quality = min(1.0, variance * 10)
    return quality

def compare_faces_with_photos(photo_id_1: int, photo_id_2: int, 
                              user_id_1: str = None, user_id_2: str = None, 
                              threshold: float = 0.7) -> dict:
    """
    Compare two photos and store comparison result
    
    Args:
        photo_id_1: ID of first photo
        photo_id_2: ID of second photo
        user_id_1: User ID for first photo
        user_id_2: User ID for second photo
        threshold: Similarity threshold
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\nComparing photos {photo_id_1} and {photo_id_2}...")
    
    try:
        # Get embeddings from database
        emb1_response = (supabase_service.client.table('face_embeddings')
                        .select("embedding")
                        .eq("id", photo_id_1)  # Using embedding_id from photos
                        .limit(1)
                        .execute())
        
        emb2_response = (supabase_service.client.table('face_embeddings')
                        .select("embedding")
                        .eq("id", photo_id_2)
                        .limit(1)
                        .execute())
        
        if not emb1_response.data or not emb2_response.data:
            print("✗ One or both photos have no embeddings")
            return None
        
        emb1 = np.array(emb1_response.data[0]['embedding'])
        emb2 = np.array(emb2_response.data[0]['embedding'])
        
        # Calculate similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        
        is_match = similarity > threshold
        
        print(f"Cosine Similarity: {similarity:.4f}")
        print(f"Result: {'✓ MATCH' if is_match else '✗ NOT A MATCH'}")
        
        # Store comparison in database
        if user_id_1 and user_id_2:
            comparison_data = {
                'photo_id_1': photo_id_1,
                'photo_id_2': photo_id_2,
                'user_id_1': user_id_1,
                'user_id_2': user_id_2,
                'similarity_score': float(similarity),
                'match': is_match
            }
            supabase_service.client.table('photo_comparisons').insert(comparison_data).execute()
        
        return {
            'similarity_score': similarity,
            'is_match': is_match,
            'threshold': threshold,
            'photo_id_1': photo_id_1,
            'photo_id_2': photo_id_2
        }
        
    except Exception as e:
        print(f"✗ Error comparing photos: {e}")
        return None

def compare_faces(path1, path2, threshold=0.7):
    """Legacy function - compares two image files directly"""
    print(f"\nComparing {path1} and {path2}...")
    
    emb1 = get_embedding(path1)
    if emb1 is None: return None

    emb2 = get_embedding(path2)
    if emb2 is None: return None

    # Normalized Cosine Similarity
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    sim = np.dot(emb1, emb2) / (norm1 * norm2)
    
    print(f"Cosine Similarity: {sim:.4f}")

    is_match = sim > threshold
    if is_match:
        print("✓ Match! They are the same person.")
    else:
        print("✗ Not a match.")
    
    return sim

if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("FACE RECOGNITION WITH SUPABASE INTEGRATION")
    print("=" * 60)
    
    # Ensure these files exist or replace with your own paths
    db_image = "db_person.jpg"
    query_image = "query.jpg"

    if os.path.exists(db_image) and os.path.exists(query_image):
        # Example 1: Upload photos with face embeddings
        print("\n--- Example 1: Upload Photos ---")
        result1 = store_face_embedding_with_photo("user_001", db_image)
        result2 = store_face_embedding_with_photo("user_002", query_image)
        
        if result1 and result2:
            # Example 2: Compare stored photos
            print("\n--- Example 2: Compare Stored Photos ---")
            compare_faces_with_photos(
                photo_id_1=result1['photo_id'],
                photo_id_2=result2['photo_id'],
                user_id_1="user_001",
                user_id_2="user_002"
            )
            
            # Example 3: Get user photos
            print("\n--- Example 3: Get User Photos ---")
            user_photos = photo_service.get_user_photos("user_001")
            print(f"User 001 has {len(user_photos)} photos")
            
            # Example 4: Get photo quality stats
            print("\n--- Example 4: Photo Quality Stats ---")
            stats = photo_service.get_photo_stats("user_001")
            print(f"Quality stats: {json.dumps(stats, indent=2)}")
            
            # Example 5: Get best photo
            print("\n--- Example 5: Best Quality Photo ---")
            best_photo = photo_service.get_best_photo("user_001")
            if best_photo:
                print(f"Best photo: {best_photo['photo_name']}")
        
        # Example 6: Direct face comparison (without storage)
        print("\n--- Example 6: Direct Face Comparison ---")
        compare_faces(db_image, query_image)
    
    else:
        print("✗ Please provide 'db_person.jpg' and 'query.jpg' to test.")
        print("\nAlternatively, use the functions directly:")
        print("  store_face_embedding_with_photo('user_id', 'path/to/photo.jpg')")
        print("  compare_faces_with_photos(photo_id_1, photo_id_2, 'user_1', 'user_2')")
        print("  photo_service.get_user_photos('user_id')")
