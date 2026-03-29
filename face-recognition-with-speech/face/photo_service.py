"""
Photo Service Module
Handles photo uploads, storage, and face analysis with Supabase
"""

import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
from pathlib import Path
from supabase_config import get_supabase_admin_client


class PhotoService:
    """Service class for photo management and storage"""
    
    def __init__(self):
        # Use admin client to bypass RLS policies on storage and tables
        self.client = get_supabase_admin_client()
        self.storage_client = self.client  # Same admin client for storage
        self.photos_table = 'photos'
        self.photo_details_table = 'photo_details'
        self.storage_bucket = 'face-photos'
    
    # ==================== PHOTO STORAGE ====================
    
    def upload_photo(self, user_id: str, file_path: str, photo_name: str = None) -> dict:
        """
        Upload photo to Supabase Storage and create metadata record
        
        Args:
            user_id: User identifier
            file_path: Local path to photo file
            photo_name: Optional custom name for photo (defaults to filename)
        
        Returns:
            Dictionary with photo metadata or None if failed
        """
        try:
            if not os.path.exists(file_path):
                print(f"✗ Photo file not found: {file_path}")
                return None
            
            # Get image info
            img = Image.open(file_path)
            width, height = img.size
            file_size = os.path.getsize(file_path)
            file_type = img.format.lower() if img.format else 'jpeg'
            
            # Determine photo name
            if photo_name is None:
                photo_name = os.path.basename(file_path)
            
            # Create storage path: face-photos/user_id/filename
            storage_path = f"{user_id}/{photo_name}"
            
            # Read and upload file using admin client to bypass RLS
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            response = self.storage_client.storage.from_(self.storage_bucket).upload(
                storage_path, 
                file_data,
                file_options={"content-type": f"image/{file_type}"}
            )
            
            # Get public URL
            file_url = self.storage_client.storage.from_(self.storage_bucket).get_public_url(storage_path)
            
            # Store metadata in database
            photo_data = {
                'user_id': user_id,
                'photo_name': photo_name,
                'file_path': storage_path,
                'file_size': file_size,
                'file_type': file_type,
                'width': width,
                'height': height,
                'storage_bucket': self.storage_bucket
            }
            
            db_response = self.client.table(self.photos_table).insert(photo_data).execute()
            
            if db_response.data and len(db_response.data) > 0:
                photo_record = db_response.data[0]
                print(f"✓ Photo '{photo_name}' uploaded successfully for user '{user_id}'")
                return {
                    'id': photo_record['id'],
                    'user_id': user_id,
                    'photo_name': photo_name,
                    'file_path': storage_path,
                    'file_url': file_url,
                    'width': width,
                    'height': height
                }
            else:
                print(f"✗ Failed to store photo metadata")
                return None
                
        except Exception as e:
            print(f"✗ Error uploading photo: {e}")
            return None
    
    def get_user_photos(self, user_id: str, limit: int = 100) -> list:
        """Get all photos for a user"""
        try:
            response = (self.client.table(self.photos_table)
                       .select("*")
                       .eq("user_id", user_id)
                       .order("uploaded_at", desc=True)
                       .limit(limit)
                       .execute())
            return response.data if response.data else []
        except Exception as e:
            print(f"✗ Error retrieving photos: {e}")
            return []
    
    def delete_photo(self, photo_id: int, user_id: str = None) -> bool:
        """Delete photo and storage file"""
        try:
            # Get photo record
            response = self.client.table(self.photos_table).select("*").eq("id", photo_id).execute()
            if not response.data:
                print(f"✗ Photo not found")
                return False
            
            photo = response.data[0]
            
            # Verify ownership if user_id provided
            if user_id and photo['user_id'] != user_id:
                print(f"✗ Unauthorized: Photo belongs to different user")
                return False
            
            # Delete from storage
            self.client.storage.from_(self.storage_bucket).remove([photo['file_path']])
            
            # Delete from database
            self.client.table(self.photos_table).delete().eq("id", photo_id).execute()
            print(f"✓ Photo deleted successfully")
            return True
        except Exception as e:
            print(f"✗ Error deleting photo: {e}")
            return False
    
    # ==================== PHOTO DETAILS (FACE ANALYSIS) ====================
    
    def save_photo_details(self, photo_id: int, user_id: str, analysis_data: dict) -> bool:
        """
        Save face analysis details for a photo
        
        Args:
            photo_id: ID of the photo
            user_id: User identifier
            analysis_data: Dictionary containing face analysis results
                Expected keys:
                - detection_confidence: float (0-1)
                - face_quality: float (0-1)
                - is_frontal: bool
                - estimated_age: int (optional)
                - gender: str (optional)
                - emotion: str (optional)
                - face_coordinates: dict {"x": int, "y": int, "width": int, "height": int}
                - face_landmarks: dict with landmark points
                - lighting_quality: float (0-1)
                - blur_score: float (0-1, lower is better)
                - is_usable: bool
                - notes: str (optional)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            details_data = {
                'photo_id': photo_id,
                'user_id': user_id,
                'detection_confidence': analysis_data.get('detection_confidence'),
                'face_quality': analysis_data.get('face_quality'),
                'is_frontal': analysis_data.get('is_frontal', False),
                'estimated_age': analysis_data.get('estimated_age'),
                'gender': analysis_data.get('gender'),
                'emotion': analysis_data.get('emotion'),
                'face_coordinates': analysis_data.get('face_coordinates'),
                'face_landmarks': analysis_data.get('face_landmarks'),
                'lighting_quality': analysis_data.get('lighting_quality'),
                'blur_score': analysis_data.get('blur_score'),
                'is_usable': analysis_data.get('is_usable', True),
                'notes': analysis_data.get('notes')
            }
            
            # Remove None values
            details_data = {k: v for k, v in details_data.items() if v is not None}
            
            response = self.client.table(self.photo_details_table).insert(details_data).execute()
            print(f"✓ Photo details saved for photo ID {photo_id}")
            return True
        except Exception as e:
            print(f"✗ Error saving photo details: {e}")
            return False
    
    def get_photo_details(self, photo_id: int) -> dict:
        """Get analysis details for a photo"""
        try:
            response = self.client.table(self.photo_details_table).select("*").eq("photo_id", photo_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"✗ Error retrieving photo details: {e}")
            return None
    
    def update_photo_details(self, photo_id: int, analysis_data: dict) -> bool:
        """Update face analysis details for a photo"""
        try:
            response = self.client.table(self.photo_details_table).update(analysis_data).eq("photo_id", photo_id).execute()
            print(f"✓ Photo details updated for photo ID {photo_id}")
            return True
        except Exception as e:
            print(f"✗ Error updating photo details: {e}")
            return False
    
    # ==================== PHOTO QUALITY QUERIES ====================
    
    def get_quality_photos(self, user_id: str) -> list:
        """Get high-quality usable photos for a user"""
        try:
            # Using the quality_photos view
            response = (self.client.table('quality_photos')
                       .select("*")
                       .eq("user_id", user_id)
                       .execute())
            return response.data if response.data else []
        except Exception as e:
            print(f"✗ Error retrieving quality photos: {e}")
            return []
    
    def get_best_photo(self, user_id: str) -> dict:
        """Get the best quality photo for a user"""
        try:
            response = (self.client.rpc('get_best_photo', {'user_id_param': user_id})
                       .execute())
            return response.data[0] if response.data and len(response.data) > 0 else None
        except Exception as e:
            print(f"✗ Error retrieving best photo: {e}")
            return None
    
    def get_photo_stats(self, user_id: str) -> dict:
        """Get photo statistics for a user"""
        try:
            response = (self.client.rpc('count_photos_by_quality', {'user_id_param': user_id})
                       .execute())
            
            stats = {
                'high_quality': 0,
                'medium_quality': 0,
                'low_quality': 0
            }
            
            if response.data:
                for item in response.data:
                    quality = item['quality_level']
                    count = item['count']
                    if quality == 'High':
                        stats['high_quality'] = count
                    elif quality == 'Medium':
                        stats['medium_quality'] = count
                    else:
                        stats['low_quality'] = count
            
            return stats
        except Exception as e:
            print(f"✗ Error retrieving photo stats: {e}")
            return {}
    
    # ==================== PHOTO VIEW QUERIES ====================
    
    def get_photos_with_details(self, user_id: str) -> list:
        """Get all photos with their face analysis details"""
        try:
            response = (self.client.table('photo_with_details')
                       .select("*")
                       .eq("user_id", user_id)
                       .execute())
            return response.data if response.data else []
        except Exception as e:
            print(f"✗ Error retrieving photos with details: {e}")
            return []
    
    # ==================== UTILITY METHODS ====================
    
    def get_storage_url(self, user_id: str, photo_name: str) -> str:
        """Get public URL for a stored photo"""
        try:
            storage_path = f"{user_id}/{photo_name}"
            url = self.client.storage.from_(self.storage_bucket).get_public_url(storage_path)
            return url
        except Exception as e:
            print(f"✗ Error getting storage URL: {e}")
            return None
    
    def calculate_quality_score(self, detection_confidence: float, blur_score: float, 
                                lighting_quality: float) -> float:
        """
        Calculate overall face quality score
        
        Args:
            detection_confidence: Detection confidence (0-1)
            blur_score: Blur score (0-1, lower is better)
            lighting_quality: Lighting quality (0-1)
        
        Returns:
            Overall quality score (0-1)
        """
        # Weighted average: detection (40%), sharpness (40%), lighting (20%)
        quality = (
            (detection_confidence * 0.4) +
            ((1 - blur_score) * 0.4) +
            (lighting_quality * 0.2)
        )
        return min(1.0, max(0.0, quality))
