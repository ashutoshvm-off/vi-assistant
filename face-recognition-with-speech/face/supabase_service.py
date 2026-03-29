"""
Supabase Service Module
Handles all Supabase operations for face recognition data
"""

import json
import numpy as np
from datetime import datetime
from supabase_config import get_supabase_admin_client


def parse_embedding(embedding_data) -> np.ndarray:
    """
    Parse embedding data from database (handles both string and list formats)
    """
    if embedding_data is None:
        return None
    
    # If it's already a list, convert directly
    if isinstance(embedding_data, list):
        return np.array(embedding_data, dtype=np.float32)
    
    # If it's a string, parse it
    if isinstance(embedding_data, str):
        try:
            # Remove any numpy string wrapper
            if embedding_data.startswith("np.str_"):
                embedding_data = embedding_data[8:-2]  # Remove "np.str_('" and "')"
            parsed = json.loads(embedding_data)
            return np.array(parsed, dtype=np.float32)
        except json.JSONDecodeError:
            print(f"✗ Could not parse embedding string")
            return None
    
    # Try direct conversion as fallback
    try:
        return np.array(embedding_data, dtype=np.float32)
    except:
        return None


class SupabaseService:
    """Service class for Supabase operations"""
    
    def __init__(self):
        # Use admin client to bypass RLS policies
        self.client = get_supabase_admin_client()
        self.embeddings_table = 'face_embeddings'
        self.comparisons_table = 'comparisons'
        self.users_table = 'users'
    
    # ==================== USER MANAGEMENT ====================
    
    def create_user(self, user_id: str, user_data: dict) -> bool:
        """
        Create a new user profile
        
        Args:
            user_id: Unique identifier for the user
            user_data: Dictionary with user information
                      (e.g., {'name': 'John', 'email': 'john@example.com'})
        
        Returns:
            True if successful, False otherwise
        """
        try:
            user_data['user_id'] = user_id
            user_data['created_at'] = datetime.now().isoformat()
            
            response = self.client.table(self.users_table).insert(user_data).execute()
            print(f"✓ User '{user_id}' created successfully")
            return True
        except Exception as e:
            print(f"✗ Error creating user: {e}")
            return False
    
    def get_user(self, user_id: str) -> dict:
        """
        Get user profile by ID
        
        Args:
            user_id: User identifier
            
        Returns:
            User data dictionary or None if not found
        """
        try:
            response = self.client.table(self.users_table).select("*").eq("user_id", user_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"✗ Error retrieving user: {e}")
            return None
    
    def update_user(self, user_id: str, user_data: dict) -> bool:
        """Update user profile"""
        try:
            user_data['updated_at'] = datetime.now().isoformat()
            response = self.client.table(self.users_table).update(user_data).eq("user_id", user_id).execute()
            print(f"✓ User '{user_id}' updated successfully")
            return True
        except Exception as e:
            print(f"✗ Error updating user: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user profile"""
        try:
            self.client.table(self.users_table).delete().eq("user_id", user_id).execute()
            print(f"✓ User '{user_id}' deleted successfully")
            return True
        except Exception as e:
            print(f"✗ Error deleting user: {e}")
            return False
    
    # ==================== FACE EMBEDDING MANAGEMENT ====================
    
    def add_embedding(self, user_id: str, embedding: np.ndarray, image_path: str = None) -> bool:
        """
        Store face embedding for a user
        
        Args:
            user_id: User identifier
            embedding: Numpy array of face embedding
            image_path: Path to the original image (optional)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.flatten().tolist()
            
            embedding_data = {
                'user_id': user_id,
                'embedding': embedding_list,
                'embedding_dimension': len(embedding_list),
                'image_path': image_path,
                'created_at': datetime.now().isoformat()
            }
            
            response = self.client.table(self.embeddings_table).insert(embedding_data).execute()
            print(f"✓ Embedding stored for user '{user_id}'")
            return True
        except Exception as e:
            print(f"✗ Error storing embedding: {e}")
            return False
    
    def get_embeddings(self, user_id: str) -> list:
        """
        Retrieve all embeddings for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of embeddings (as numpy arrays) or empty list
        """
        try:
            response = self.client.table(self.embeddings_table).select("*").eq("user_id", user_id).execute()
            embeddings = []
            if response.data:
                for record in response.data:
                    embedding_array = parse_embedding(record['embedding'])
                    if embedding_array is not None:
                        embeddings.append({
                            'id': record.get('id'),
                            'embedding': embedding_array,
                            'created_at': record.get('created_at')
                        })
            return embeddings
        except Exception as e:
            print(f"✗ Error retrieving embeddings: {e}")
            return []
    
    def delete_embedding(self, embedding_id: int) -> bool:
        """Delete a specific embedding"""
        try:
            self.client.table(self.embeddings_table).delete().eq("id", embedding_id).execute()
            print(f"✓ Embedding deleted successfully")
            return True
        except Exception as e:
            print(f"✗ Error deleting embedding: {e}")
            return False
    
    def get_all_user_embeddings(self) -> dict:
        """
        Retrieve all embeddings grouped by user
        
        Returns:
            Dictionary with user_id as key and list of embeddings as value
        """
        try:
            response = self.client.table(self.embeddings_table).select("*").execute()
            embeddings_by_user = {}
            
            if response.data:
                for record in response.data:
                    user_id = record['user_id']
                    if user_id not in embeddings_by_user:
                        embeddings_by_user[user_id] = []
                    
                    embedding_array = parse_embedding(record['embedding'])
                    if embedding_array is not None:
                        embeddings_by_user[user_id].append({
                            'id': record.get('id'),
                            'embedding': embedding_array,
                            'created_at': record.get('created_at')
                        })
            
            return embeddings_by_user
        except Exception as e:
            print(f"✗ Error retrieving all embeddings: {e}")
            return {}
    
    # ==================== COMPARISON LOGGING ====================
    
    def log_comparison(self, user_id_1: str, user_id_2: str, similarity_score: float, match: bool) -> bool:
        """
        Log a face comparison operation
        
        Args:
            user_id_1: First user identifier
            user_id_2: Second user identifier
            similarity_score: Similarity score between embeddings
            match: Boolean indicating if faces match
        
        Returns:
            True if successful, False otherwise
        """
        try:
            comparison_data = {
                'user_id_1': user_id_1,
                'user_id_2': user_id_2,
                'similarity_score': float(similarity_score),
                'match': match,
                'compared_at': datetime.now().isoformat()
            }
            
            response = self.client.table(self.comparisons_table).insert(comparison_data).execute()
            print(f"✓ Comparison logged between '{user_id_1}' and '{user_id_2}'")
            return True
        except Exception as e:
            print(f"✗ Error logging comparison: {e}")
            return False
    
    def get_comparisons(self, user_id: str, limit: int = 100) -> list:
        """
        Get comparison history for a user
        
        Args:
            user_id: User identifier
            limit: Maximum number of records to retrieve
            
        Returns:
            List of comparison records
        """
        try:
            response = (self.client.table(self.comparisons_table)
                       .select("*")
                       .or_(f"user_id_1.eq.{user_id},user_id_2.eq.{user_id}")
                       .order("compared_at", desc=True)
                       .limit(limit)
                       .execute())
            return response.data if response.data else []
        except Exception as e:
            print(f"✗ Error retrieving comparisons: {e}")
            return []
    
    # ==================== UTILITY METHODS ====================
    
    def clear_all_data(self) -> bool:
        """
        Clear all data from all tables (use with caution!)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete in order of foreign key dependencies
            self.client.table(self.comparisons_table).delete().neq("id", 0).execute()
            self.client.table(self.embeddings_table).delete().neq("id", 0).execute()
            self.client.table(self.users_table).delete().neq("id", 0).execute()
            print("✓ All data cleared successfully")
            return True
        except Exception as e:
            print(f"✗ Error clearing data: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """Get database statistics"""
        try:
            users_response = self.client.table(self.users_table).select("count", count="exact").execute()
            embeddings_response = self.client.table(self.embeddings_table).select("count", count="exact").execute()
            comparisons_response = self.client.table(self.comparisons_table).select("count", count="exact").execute()
            
            return {
                'total_users': len(users_response.data) if users_response.data else 0,
                'total_embeddings': len(embeddings_response.data) if embeddings_response.data else 0,
                'total_comparisons': len(comparisons_response.data) if comparisons_response.data else 0
            }
        except Exception as e:
            print(f"✗ Error retrieving statistics: {e}")
            return {}
