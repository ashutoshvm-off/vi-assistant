"""
Debug script for Face Recognition System
Run this to check database state and diagnose issues
"""

from supabase_service import SupabaseService
from photo_service import PhotoService

def debug_database():
    print("\n" + "="*70)
    print("FACE RECOGNITION DATABASE DEBUG")
    print("="*70)
    
    supabase = SupabaseService()
    photo = PhotoService()
    
    # Check connection
    print("\n📡 Checking Supabase connection...")
    try:
        # Try a simple query
        response = supabase.client.table('users').select("count", count="exact").execute()
        print("   ✓ Connection successful")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return
    
    # Check users
    print("\n👥 Checking USERS table...")
    try:
        response = supabase.client.table('users').select("*").execute()
        if response.data:
            print(f"   ✓ Found {len(response.data)} users:")
            for user in response.data:
                print(f"      - {user.get('user_id', 'N/A')} ({user.get('name', 'N/A')})")
        else:
            print("   ⚠️  No users found in database")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Check embeddings
    print("\n🧠 Checking FACE_EMBEDDINGS table...")
    try:
        response = supabase.client.table('face_embeddings').select("*").execute()
        if response.data:
            print(f"   ✓ Found {len(response.data)} embeddings:")
            for emb in response.data:
                user_id = emb.get('user_id', 'N/A')
                dim = emb.get('embedding_dimension', 'N/A')
                created = emb.get('created_at', 'N/A')[:10] if emb.get('created_at') else 'N/A'
                embedding = emb.get('embedding')
                emb_status = f"✓ {len(embedding)} values" if embedding else "✗ EMPTY"
                print(f"      - User: {user_id}, Dim: {dim}, Created: {created}, Embedding: {emb_status}")
        else:
            print("   ⚠️  No embeddings found in database")
            print("   💡 This is why recognition fails!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Check photos
    print("\n📷 Checking PHOTOS table...")
    try:
        response = supabase.client.table('photos').select("*").execute()
        if response.data:
            print(f"   ✓ Found {len(response.data)} photos:")
            for photo in response.data[:5]:  # Show first 5
                print(f"      - {photo.get('photo_name', 'N/A')} (User: {photo.get('user_id', 'N/A')})")
        else:
            print("   ⚠️  No photos found in database")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test get_all_user_embeddings
    print("\n🔍 Testing get_all_user_embeddings()...")
    try:
        embeddings = supabase.get_all_user_embeddings()
        if embeddings:
            print(f"   ✓ Retrieved embeddings for {len(embeddings)} users:")
            for user_id, embs in embeddings.items():
                print(f"      - {user_id}: {len(embs)} embeddings")
        else:
            print("   ⚠️  get_all_user_embeddings() returned EMPTY!")
            print("   💡 This is the root cause of recognition failure")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Check storage bucket
    print("\n📦 Checking Storage Bucket...")
    try:
        files = supabase.client.storage.from_('face-photos').list()
        if files:
            print(f"   ✓ Found {len(files)} items in storage")
            for f in files[:5]:
                print(f"      - {f.get('name', 'N/A')}")
        else:
            print("   ⚠️  Storage bucket is empty or not accessible")
    except Exception as e:
        print(f"   ✗ Error (might be RLS policy): {e}")
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    debug_database()
