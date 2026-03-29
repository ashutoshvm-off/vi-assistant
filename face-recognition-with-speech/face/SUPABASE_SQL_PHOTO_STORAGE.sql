-- ============================================
-- SUPABASE SQL SETUP FOR PHOTO & DETAILS STORAGE
-- ============================================
-- Run these commands in your Supabase SQL Editor
-- Go to: SQL Editor → New Query → Paste & Run

-- ⚠️ IMPORTANT: Before running this script, create the storage bucket manually:
-- 1. Go to Supabase Dashboard → Storage
-- 2. Click "Create Bucket"
-- 3. Name: face-photos
-- 4. Mark as Public ✓
-- 5. Click Create
-- Then run this SQL script

-- ============================================
-- ENABLE PGVECTOR EXTENSION (for vector embeddings)
-- ============================================
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================
-- 0. CREATE USERS TABLE (if not exists)
-- ============================================
CREATE TABLE IF NOT EXISTS users (
  id BIGSERIAL PRIMARY KEY,
  user_id TEXT UNIQUE NOT NULL,
  name TEXT,
  email TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_user_id ON users(user_id);


-- ============================================
-- 1. CREATE FACE EMBEDDINGS TABLE (dependency)
-- ============================================
CREATE TABLE IF NOT EXISTS face_embeddings (
  id BIGSERIAL PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  embedding VECTOR(512),
  embedding_dimension INTEGER,
  image_path TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embeddings_user_id ON face_embeddings(user_id);


-- ============================================
-- 2. CREATE PHOTOS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS photos (
  id BIGSERIAL PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  photo_name TEXT NOT NULL,
  file_path TEXT,  -- Path in Supabase Storage
  storage_bucket TEXT DEFAULT 'face-photos',  -- Storage bucket name
  file_size INTEGER,  -- Size in bytes
  file_type TEXT,  -- MIME type (e.g., 'image/jpeg')
  width INTEGER,  -- Image width
  height INTEGER,  -- Image height
  embedding_id BIGINT REFERENCES face_embeddings(id) ON DELETE SET NULL,
  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_photos_user_id ON photos(user_id);
CREATE INDEX idx_photos_uploaded_at ON photos(uploaded_at DESC);
CREATE INDEX idx_photos_file_path ON photos(file_path);


-- ============================================
-- 3. CREATE PHOTO DETAILS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS photo_details (
  id BIGSERIAL PRIMARY KEY,
  photo_id BIGINT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  detection_confidence FLOAT8,  -- Confidence score for face detection (0-1)
  face_quality FLOAT8,  -- Quality score of detected face
  is_frontal BOOLEAN DEFAULT FALSE,  -- Is face frontal view
  estimated_age INT,  -- Estimated age from face
  gender TEXT,  -- Estimated gender
  emotion TEXT,  -- Detected emotion (happy, sad, etc.)
  face_coordinates JSONB,  -- Bounding box: {"x": int, "y": int, "width": int, "height": int}
  face_landmarks JSONB,  -- Facial landmarks: {"eyes": [...], "nose": [...], "mouth": [...]}
  lighting_quality FLOAT8,  -- Lighting quality score
  blur_score FLOAT8,  -- Blur detection score (0-1, lower is better)
  is_usable BOOLEAN DEFAULT TRUE,  -- Can be used for recognition
  notes TEXT,  -- Additional notes/metadata
  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_photo_details_photo_id ON photo_details(photo_id);
CREATE INDEX idx_photo_details_user_id ON photo_details(user_id);
CREATE INDEX idx_photo_details_is_usable ON photo_details(is_usable);
CREATE INDEX idx_photo_details_detection_confidence ON photo_details(detection_confidence DESC);


-- ============================================
-- 4. CREATE PHOTO TAGS TABLE (Optional)
-- ============================================
CREATE TABLE IF NOT EXISTS photo_tags (
  id BIGSERIAL PRIMARY KEY,
  photo_id BIGINT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  tag_name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_photo_tags_photo_id ON photo_tags(photo_id);
CREATE INDEX idx_photo_tags_tag_name ON photo_tags(tag_name);


-- ============================================
-- 5. CREATE PHOTO COMPARISON HISTORY TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS photo_comparisons (
  id BIGSERIAL PRIMARY KEY,
  photo_id_1 BIGINT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  photo_id_2 BIGINT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  user_id_1 TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  user_id_2 TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  similarity_score FLOAT8 NOT NULL,
  match BOOLEAN NOT NULL,
  compared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_photo_comparisons_photo_id_1 ON photo_comparisons(photo_id_1);
CREATE INDEX idx_photo_comparisons_photo_id_2 ON photo_comparisons(photo_id_2);
CREATE INDEX idx_photo_comparisons_user_ids ON photo_comparisons(user_id_1, user_id_2);


-- ============================================
-- 6. CREATE STORAGE POLICIES (Row Level Security)
-- ============================================
ALTER TABLE photos ENABLE ROW LEVEL SECURITY;
ALTER TABLE photo_details ENABLE ROW LEVEL SECURITY;
ALTER TABLE photo_tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE photo_comparisons ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own photos
CREATE POLICY "Users can select own photos" ON photos
  FOR SELECT
  USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own photos" ON photos
  FOR INSERT
  WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can update own photos" ON photos
  FOR UPDATE
  USING (auth.uid()::text = user_id);

CREATE POLICY "Users can delete own photos" ON photos
  FOR DELETE
  USING (auth.uid()::text = user_id);

-- Policy: Users can access photo details for their own photos
CREATE POLICY "Users can select own photo details" ON photo_details
  FOR SELECT
  USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own photo details" ON photo_details
  FOR INSERT
  WITH CHECK (auth.uid()::text = user_id);

-- ============================================
-- 7. CREATE VIEWS FOR EASY QUERYING
-- ============================================

-- View: Photos with their details
CREATE OR REPLACE VIEW photo_with_details AS
SELECT 
  p.id,
  p.user_id,
  p.photo_name,
  p.file_path,
  p.file_size,
  p.file_type,
  p.width,
  p.height,
  p.uploaded_at,
  pd.detection_confidence,
  pd.face_quality,
  pd.is_frontal,
  pd.estimated_age,
  pd.gender,
  pd.emotion,
  pd.is_usable,
  pd.processed_at
FROM photos p
LEFT JOIN photo_details pd ON p.id = pd.photo_id
ORDER BY p.uploaded_at DESC;

-- View: Quality photos (high quality & usable)
CREATE OR REPLACE VIEW quality_photos AS
SELECT 
  p.id,
  p.user_id,
  p.photo_name,
  p.file_path,
  pd.detection_confidence,
  pd.face_quality,
  p.uploaded_at
FROM photos p
JOIN photo_details pd ON p.id = pd.photo_id
WHERE pd.is_usable = TRUE
  AND pd.detection_confidence > 0.8
  AND pd.face_quality > 0.7
ORDER BY pd.face_quality DESC;


-- ============================================
-- 8. OPTIONAL: FUNCTIONS FOR COMMON OPERATIONS
-- ============================================

-- Function: Get best photo for a user
CREATE OR REPLACE FUNCTION get_best_photo(user_id_param TEXT)
RETURNS TABLE (
  id BIGINT,
  photo_name TEXT,
  file_path TEXT,
  face_quality FLOAT8,
  uploaded_at TIMESTAMP
) AS $$
SELECT 
  p.id,
  p.photo_name,
  p.file_path,
  pd.face_quality,
  p.uploaded_at
FROM photos p
JOIN photo_details pd ON p.id = pd.photo_id
WHERE p.user_id = user_id_param
  AND pd.is_usable = TRUE
ORDER BY pd.face_quality DESC
LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Function: Get photos count by quality
CREATE OR REPLACE FUNCTION count_photos_by_quality(user_id_param TEXT)
RETURNS TABLE (
  quality_level TEXT,
  count BIGINT
) AS $$
SELECT 
  CASE 
    WHEN face_quality > 0.8 THEN 'High'
    WHEN face_quality > 0.6 THEN 'Medium'
    ELSE 'Low'
  END as quality_level,
  COUNT(*) as count
FROM photos p
JOIN photo_details pd ON p.id = pd.photo_id
WHERE p.user_id = user_id_param
GROUP BY quality_level;
$$ LANGUAGE SQL STABLE;


-- ============================================
-- 9. SAMPLE DATA (for testing - remove later)
-- ============================================

-- Uncomment to add sample photo (replace 'john' with actual user_id):
/*
INSERT INTO photos (user_id, photo_name, file_size, file_type, width, height, file_path)
VALUES ('john', 'profile_pic.jpg', 2048000, 'image/jpeg', 1920, 1080, 'face-photos/john/profile_pic.jpg');

INSERT INTO photo_details (photo_id, user_id, detection_confidence, face_quality, is_frontal, blur_score, is_usable)
VALUES (1, 'john', 0.95, 0.88, true, 0.1, true);
*/

-- ============================================
-- 10. STORAGE BUCKET POLICIES (Face-Photos Bucket)
-- ============================================
-- ⚠️ IMPORTANT: Run this section SEPARATELY after:
-- 1. Go to Supabase Dashboard → Storage
-- 2. Click "Create Bucket" → Name: face-photos → Mark as Public ✓
-- 3. Then run ONLY this section below

-- Uncomment and run these policies AFTER creating the storage bucket:
/*
CREATE POLICY "Users can upload photos to their folder" ON storage.objects
  FOR INSERT TO authenticated
  WITH CHECK (
    bucket_id = 'face-photos' AND
    (auth.uid()::text = (string_to_array(name, '/'))[1])
  );

CREATE POLICY "Users can update their photos" ON storage.objects
  FOR UPDATE TO authenticated
  USING (
    bucket_id = 'face-photos' AND
    (auth.uid()::text = (string_to_array(name, '/'))[1])
  )
  WITH CHECK (
    bucket_id = 'face-photos' AND
    (auth.uid()::text = (string_to_array(name, '/'))[1])
  );

CREATE POLICY "Users can delete their photos" ON storage.objects
  FOR DELETE TO authenticated
  USING (
    bucket_id = 'face-photos' AND
    (auth.uid()::text = (string_to_array(name, '/'))[1])
  );

CREATE POLICY "Anyone can read public photos" ON storage.objects
  FOR SELECT
  USING (bucket_id = 'face-photos');
*/


-- ============================================
-- 11. VERIFICATION - Run these queries to verify setup
-- ============================================

-- Check all tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check storage bucket exists (run only if storage is enabled)
-- SELECT name, id, public FROM storage.buckets WHERE name = 'face-photos';

-- Check storage policies (run only if storage is enabled)
-- SELECT * FROM pg_policies WHERE tablename = 'objects' AND schemaname = 'storage';

-- Verify face_embeddings table has vector type
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'face_embeddings' AND column_name = 'embedding';

-- Check all indexes
SELECT indexname FROM pg_indexes 
WHERE tablename IN ('photos', 'photo_details', 'face_embeddings', 'users')
ORDER BY tablename;
