"""
Configuration settings for Hand OCR Reader.
API key is read from the OCR_SPACE_API_KEY environment variable.
Set it in your shell:  export OCR_SPACE_API_KEY="your_key_here"
Get a free key (25,000 req/month) at: https://ocr.space/ocrapi/freekey
"""

import os

# ==================== API ====================
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "helloworld")
OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"
OCR_ENGINE   = 2        # 1 = fast/multilingual, 2 = best for numbers/special chars
OCR_LANGUAGE = "eng"    # eng, fre, ger, spa, chi_sim, jpn, …

# ==================== CAMERA ====================
CAMERA_INDEX = int(os.getenv("VOICE_CAMERA_INDEX", "1"))  # 0 = built-in webcam, 1 = DroidCam / external

# ==================== TIMING ====================
SPEAK_COOLDOWN   = 3.0  # seconds between TTS plays
STABLE_TIME      = 0.8  # seconds finger must stay still before OCR fires
STABILITY_THRESHOLD = 6 # pixel-movement tolerance for "stable"
OCR_INTERVAL     = 1.5  # minimum seconds between successive API calls
OCR_TIMEOUT      = 10   # HTTP request timeout (seconds)

# ==================== ROI ====================
ROI_WIDTH  = 320        # pixels wide
ROI_HEIGHT = 90         # pixels tall (above finger tip)

# ==================== OCR ====================
OCR_SCALE_FACTOR  = 2.0 # upscale factor before sending to API
OCR_JPEG_QUALITY  = 78  # JPEG compression (75-85 is plenty for OCR)
MIN_TEXT_LENGTH   = 3   # discard results shorter than this

# ==================== HAND TRACKING ====================
MAX_HANDS                 = 1
MIN_DETECTION_CONFIDENCE  = 0.7
MIN_TRACKING_CONFIDENCE   = 0.7

# ==================== DISPLAY ====================
WINDOW_NAME      = "Hand OCR Reader"
OCR_DEBUG_WINDOW = "OCR INPUT"