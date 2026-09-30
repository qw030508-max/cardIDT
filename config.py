# ── 경로 / 설정 (모든 스크립트가 여기서 가져다 씀) ──
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
DB_PATH      = BASE_DIR / "yugioh.db"
IMG_DIR      = BASE_DIR / "card_images"        # 일러스트 크롭 이미지 (CLIP 비교용)
FULL_IMG_DIR = BASE_DIR / "card_images_full"   # 카드 전체 이미지 (필요할 때만)
VEC_PATH     = BASE_DIR / "clip_vectors.npz"
FEEDBACK_DIR = BASE_DIR / "feedback_images"
YOLO_PATH    = BASE_DIR / "best.pt"
FONT_PATH    = "C:/Windows/Fonts/malgun.ttf"

API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

# ── 카메라 ──
CAMERA_BACKEND = "opencv"   # 라즈베리 파이 카메라 모듈이면 "picamera2"
CAMERA_INDEX  = 1        # (opencv일 때) DroidCam이면 1, 웹캠이면 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
SHOTS_PER_SCAN = 3       # Space 한 번에 연속 촬영 수

# ── 인식 ──
CLIP_MODEL      = "ViT-B/32"
USE_YOLO        = True         # 카드 테두리를 못 찾았을 때 YOLO Arts 크롭으로 대체 (기계는 끄는 걸 추천)
YOLO_CONF       = 0.4
CLIP_CANDIDATES = 20           # CLIP으로 뽑을 후보 수 → 특징점 매칭으로 재정렬
TOP_K           = 5            # 확인창에 보여줄 후보 수

# 확신 기준: 1위 매칭점이 MIN_INLIERS 이상이고 2위의 INLIER_RATIO배 이상
MIN_INLIERS  = 20
INLIER_RATIO = 2.5
AUTO_SAVE    = True            # 확신하면 확인창 없이 바로 재고 저장 (U로 되돌리기)

# ── 분류기(기계) ──
# 통 0, 1, 2에 넣을 분류 (monster / spell / trap, 중복 X). 통 3 = 나머지 + 확신 없음
BIN_FILTERS = ["monster", "spell", "trap"]

# ── OCR (기본 꺼짐: 그림 매칭 위주, 켜면 easyocr 필요) ──
USE_OCR      = False
OCR_MIN_CONF = 0.4
