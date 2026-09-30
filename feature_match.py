# ── SIFT 특징점 매칭: "정말 같은 그림인가" 검증 ──
# 같은 일러스트면 기하학적으로 일관된 매칭점(inlier)이 수십~수백 개 나오고,
# 다른 그림이면 한 자릿수라서 CLIP보다 차이가 확실하게 벌어짐
import cv2
import numpy as np

import config

_sift    = cv2.SIFT_create(nfeatures=1000)
_matcher = cv2.BFMatcher(cv2.NORM_L2)
_db_cache = {}


def features(img_bgr, size=320):
    h, w = img_bgr.shape[:2]
    s    = size / max(h, w)
    gray = cv2.cvtColor(cv2.resize(img_bgr, (int(w * s), int(h * s))), cv2.COLOR_BGR2GRAY)
    return _sift.detectAndCompute(gray, None)


def db_features(card_id):
    """DB 카드 이미지의 특징점 (한 번 계산하면 캐시)"""
    if card_id not in _db_cache:
        img = cv2.imread(str(config.IMG_DIR / f"{card_id}.jpg"))
        _db_cache[card_id] = features(img) if img is not None else (None, None)
    return _db_cache[card_id]


def count_inliers(query_feat, card_id):
    kq, dq = query_feat
    kd, dd = db_features(card_id)
    if dq is None or dd is None or len(dq) < 8 or len(dd) < 8:
        return 0
    # Lowe ratio test
    good = [m for m, n in (p for p in _matcher.knnMatch(dq, dd, k=2) if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good) // 2
    src = np.float32([kq[m.queryIdx].pt for m in good])
    dst = np.float32([kd[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0
