# ── 실험: 인식 방식 비교 ──
#   A = 기존 (YOLO Arts 크롭 → CLIP top1)
#   B = 테두리 찾기 → 펴기 → 고정 비율 크롭 → CLIP top1
#   C = B의 CLIP 후보 20장을 SIFT 특징점 매칭으로 재정렬
#   python experiments/recog_test.py
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from recognizer import Recognizer
from card_warp import ART_NORMAL, ART_PENDULUM, crop_ratio, find_card_quad, warp_card

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

sift    = cv2.SIFT_create(nfeatures=1000)
matcher = cv2.BFMatcher(cv2.NORM_L2)
_db_feat_cache = {}


def sift_feat(img_bgr, size=320):
    h, w = img_bgr.shape[:2]
    s    = size / max(h, w)
    gray = cv2.cvtColor(cv2.resize(img_bgr, (int(w * s), int(h * s))), cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)


def db_feat(card_id):
    if card_id not in _db_feat_cache:
        img = cv2.imread(str(config.IMG_DIR / f"{card_id}.jpg"))
        _db_feat_cache[card_id] = sift_feat(img) if img is not None else (None, None)
    return _db_feat_cache[card_id]


def inliers(q, card_id):
    """같은 그림이면 기하학적으로 일관된 매칭점(inlier)이 많이 나옴"""
    kq, dq = q
    kd, dd = db_feat(card_id)
    if dq is None or dd is None or len(dq) < 8 or len(dd) < 8:
        return 0
    good = [m for m, n in (p for p in matcher.knnMatch(dq, dd, k=2) if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good) // 2
    src = np.float32([kq[m.queryIdx].pt for m in good])
    dst = np.float32([kd[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main():
    r = Recognizer()
    rows = []
    for p in sorted(glob.glob(str(config.FEEDBACK_DIR / "snap_*.jpg"))):
        frame = cv2.imread(p)
        row = {"file": Path(p).name}

        # A: 기존 방식
        pil, preds = r.detect(frame)
        arts = r.best_box(preds, "Arts")
        if arts:
            a = r.clip_search(pil.crop(arts["box"]), k=1)[0]
            row.update(A_id=a["card_id"], A_name=a["name_ko"] or a["name_en"], A_score=round(a["score"], 3))

        # B / C: 펴서 자르기 (+ 특징점 재정렬)
        quad = find_card_quad(frame)
        if quad is not None:
            card  = warp_card(frame, quad)
            crops = [crop_ratio(card, ART_NORMAL), crop_ratio(card, ART_PENDULUM)]
            cands = {}
            for crop in crops:
                for c in r.clip_search(to_pil(crop), k=20):
                    if c["card_id"] not in cands or c["score"] > cands[c["card_id"]]["score"]:
                        cands[c["card_id"]] = c
            b = max(cands.values(), key=lambda c: c["score"])
            row.update(B_id=b["card_id"], B_name=b["name_ko"] or b["name_en"], B_score=round(b["score"], 3))

            feats = [sift_feat(crop) for crop in crops]
            for c in cands.values():
                c["inliers"] = max(inliers(f, c["card_id"]) for f in feats)
            ranked = sorted(cands.values(), key=lambda c: (c["inliers"], c["score"]), reverse=True)
            best, second = ranked[0], ranked[1]
            row.update(C_id=best["card_id"], C_name=best["name_ko"] or best["name_en"],
                       C_inliers=best["inliers"], C_2nd=second["inliers"])
            cv2.imwrite(str(OUT / f"crop_{row['file']}"), crops[0])

        rows.append(row)
        print({k: v for k, v in row.items() if not k.endswith("_id")})

    keys = ["file", "A_id", "A_name", "A_score", "B_id", "B_name", "B_score",
            "C_id", "C_name", "C_inliers", "C_2nd"]
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
