# ── 카드 인식 ──
#   1. 카드 테두리 찾아서 펴기 → 고정 비율로 일러스트 크롭 (못 찾으면 YOLO Arts 크롭)
#   2. CLIP 벡터 검색으로 후보 뽑기
#   3. SIFT 특징점 매칭으로 재정렬 + 확신 여부 판단
#   (선택) EasyOCR 카드명 — config.USE_OCR
import pathlib
import platform

import cv2
import numpy as np
import torch
import clip
from PIL import Image

import config
import db
import feature_match
from card_warp import art_crops, find_card_quad, warp_card


def load_yolo():
    # best.pt는 리눅스(Colab)에서 학습돼서 PosixPath가 저장돼 있음 → 윈도우에선 로딩 중에만 바꿔치기
    posix = pathlib.PosixPath
    if platform.system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.hub.load("ultralytics/yolov5", "custom", path=str(config.YOLO_PATH))
    finally:
        pathlib.PosixPath = posix


def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class Recognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.yolo = None
        if config.USE_YOLO or config.USE_OCR:   # OCR도 YOLO의 Cname 박스가 필요
            print("YOLOv5 모델 로딩 중...")
            self.yolo = load_yolo()
            self.yolo.conf = config.YOLO_CONF
            print("YOLOv5 로드 완료!")

        print("CLIP 로딩 중...")
        self.clip_model, self.clip_prep = clip.load(config.CLIP_MODEL, device=self.device)
        print(f"CLIP 로드 완료 ({self.device})")

        self.reader = None
        if config.USE_OCR:
            import easyocr
            print("OCR 로딩 중...")
            self.reader = easyocr.Reader(["ko", "en"], gpu=torch.cuda.is_available())
            print("OCR 로드 완료")

        print("CLIP 벡터 DB 로딩 중...")
        data          = np.load(config.VEC_PATH, allow_pickle=True)
        self.vec_ids  = data["card_ids"]
        self.names_en = data["name_ens"]
        self.names_ko = data["name_kos"]
        self.vectors  = torch.tensor(data["vectors"], dtype=torch.float32).to(self.device)
        print(f"{len(self.vec_ids)}장 벡터 로드 완료")

    # ══════════════════════════════════════
    # YOLO 탐지 → [{"class", "confidence", "box": (x1,y1,x2,y2)}]
    # ══════════════════════════════════════
    def detect(self, frame):
        results = self.yolo(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        predictions = [{
            "class":      self.yolo.names[int(cls)],
            "confidence": float(conf),
            "box":        tuple(max(0, int(v)) for v in box),
        } for *box, conf, cls in results.xyxy[0]]

        detected = [(p["class"], f"{p['confidence']*100:.1f}%") for p in predictions]
        print(f"  YOLO 탐지: {detected}")
        return predictions

    @staticmethod
    def best_box(predictions, class_name, min_conf=0.0):
        boxes = [p for p in predictions
                 if p["class"] == class_name and p["confidence"] > min_conf]
        return max(boxes, key=lambda p: p["confidence"]) if boxes else None

    # ══════════════════════════════════════
    # 1. 일러스트 크롭
    # ══════════════════════════════════════
    def crop_arts(self, frame):
        """반환: (크롭 리스트[BGR], 외곽선 4점, YOLO 결과, 펴진 카드) — 없는 값은 None"""
        quad = find_card_quad(frame)
        if quad is not None:
            card = warp_card(frame, quad)
            return art_crops(card), quad, None, card

        # 테두리 못 찾음 → YOLO Arts 박스로 대체
        if self.yolo is None:
            return [], None, None, None
        predictions = self.detect(frame)
        arts = self.best_box(predictions, "Arts")
        if not arts:
            return [], None, predictions, None
        x1, y1, x2, y2 = arts["box"]
        outline = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        return [frame[y1:y2, x1:x2]], outline, predictions, None

    # ══════════════════════════════════════
    # 2. CLIP 벡터 검색
    # ══════════════════════════════════════
    def clip_search(self, crop_bgr, k):
        img_tensor = self.clip_prep(to_pil(crop_bgr)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_vec  = self.clip_model.encode_image(img_tensor).float()
            query_vec /= query_vec.norm(dim=-1, keepdim=True)

        scores, idxs = (query_vec @ self.vectors.T).squeeze(0).topk(k)
        return [{
            "card_id": str(self.vec_ids[i]),
            "name_en": str(self.names_en[i]),
            "name_ko": str(self.names_ko[i]),
            "score":   s,
        } for i, s in zip(idxs.tolist(), scores.tolist())]

    # ══════════════════════════════════════
    # 3. 한 프레임 인식
    # ══════════════════════════════════════
    def identify(self, frame):
        """
        반환: None 또는 dict(
            candidates: 재정렬된 후보 (각각 card_id, name_en, name_ko, score, inliers),
            confident:  1위를 믿을 만한지,
            arts:       대표 크롭(BGR), outline: 외곽선 4점, predictions: YOLO 결과(대체 시))
        """
        crops, outline, predictions, card = self.crop_arts(frame)
        if not crops:
            return None

        ranked, confident = self.rank(crops)
        # 확신이 없고 카드를 펴서 잘랐으면 → 위아래가 뒤집혀 들어온 카드일 수 있으니 180° 돌려서 한 번 더
        if not confident and card is not None:
            flipped = art_crops(cv2.rotate(card, cv2.ROTATE_180))
            ranked_f, confident_f = self.rank(flipped)
            if (confident_f, ranked_f[0]["inliers"]) > (confident, ranked[0]["inliers"]):
                print("  (180° 뒤집힌 카드로 인식)")
                crops, ranked, confident = flipped, ranked_f, confident_f

        return {
            "candidates":  ranked,
            "confident":   confident,
            "arts":        crops[0],
            "outline":     outline,
            "predictions": predictions,
        }

    def rank(self, crops):
        """크롭들 → (CLIP 후보를 특징점 매칭으로 재정렬한 리스트, 확신 여부)"""
        # 크롭마다 CLIP 후보를 모아서 합치기 (일반/펜듈럼 비율 둘 다)
        cands = {}
        for crop in crops:
            for c in self.clip_search(crop, config.CLIP_CANDIDATES):
                if c["card_id"] not in cands or c["score"] > cands[c["card_id"]]["score"]:
                    cands[c["card_id"]] = c

        feats = [feature_match.features(crop) for crop in crops]
        for c in cands.values():
            c["inliers"] = max(feature_match.count_inliers(f, c["card_id"]) for f in feats)

        # 매칭점이 기준 이상인 후보만 매칭점 순, 나머지(=노이즈 수준)는 CLIP 점수 순
        ranked = sorted(cands.values(), reverse=True, key=lambda c: (
            c["inliers"] if c["inliers"] >= config.MIN_INLIERS else 0, c["score"]))
        first  = ranked[0]["inliers"]
        second = max((c["inliers"] for c in ranked[1:]), default=0)
        confident = first >= config.MIN_INLIERS and first >= config.INLIER_RATIO * second
        return ranked, confident

    # ══════════════════════════════════════
    # (선택) OCR 카드명
    # ══════════════════════════════════════
    def ocr_card_name(self, frame, predictions):
        if self.reader is None:
            return None
        if predictions is None:
            predictions = self.detect(frame)
        best = self.best_box(predictions, "Cname", config.OCR_MIN_CONF)
        if not best:
            return None

        x1, y1, x2, y2 = best["box"]
        crop      = cv2.resize(frame[y1:y2, x1:x2], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp     = cv2.filter2D(gray, -1, kernel)
        _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            text = " ".join(self.reader.readtext(binary, detail=0)).strip()
            print(f"  OCR: '{text}'")
            row = db.find_card_by_name(text)
            return (row[2] or row[1]) if row else None
        except Exception as e:
            print(f"  OCR 오류: {e}")
            return None

    # ══════════════════════════════════════
    # 4. 연속 촬영 → 가장 좋은 결과 (확신하면 바로 멈춤)
    # ══════════════════════════════════════
    def capture_best(self, cap, n=config.SHOTS_PER_SCAN):
        """반환: identify() 결과 + frame, ocr_name 또는 None"""
        best = None
        for i in range(n):
            ret, frame = cap.read()
            if not ret:
                continue

            res = self.identify(frame)
            if res is None:
                print(f"  [{i+1}/{n}] 카드 못 찾음")
                continue

            top = res["candidates"][0]
            print(f"  [{i+1}/{n}] {top['name_ko'] or top['name_en']} "
                  f"(매칭점 {top['inliers']}, CLIP {top['score']*100:.1f}%)"
                  f"{' ✔ 확신' if res['confident'] else ''}")

            res["frame"] = frame
            key = (res["confident"], top["inliers"], top["score"])
            if best is None or key > best["_key"]:
                best = res | {"_key": key}
            if res["confident"]:
                break

        if best is None:
            return None
        best["ocr_name"] = self.ocr_card_name(best["frame"], best["predictions"])
        return best
