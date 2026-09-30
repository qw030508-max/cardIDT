# ── 카드 테두리 찾기 → 원근 보정(반듯하게 펴기) → 고정 비율로 일러스트 크롭 ──
import cv2
import numpy as np

CARD_W, CARD_H = 421, 614          # 카드 비율 59 x 86 mm
ASPECT = CARD_W / CARD_H           # ≈ 0.686

# 카드 안 일러스트 위치 (x1, y1, x2, y2 비율)
# full 이미지와 card_images를 템플릿 매칭해서 측정한 값 — card_images 크롭과 같은 영역
ART_NORMAL   = (0.117, 0.181, 0.884, 0.707)
ART_PENDULUM = (0.062, 0.179, 0.937, 0.945)


def order_corners(pts):
    """4점을 좌상, 우상, 우하, 좌하 순서로"""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def _side_lengths(quad):
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return w, h


def _quad_score(quad, img_area):
    """카드처럼 생긴 사각형일수록 높은 점수 (아니면 -1)"""
    w, h = _side_lengths(quad)
    if w < 20 or h < 20:
        return -1
    area = cv2.contourArea(quad)
    if area < img_area * 0.005 or area > img_area * 0.9:
        return -1
    aspect_err = abs(min(w, h) / max(w, h) - ASPECT)
    if aspect_err > 0.12:
        return -1
    return area * (1 - aspect_err * 5)


def find_card_quad(frame):
    """프레임에서 가장 카드처럼 생긴 사각형(4점) 찾기. 없으면 None"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    img_area = frame.shape[0] * frame.shape[1]

    # 조명/배경마다 잘 먹히는 전처리가 달라서 여러 개 시도
    edge_maps = [cv2.dilate(cv2.Canny(blur, lo, hi), None, iterations=1)
                 for lo, hi in [(30, 100), (50, 150), (80, 200)]]
    edge_maps.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 21, 5))

    best, best_score = None, 0
    for edges in edge_maps:
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < img_area * 0.005:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = order_corners(approx)
            else:
                # 꼭짓점이 딱 4개로 안 나오면 최소 외접 사각형 (윤곽이 꽉 찬 경우만)
                quad = order_corners(cv2.boxPoints(cv2.minAreaRect(cnt)))
                if cv2.contourArea(cnt) / max(cv2.contourArea(quad), 1) < 0.85:
                    continue
            score = _quad_score(quad, img_area)
            if score > best_score:
                best, best_score = quad, score
    return best


def warp_card(frame, quad):
    """사각형 영역을 CARD_W x CARD_H 세로 카드로 펴기"""
    w, h = _side_lengths(quad)
    if w > h:   # 카드가 옆으로 누워 있으면 세로로 세우기
        tl, tr, br, bl = quad
        quad = np.array([bl, tl, tr, br])
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(frame, M, (CARD_W, CARD_H))


def crop_ratio(card, box):
    x1, y1, x2, y2 = box
    h, w = card.shape[:2]
    return card[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def art_crops(card):
    """펴진 카드에서 일러스트 후보 크롭 (일반 / 펜듈럼)"""
    return [crop_ratio(card, ART_NORMAL), crop_ratio(card, ART_PENDULUM)]
