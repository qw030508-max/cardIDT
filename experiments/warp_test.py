# ── 실험: 카드 테두리 찾기 → 원근 보정 → 고정 비율로 일러스트 크롭 ──
#   python experiments/warp_test.py
#   결과: experiments/out/ (사진마다 [원본+테두리 | 펴진 카드 | 크롭] 이미지)
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

from card_warp import ART_NORMAL, CARD_H, crop_ratio, find_card_quad, warp_card


def main():
    paths = sorted(glob.glob(str(config.FEEDBACK_DIR / "snap_*.jpg")))
    found = 0
    for p in paths:
        frame = cv2.imread(p)
        quad  = find_card_quad(frame)
        vis   = frame.copy()
        panels = []
        if quad is not None:
            found += 1
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 3)
            card = warp_card(frame, quad)
            art  = cv2.resize(crop_ratio(card, ART_NORMAL), (300, 300))
            panels = [card, np.vstack([art, np.zeros((CARD_H - 300, 300, 3), np.uint8)])]
        vis = cv2.resize(vis, (int(1280 * CARD_H / 720), CARD_H))
        cv2.imwrite(str(OUT / Path(p).name), np.hstack([vis] + panels))
        print(f"{Path(p).name}: {'OK' if quad is not None else '카드 못 찾음'}")
    print(f"\n{found}/{len(paths)}장에서 카드 테두리 찾음 → {OUT}")


if __name__ == "__main__":
    main()
