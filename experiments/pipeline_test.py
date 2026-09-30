# ── 본 코드(recognizer.capture_best)를 저장된 사진으로 돌려보기 ──
#   python experiments/pipeline_test.py
import csv
import glob
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from recognizer import Recognizer

OUT = Path(__file__).resolve().parent / "out"


class FakeCap:
    """카메라 대신 사진 1장을 계속 돌려줌"""
    def __init__(self, img):
        self.img = img

    def read(self):
        return True, self.img


def main():
    r = Recognizer()
    exp_path = OUT / "results.csv"
    old = {row["file"]: row for row in csv.DictReader(open(exp_path, encoding="utf-8-sig"))} if exp_path.exists() else {}

    stats = {"confident": 0, "uncertain": 0, "none": 0, "agree_with_exp": 0, "exp_conf": 0}
    times = []
    for p in sorted(glob.glob(str(config.FEEDBACK_DIR / "snap_*.jpg"))):
        name = Path(p).name
        t = time.time()
        res = r.capture_best(FakeCap(cv2.imread(p)))
        times.append(time.time() - t)
        if res is None:
            stats["none"] += 1
            print(name, "-> 없음")
            continue
        top = res["candidates"][0]
        stats["confident" if res["confident"] else "uncertain"] += 1
        o = old.get(name)
        if o and o["C_id"] and int(o["C_inliers"]) >= 20 and int(o["C_inliers"]) >= 2.5 * int(o["C_2nd"]):
            stats["exp_conf"] += 1
            stats["agree_with_exp"] += top["card_id"] == o["C_id"]
        print(name, "->", top["name_ko"] or top["name_en"], top["inliers"],
              "✔" if res["confident"] else "?", f"{times[-1]:.2f}s")
    print(stats)
    print(f"평균 {sum(times)/len(times):.2f}s / 최대 {max(times):.2f}s")


if __name__ == "__main__":
    main()
