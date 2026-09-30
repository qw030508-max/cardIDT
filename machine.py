# ── 카드 분류기 (기계 모드, 화면 없이 동작) ──
#   python machine.py            → 엔터 칠 때마다 1장 인식 (1단계: 촬영부 테스트, 카드는 손으로 놓기)
#   python machine.py --no-save  → 재고 DB에 저장하지 않고 인식/분류 결과만 보기
#
# 나중에 급지부/분류부가 생기면 wait_for_card() / send_to_bin() 만 모터·센서 코드로 바꾸면 됨
import argparse
import sys
import time
from collections import Counter
from datetime import datetime

import cv2

import config
import db
import sorter
from camera import open_camera
from recognizer import Recognizer


# ══════════════════════════════════════
# 하드웨어 자리 (지금은 키보드/출력으로 대신함)
# ══════════════════════════════════════
def wait_for_card():
    """카드가 촬영 위치에 올 때까지 대기. 반환: False면 종료"""
    cmd = input("\n카드를 놓고 Enter (q: 종료) > ").strip().lower()
    return cmd != "q"


def send_to_bin(bin_no):
    """분류부 모터 제어 자리"""
    print(f"  → {sorter.bin_label(bin_no)}")


# ══════════════════════════════════════
# 1장 처리
# ══════════════════════════════════════
def process_one(recognizer, cap, save):
    t0 = time.time()
    res = recognizer.capture_best(cap)
    elapsed = time.time() - t0

    if res is None:
        print(f"  카드 못 찾음 ({elapsed:.1f}s)")
        send_to_bin(sorter.REJECT_BIN)
        return sorter.REJECT_BIN, None

    top = res["candidates"][0]
    card_type = db.get_card_type(top["card_id"])
    bin_no, cat = sorter.choose_bin(card_type, res["confident"])

    name = top["name_ko"] or top["name_en"]
    mark = "✔" if res["confident"] else "? 확신 없음"
    print(f"  {name} [{sorter.CATEGORY_NAMES[cat]}] 매칭점 {top['inliers']} {mark} ({elapsed:.1f}s)")

    if not res["confident"]:
        # 확신 없는 카드는 사진을 남겨서 나중에 확인/개선
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(config.FEEDBACK_DIR / f"machine_{ts}.jpg"), res["frame"])
    elif save:
        db.add_to_inventory(top["card_id"], top["name_en"], top["name_ko"], top["score"])

    send_to_bin(bin_no)
    return bin_no, top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true", help="재고 DB에 저장하지 않기")
    args = parser.parse_args()

    sorter.validate_filters()
    config.FEEDBACK_DIR.mkdir(exist_ok=True)
    db.init_db()
    if db.card_count() == 0:
        sys.exit("cards 테이블이 비어 있습니다. 먼저 `python tools/build_db.py`를 실행하세요.")

    recognizer = Recognizer()
    cap = open_camera()

    print("\n=== 카드 분류기 (기계 모드) ===")
    for b in range(4):
        print(f"  {sorter.bin_label(b)}")

    counts, times = Counter(), []
    try:
        while wait_for_card():
            t0 = time.time()
            bin_no, _ = process_one(recognizer, cap, save=not args.no_save)
            times.append(time.time() - t0)
            counts[bin_no] += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    if times:
        print(f"\n=== 결과: {len(times)}장, 평균 {sum(times)/len(times):.1f}s/장 ===")
        for b in range(4):
            print(f"  {sorter.bin_label(b)}: {counts[b]}장")


if __name__ == "__main__":
    main()
