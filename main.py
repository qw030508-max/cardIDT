# ── 유희왕 카드 인식기: 메인 실행 파일 ──
#   python main.py
import sys
from datetime import datetime

import cv2

import config
import db
from recognizer import Recognizer
from ui import put_korean_text, show_debug_window

HELP = "Space:촬영  C:확인창  U:되돌리기  O:오버프레임  I:재고  F:통계  Q:종료"


def print_inventory():
    rows = db.get_inventory()
    print("\n" + "=" * 65)
    print(f"  {'카드명(한글)':<20} {'카드명(영어)':<25} {'수량':>4}  {'사양':<10}")
    print("=" * 65)
    if not rows:
        print("  등록된 카드 없음")
    for name_ko, name_en, count, edition, _ in rows:
        print(f"  {name_ko or name_en:<20} {name_en:<25} {count:>4}장  [{edition}]")
    print("=" * 65 + "\n")


def print_feedback_stats():
    total, correct, top_wrong = db.get_feedback_stats()
    print("\n=== 피드백 통계 (확인창에서 고른 것만) ===")
    if total:
        print(f"  총 평가: {total}건")
        print(f"  정답:    {correct}건")
        print(f"  오답:    {total - correct}건")
        print(f"  정확도:  {correct / total * 100:.1f}%")
    if top_wrong:
        print("\n  자주 틀리는 카드 TOP5:")
        for name, cnt in top_wrong:
            print(f"    {name}: {cnt}회 오답")
    print("=" * 20 + "\n")


def save_snapshots(res):
    """촬영 원본/크롭 저장 (오답 분석, 재학습용)"""
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_path = config.FEEDBACK_DIR / f"snap_{ts}.jpg"
    arts_path = config.FEEDBACK_DIR / f"arts_{ts}.jpg"
    cv2.imwrite(str(snap_path), res["frame"])
    cv2.imwrite(str(arts_path), res["arts"])
    return snap_path, arts_path


def confirm(res):
    """확인창 → 선택한 카드 저장. 반환: 저장한 카드 dict 또는 None"""
    best = res["candidates"][0]
    snap_path, arts_path = save_snapshots(res)

    choice = show_debug_window(res["frame"], res["arts"], res["candidates"],
                               res["outline"], res.get("ocr_name"), res["confident"])
    if choice["action"] in (None, "skip"):
        print("스킵")
        return None

    sel_id, sel_ko, sel_en = choice["selected"]
    is_correct = sel_id == best["card_id"]
    db.save_feedback(
        snap_path, arts_path,
        predicted={k: best[k] for k in ("card_id", "name_ko", "name_en")},
        actual={"card_id": sel_id, "name_ko": sel_ko, "name_en": sel_en},
        confidence=best["score"],
        is_correct=is_correct,
    )
    if is_correct:
        print(f"✅ 정답 → [{sel_ko or sel_en}]")
    else:
        print(f"❌ 오답 수정 → [{best['name_ko'] or best['name_en']}] → [{sel_ko or sel_en}]")
    return {"card_id": sel_id, "name_en": sel_en, "name_ko": sel_ko,
            "score": best["score"], "confident": True}


def scan(recognizer, cap):
    """Space 한 번 처리. 반환: (저장한 카드 dict 또는 None, 인식 결과)"""
    print("\n===== 촬영 시작 =====")
    res = recognizer.capture_best(cap)
    if res is None:
        print("인식 실패 — 카드를 카메라 정면에 맞춰주세요.")
        return None, None

    best = res["candidates"][0]
    if config.AUTO_SAVE and res["confident"]:
        print(f"✔ 확신 → [{best['name_ko'] or best['name_en']}] 자동 저장 "
              f"(매칭점 {best['inliers']}) — 틀렸으면 U 되돌리기 / C 확인창")
        saved = dict(best, confident=True)
    else:
        saved = confirm(res)

    if saved:
        db.add_to_inventory(saved["card_id"], saved["name_en"], saved["name_ko"], saved["score"])
        saved["edition"] = "normal"
    return saved, res


def main():
    config.FEEDBACK_DIR.mkdir(exist_ok=True)
    db.init_db()
    if db.card_count() == 0:
        sys.exit("cards 테이블이 비어 있습니다. 먼저 `python tools/build_db.py`를 실행하세요.")

    recognizer = Recognizer()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("\n=== 유희왕 카드 인식기 (그림 매칭) ===")
    print(f"  Space : 촬영 및 인식 (최대 {config.SHOTS_PER_SCAN}장, 확신하면 바로 멈춤)")
    print("  C     : 마지막 인식 결과를 확인창으로 다시 보기 (자동 저장된 것 수정)")
    print("  U     : 마지막 저장 되돌리기 (재고 -1)")
    print("  O     : 마지막 카드 오버프레임으로 저장")
    print("  I     : 재고 목록 조회")
    print("  F     : 피드백 통계 조회")
    print("  Q     : 종료\n")

    last_saved = None   # 마지막으로 재고에 넣은 카드 (U 되돌리기 / 화면 표시)
    last_res   = None   # 마지막 인식 결과 (C 확인창)

    def undo():
        nonlocal last_saved
        if not last_saved:
            print("되돌릴 저장이 없습니다")
            return
        left = db.remove_from_inventory(last_saved["card_id"], last_saved["edition"])
        print(f"↩ 되돌림 → [{last_saved['name_ko'] or last_saved['name_en']}] 남은 수량 {left}")
        last_saved = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        if last_saved:
            color = (0, 255, 0) if last_saved["confident"] else (0, 100, 255)
            display = put_korean_text(display, last_saved["name_ko"] or last_saved["name_en"],
                                      (10, 10), color, size=36)
            display = put_korean_text(display, f"매칭점 {last_saved.get('inliers', '-')}",
                                      (10, 55), color, size=28)
        display = put_korean_text(display, HELP, (10, display.shape[0] - 35),
                                  (200, 200, 200), size=22)

        cv2.imshow("YuGiOh Card Recognizer", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            saved, res = scan(recognizer, cap)
            last_res   = res or last_res
            last_saved = saved or last_saved
        elif key == ord("c"):
            # 자동 저장된 결과가 틀렸을 때: 되돌리고 확인창에서 다시 고르기
            if not last_res:
                print("먼저 카드를 촬영하세요")
                continue
            auto_saved = last_saved and last_saved.get("card_id") == last_res["candidates"][0]["card_id"]
            if auto_saved:
                undo()
            saved = confirm(last_res)
            if saved:
                db.add_to_inventory(saved["card_id"], saved["name_en"], saved["name_ko"], saved["score"])
                last_saved = dict(saved, edition="normal")
        elif key == ord("u"):
            undo()
        elif key == ord("o"):
            if last_saved:
                db.add_to_inventory(last_saved["card_id"], last_saved["name_en"],
                                    last_saved["name_ko"], last_saved["score"], edition="overframe")
                last_saved = dict(last_saved, edition="overframe")
            else:
                print("먼저 카드를 촬영하세요")
        elif key == ord("i"):
            print_inventory()
        elif key == ord("f"):
            print_feedback_stats()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
