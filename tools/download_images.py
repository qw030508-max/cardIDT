# ── 카드 이미지 다운로드 (이미 받은 파일은 건너뛰므로 중단 후 다시 실행하면 이어받기) ──
#   python tools/download_images.py          → 일러스트 크롭 이미지 → card_images/
#   python tools/download_images.py --full   → 카드 전체 이미지   → card_images_full/
#   (예전 C_image_download.py + FullCardimgDL.py + resueme_img_dl.py 를 합친 것)
import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def download_images(full=False, delay=0.1):
    url_key  = "image_url" if full else "image_url_cropped"
    save_dir = config.FULL_IMG_DIR if full else config.IMG_DIR
    save_dir.mkdir(exist_ok=True)

    print("전체 카드 목록 불러오는 중...")
    res = requests.get(config.API_URL, timeout=120)
    res.raise_for_status()
    cards = res.json()["data"]

    # 카드 한 장에 일러스트가 여러 개(어나더 일러스트)일 수 있음 → 전부 받기
    todo = [(img["id"], img[url_key])
            for card in cards for img in card.get("card_images", [])
            if not (save_dir / f"{img['id']}.jpg").exists()]
    print(f"카드 {len(cards)}종 | 받을 이미지 {len(todo)}장 → {save_dir}")

    success = fail = 0
    for i, (img_id, url) in enumerate(todo, 1):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                (save_dir / f"{img_id}.jpg").write_bytes(r.content)
                success += 1
            else:
                fail += 1
        except requests.RequestException as e:
            print(f"오류: {img_id} - {e}")
            fail += 1

        if i % 100 == 0:
            print(f"  [{i}/{len(todo)}] 성공: {success} / 실패: {fail}")
        time.sleep(delay)   # API 요청 제한 (초당 약 10회)

    print(f"\n완료! 성공: {success} / 실패: {fail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="크롭 대신 카드 전체 이미지 받기")
    download_images(full=parser.parse_args().full)
