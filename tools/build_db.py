# ── YGOPRODeck API → yugioh.db (cards 테이블) 구축 ──
#   python tools/build_db.py
#   (예전 createDB.py + add_korean_names.py + setup.py 를 합친 것. JSON 중간 파일 없이 바로 DB로 저장)
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import db


def fetch(language=None):
    params = {"language": language} if language else {}
    res = requests.get(config.API_URL, params=params, timeout=120)
    res.raise_for_status()
    return res.json()["data"]


def text(v):
    return "" if v is None else str(v)


def build_db():
    print("영문 카드 데이터 다운로드 중...")
    cards = fetch()
    print(f"  {len(cards)}장")

    print("한글 카드 데이터 다운로드 중...")
    ko_names = {c["id"]: c.get("name", "") for c in fetch("ko")}
    print(f"  한글명 {len(ko_names)}장")

    db.init_db()
    rows, with_img = [], 0
    for c in cards:
        # 카드 ID = 첫 번째 일러스트 ID (card_images/{id}.jpg 파일명과 동일)
        card_id  = str(c["card_images"][0]["id"])
        img_path = config.IMG_DIR / f"{card_id}.jpg"
        has_img  = img_path.exists()
        with_img += has_img
        rows.append((
            card_id,
            c.get("name", ""),
            ko_names.get(c["id"], ""),
            c.get("type", ""),
            c.get("attribute", ""),
            text(c.get("level")),
            text(c.get("atk")),
            text(c.get("def")),
            c.get("desc", ""),
            str(img_path) if has_img else "",
        ))

    with db.connect() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO cards
            (card_id, name_en, name_ko, card_type, attribute, level, atk, def_, card_text, image_path)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, rows)

    print(f"\n✅ 완료! 카드 {len(rows)}장 저장 "
          f"(한글명 {sum(1 for r in rows if r[2])}장 / 이미지 있음 {with_img}장)")
    print(f"  → {config.DB_PATH}")


if __name__ == "__main__":
    build_db()
