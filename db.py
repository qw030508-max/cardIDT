# ── SQLite 스키마 + 쿼리 모음 ──
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    card_id    TEXT PRIMARY KEY,
    name_en    TEXT,
    name_ko    TEXT,
    card_type  TEXT,
    attribute  TEXT,
    level      TEXT,
    atk        TEXT,
    def_       TEXT,
    card_text  TEXT,
    image_path TEXT
);
CREATE TABLE IF NOT EXISTS inventory (
    inv_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id    TEXT,
    count      INTEGER DEFAULT 1,
    edition    TEXT    DEFAULT 'normal',
    scanned_at TEXT,
    confidence REAL,
    FOREIGN KEY (card_id) REFERENCES cards(card_id)
);
CREATE TABLE IF NOT EXISTS feedback_log (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path   TEXT,
    arts_path    TEXT,
    predicted_id TEXT,
    predicted_ko TEXT,
    predicted_en TEXT,
    actual_id    TEXT,
    actual_ko    TEXT,
    actual_en    TEXT,
    confidence   REAL,
    is_correct   INTEGER,
    timestamp    TEXT
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)


def card_count():
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]


# ══════════════════════════════════════
# 카드 검색
# ══════════════════════════════════════
def find_card_by_name(text):
    """OCR 텍스트로 카드 1장 찾기 (한글 → 영어 순)"""
    if not text or len(text) < 2:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT card_id, name_en, name_ko FROM cards WHERE name_ko LIKE ? LIMIT 1",
            (f"%{text}%",)).fetchone()
        if not row:
            row = conn.execute(
                "SELECT card_id, name_en, name_ko FROM cards WHERE LOWER(name_en) LIKE ? LIMIT 1",
                (f"%{text.lower()}%",)).fetchone()
    return row


def get_card_type(card_id):
    with connect() as conn:
        row = conn.execute("SELECT card_type FROM cards WHERE card_id=?", (card_id,)).fetchone()
    return row[0] if row else None


def search_cards(keyword, limit=10):
    """수동 검색용 (한글/영어 둘 다)"""
    with connect() as conn:
        return conn.execute("""
            SELECT card_id, name_en, name_ko FROM cards
            WHERE name_ko LIKE ? OR LOWER(name_en) LIKE ?
            LIMIT ?
        """, (f"%{keyword}%", f"%{keyword.lower()}%", limit)).fetchall()


# ══════════════════════════════════════
# 재고
# ══════════════════════════════════════
def add_to_inventory(card_id, name_en, name_ko, confidence, edition="normal"):
    with connect() as conn:
        row = conn.execute(
            "SELECT inv_id, count FROM inventory WHERE card_id=? AND edition=?",
            (card_id, edition)).fetchone()
        if row:
            conn.execute("UPDATE inventory SET count=count+1 WHERE inv_id=?", (row[0],))
            print(f"재고 +1 → [{name_ko or name_en}] 현재 {row[1]+1}장")
        else:
            conn.execute("""
                INSERT INTO inventory (card_id, count, edition, scanned_at, confidence)
                VALUES (?, 1, ?, ?, ?)
            """, (card_id, edition, datetime.now().isoformat(), confidence))
            print(f"신규 등록 → [{name_ko or name_en}]")


def remove_from_inventory(card_id, edition="normal"):
    """재고 -1 (0장이 되면 행 삭제). 반환: 남은 수량, 재고에 없으면 None"""
    with connect() as conn:
        row = conn.execute(
            "SELECT inv_id, count FROM inventory WHERE card_id=? AND edition=?",
            (card_id, edition)).fetchone()
        if not row:
            return None
        if row[1] <= 1:
            conn.execute("DELETE FROM inventory WHERE inv_id=?", (row[0],))
            return 0
        conn.execute("UPDATE inventory SET count=count-1 WHERE inv_id=?", (row[0],))
        return row[1] - 1


def get_inventory():
    with connect() as conn:
        return conn.execute("""
            SELECT c.name_ko, c.name_en, i.count, i.edition, i.scanned_at
            FROM inventory i
            JOIN cards c ON i.card_id = c.card_id
            ORDER BY i.scanned_at DESC
        """).fetchall()


# ══════════════════════════════════════
# 피드백
# ══════════════════════════════════════
def save_feedback(img_path, arts_path, predicted, actual, confidence, is_correct):
    with connect() as conn:
        conn.execute("""
            INSERT INTO feedback_log
            (image_path, arts_path, predicted_id, predicted_ko, predicted_en,
             actual_id, actual_ko, actual_en, confidence, is_correct, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            str(img_path), str(arts_path),
            predicted["card_id"], predicted["name_ko"], predicted["name_en"],
            actual["card_id"],    actual["name_ko"],    actual["name_en"],
            confidence, 1 if is_correct else 0,
            datetime.now().isoformat()
        ))


def get_feedback_stats():
    """(총 건수, 정답 수, 자주 틀리는 카드 TOP5)"""
    with connect() as conn:
        total, correct = conn.execute(
            "SELECT COUNT(*), SUM(is_correct) FROM feedback_log").fetchone()
        top_wrong = conn.execute("""
            SELECT predicted_ko, COUNT(*) AS cnt
            FROM feedback_log WHERE is_correct=0
            GROUP BY predicted_ko
            ORDER BY cnt DESC LIMIT 5
        """).fetchall()
    return total, correct or 0, top_wrong
