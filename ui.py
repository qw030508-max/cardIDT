# ── 화면 출력: OpenCV 한글 텍스트 + tkinter 확인창 ──
import tkinter as tk
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import config
import db

BG, PANEL = "#1e1e2e", "#2a2a3e"
FONT = "맑은 고딕"


@lru_cache(maxsize=None)
def _font(size):
    try:
        return ImageFont.truetype(config.FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def put_korean_text(frame, text, pos, color, size=36):
    """cv2.putText는 한글이 깨져서 PIL로 그림 (color는 BGR)"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil_img).text(pos, text, font=_font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def candidate_label(c):
    return f"{c['name_ko'] or c['name_en']}  (매칭점 {c['inliers']} · CLIP {c['score']*100:.1f}%)"


def show_debug_window(frame, arts_bgr, candidates, outline, ocr_text, confident):
    """
    인식 결과 확인창.
    반환: {"action": "correct"|"wrong"|"skip"|None,
           "selected": (card_id, name_ko, name_en) 또는 None}
    """
    result = {"action": None, "selected": None}
    best   = candidates[0]
    db_img_path = config.IMG_DIR / f"{best['card_id']}.jpg"

    frame_vis = frame.copy()
    if outline is not None:
        cv2.polylines(frame_vis, [outline.astype(np.int32)], True, (0, 255, 0), 3)
    arts_crop = Image.fromarray(cv2.cvtColor(arts_bgr, cv2.COLOR_BGR2RGB)) if arts_bgr is not None else None

    root = tk.Tk()
    root.title("🔍 카드 인식 디버그 확인창")
    root.configure(bg=BG)
    root.resizable(False, False)

    # ── 이미지 3개 나란히 ──
    img_frame = tk.Frame(root, bg=BG)
    img_frame.pack(padx=16, pady=12)

    def add_image(pil_img, w, h, title):
        box = tk.Frame(img_frame, bg=BG)
        box.pack(side=tk.LEFT, padx=8)
        tk.Label(box, text=title, bg=BG, fg="#7dd3fc",
                 font=("Arial", 10, "bold")).pack()
        photo = ImageTk.PhotoImage(pil_img.resize((w, h)))
        lbl   = tk.Label(box, image=photo, bg=BG, relief="solid", bd=1)
        lbl.image = photo
        lbl.pack()

    add_image(Image.fromarray(cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)),
              280, 210, "📷 촬영 원본 (카드 영역 표시)")
    if arts_crop:
        add_image(arts_crop, 160, 210, "✂️ Arts 크롭")
    if db_img_path.exists():
        add_image(Image.open(db_img_path), 160, 210, "🗃️ DB 매칭 이미지")

    # ── 인식 결과 ──
    info = tk.Frame(root, bg=PANEL)
    info.pack(fill=tk.X, padx=16, pady=4)
    tk.Label(info, text=f"인식된 카드: {best['name_ko'] or best['name_en']}",
             bg=PANEL, fg="#a5f3fc", font=(FONT, 13, "bold")).pack(anchor="w", padx=12, pady=4)
    status = "✔ 확신" if confident else "⚠ 확신 없음 — 후보를 확인하세요"
    ocr    = f"   |   OCR: {ocr_text or '없음'}" if config.USE_OCR else ""
    tk.Label(info, text=f"{status}   |   매칭점 {best['inliers']} · CLIP {best['score']*100:.1f}%{ocr}",
             bg=PANEL, fg="#94a3b8", font=(FONT, 10)).pack(anchor="w", padx=12, pady=2)

    # ── 후보 목록 ── (라디오 값은 "id|ko|en" 문자열)
    cand_frame = tk.Frame(root, bg=BG)
    cand_frame.pack(fill=tk.X, padx=16, pady=6)
    tk.Label(cand_frame, text="후보 카드 목록:", bg=BG, fg="#fbbf24",
             font=(FONT, 10, "bold")).pack(anchor="w")

    def encode(card_id, name_ko, name_en):
        return f"{card_id}|{name_ko}|{name_en}"

    selected = tk.StringVar(value=encode(best["card_id"], best["name_ko"], best["name_en"]))

    for i, cand in enumerate(candidates[:config.TOP_K]):
        tk.Radiobutton(
            cand_frame,
            text=f"{'★' if i == 0 else ' '} {candidate_label(cand)}",
            variable=selected, value=encode(cand["card_id"], cand["name_ko"], cand["name_en"]),
            bg=BG, fg="#a5f3fc" if i == 0 else "#cbd5e1", selectcolor=PANEL,
            font=(FONT, 10), activebackground=BG,
        ).pack(anchor="w", padx=8)

    # ── 직접 검색 ──
    search_frame = tk.Frame(root, bg=BG)
    search_frame.pack(fill=tk.X, padx=16, pady=4)
    tk.Label(search_frame, text="직접 검색:", bg=BG, fg="#fbbf24",
             font=(FONT, 10, "bold")).pack(side=tk.LEFT)
    search_var = tk.StringVar()
    tk.Entry(search_frame, textvariable=search_var, width=24, font=(FONT, 10),
             bg=PANEL, fg="white", insertbackground="white").pack(side=tk.LEFT, padx=6)

    listbox = tk.Listbox(root, height=4, width=48, bg=PANEL, fg="white",
                         font=(FONT, 10), selectbackground="#3b82f6")
    listbox.pack(padx=16, pady=2)
    search_results = []

    def do_search(*_):
        keyword = search_var.get().strip()
        if len(keyword) < 2:
            return
        listbox.delete(0, tk.END)
        search_results.clear()
        for card_id, name_en, name_ko in db.search_cards(keyword):
            listbox.insert(tk.END, f"{name_ko or name_en}  ({name_en})")
            search_results.append((card_id, name_en, name_ko))

    def on_select(_):
        sel = listbox.curselection()
        if sel:
            card_id, name_en, name_ko = search_results[sel[0]]
            selected.set(encode(card_id, name_ko, name_en))

    search_var.trace_add("write", do_search)
    listbox.bind("<<ListboxSelect>>", on_select)

    # ── 버튼 ──
    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(pady=12)

    def finish(action):
        result["action"] = action
        if action != "skip":
            card_id, name_ko, name_en = (selected.get().split("|") + ["", ""])[:3]
            result["selected"] = (card_id, name_ko, name_en)
        root.destroy()

    for text, action, color, width in [
        ("✅  정답 — DB 저장",       "correct", "#22c55e", 18),
        ("❌  오답 — 수정 후 저장",  "wrong",   "#ef4444", 18),
        ("⏭  스킵",                 "skip",    "#6b7280", 10),
    ]:
        tk.Button(btn_frame, text=text, command=lambda a=action: finish(a),
                  width=width, height=2, bg=color, fg="white",
                  font=(FONT, 11, "bold"), relief="flat", cursor="hand2"
                  ).pack(side=tk.LEFT, padx=6)

    root.mainloop()
    return result
