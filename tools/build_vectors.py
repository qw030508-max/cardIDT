# ── DB의 카드 이미지 → CLIP 벡터(clip_vectors.npz) 미리 계산 ──
#   python tools/build_vectors.py
#   (tools/build_db.py 를 먼저 실행해야 함)
import sys
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import db


def build_vectors():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CLIP 로딩 중...")
    model, preprocess = clip.load(config.CLIP_MODEL, device=device)
    print(f"CLIP 로드 완료 ({device})")

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT card_id, name_en, name_ko, image_path FROM cards WHERE image_path != ''"
        ).fetchall()
    print(f"총 {len(rows)}장 처리 예정")

    card_ids, name_ens, name_kos, vectors = [], [], [], []
    fail = 0
    for i, (card_id, name_en, name_ko, img_path) in enumerate(rows, 1):
        try:
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                vec  = model.encode_image(img)
                vec /= vec.norm(dim=-1, keepdim=True)
            card_ids.append(card_id)
            name_ens.append(name_en)
            name_kos.append(name_ko or "")
            vectors.append(vec.cpu().numpy())
        except Exception as e:
            print(f"  실패: {card_id} - {e}")
            fail += 1

        if i % 500 == 0:
            print(f"  [{i}/{len(rows)}] 성공: {len(vectors)} / 실패: {fail}")

    vectors_np = np.vstack(vectors)
    np.savez(config.VEC_PATH,
             card_ids=np.array(card_ids),
             name_ens=np.array(name_ens),
             name_kos=np.array(name_kos),
             vectors=vectors_np)
    print(f"\n완료! 성공: {len(vectors)} / 실패: {fail}")
    print(f"벡터 shape: {vectors_np.shape} → {config.VEC_PATH}")


if __name__ == "__main__":
    build_vectors()
