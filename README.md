# cardIDT — 유희왕 카드 인식 + 재고 관리

카메라로 카드를 찍으면 **일러스트 그림**으로 카드를 판별해서 `yugioh.db` 재고에 저장합니다.
(언어와 상관없이 동작하도록 OCR은 쓰지 않는 게 기본)

1. 카드 테두리 찾기 → 반듯하게 펴기 → 고정 비율로 일러스트 크롭 (못 찾으면 YOLO Arts 크롭)
2. CLIP 벡터 검색으로 후보 20장
3. SIFT 특징점 매칭으로 재정렬 — 같은 그림이면 매칭점이 수십~수백 개, 다른 그림은 한 자릿수
4. 확신(매칭점 20↑, 2위의 2.5배↑)하면 자동 저장, 애매하면 확인창

## 구조

```
machine.py       기계 모드 (화면 없이, 분류 통 결정)
sorter.py        분류 규칙 (몬스터/마법/함정/그 외 → 통 번호)
camera.py        카메라 열기 (Picamera2 / OpenCV)
main.py          카메라 루프 (Space 촬영 / C 확인창 / U 되돌리기 / O 오버프레임 / I 재고 / F 통계 / Q 종료)
recognizer.py    인식 파이프라인 (크롭 → CLIP → 특징점 재정렬, 연속 촬영)
card_warp.py     카드 테두리 찾기, 원근 보정, 일러스트 위치 비율
feature_match.py SIFT 특징점 매칭
ui.py            한글 텍스트 출력, tkinter 확인창
db.py            SQLite 스키마(cards / inventory / feedback_log)와 쿼리
config.py        경로, 카메라 번호, 임계값 등 설정

tools/
  build_db.py        API → yugioh.db (카드 정보 + 한글명)
  download_images.py 카드 이미지 다운로드 (--full: 전체 이미지)
  build_vectors.py   card_images → clip_vectors.npz

best.pt            학습된 YOLOv5 모델 (클래스: Arts, Cname …)
card_images/       카드 일러스트 크롭 이미지 ({card_id}.jpg)
clip_vectors.npz   카드별 CLIP 벡터
feedback_images/   촬영 원본/크롭 (오답 분석, 재학습용)
experiments/       인식 방식 비교 실험 (pipeline_test.py: 저장된 사진으로 전체 파이프라인 테스트)
_archive/          예전 버전 코드 (사용 안 함)
```

## 처음 세팅 순서

```
pip install -r requirements.txt
python tools/download_images.py   # 이미지 (이미 있으면 건너뜀)
python tools/build_db.py          # yugioh.db 생성
python tools/build_vectors.py     # 벡터 (이미지/카드가 바뀌었을 때만)
python main.py
```

카메라 번호는 `config.py`의 `CAMERA_INDEX` (DroidCam 1, 웹캠 0).

## 기계 모드 (카드 분류기)

```
python machine.py            # 카드를 놓고 Enter → 인식 → 몇 번 통인지 출력
python machine.py --no-save  # 재고에 저장하지 않고 결과만 보기
```

- 통 0~2: `config.BIN_FILTERS` (기본 몬스터 / 마법 / 함정), 통 3: 나머지 + 확신 없음
- 위아래가 뒤집혀 들어온 카드도 180° 돌려서 다시 확인
- 확신 없는 카드 사진은 `feedback_images/machine_*.jpg` 로 저장
- 급지/분류 모터가 생기면 `machine.py`의 `wait_for_card()` / `send_to_bin()` 만 바꾸면 됨

### 라즈베리 파이 5 설치

```
sudo apt install -y python3-picamera2 git
git clone https://github.com/qw030508-max/cardIDT.git
cd cardIDT
python3 -m venv --system-site-packages .venv   # apt의 picamera2를 venv에서 쓰기 위해
source .venv/bin/activate
pip install -r requirements-pi.txt
```

데이터 파일은 저장소에 없으니 PC에서 복사하거나(USB 등) 파이에서 다시 만들기:
`yugioh.db`, `clip_vectors.npz`, `card_images/` (약 2GB, 특징점 매칭에 필요)

`config.py`에서 `CAMERA_BACKEND = "picamera2"`, `USE_YOLO = False` 로 바꾸고 `python machine.py`.
