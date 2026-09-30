# ── 카메라 열기: 라즈베리 파이 카메라(Picamera2) 또는 일반 웹캠(OpenCV) ──
# 둘 다 cap.read() → (ok, BGR 프레임) 형태로 똑같이 쓸 수 있게 맞춤
import cv2

import config


class PiCamera:
    def __init__(self, width, height):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        # Picamera2의 "RGB888"은 실제 메모리 순서가 BGR이라 OpenCV에 그대로 쓰면 됨
        self.cam.configure(self.cam.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}))
        self.cam.start()

    def read(self):
        return True, self.cam.capture_array()

    def release(self):
        self.cam.stop()


def open_camera():
    if config.CAMERA_BACKEND == "picamera2":
        print("Picamera2 카메라 사용")
        return PiCamera(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    print(f"OpenCV 카메라 사용 (번호 {config.CAMERA_INDEX})")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap
