# input_handler.py
# 각종 영상 정보 소스를 통일된 인터페이스로 뽑아주는 모듈

import cv2
from config import FRAME_WIDTH, FRAME_HEIGHT

# Picamera2 지원 시도
try:
    from picamera2 import Picamera2
    HAVE_PICAMERA2 = True
except ImportError:
    HAVE_PICAMERA2 = False

class InputHandler:
    def __init__(self, source=0, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        """
        영상 소스 초기화
        :param source: int(웹캠 인덱스), str(동영상 파일 경로), 또는 "picam2"
        """
        self.use_picam2 = False
        # Picamera2 분기
        if HAVE_PICAMERA2 and source == "picam2":
            print("[InputHandler] Picamera2 모드 진입")
            self.picam2 = Picamera2()
            cfg = self.picam2.create_preview_configuration({
                "size": (width, height),
                "format": "XRGB8888"
            })
            self.picam2.configure(cfg)
            self.picam2.start()
            self.use_picam2 = True
        else:
            # OpenCV VideoCapture
            print(f"[InputHandler] VideoCapture 모드 (source={source})")
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print(f"⚠️ 카메라 열기 실패 (source={source})")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_opened(self):
        if self.use_picam2:
            return True
        return self.cap.isOpened()

    def get_frame(self):
        """
        한 프레임을 읽어서 반환합니다.
        읽기 실패 시 None을 반환합니다.
        :return: BGR 이미지(np.ndarray) 또는 None
        """
        if self.use_picam2:
            img = self.picam2.capture_array()
            # RGB->BGR 변환
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        success, frame = self.cap.read()
        return frame if success else None

    def release(self):
        """
        캡처 리소스를 해제합니다.
        """
        if self.use_picam2:
            self.picam2.stop()
        else:
            if self.cap.isOpened():
                self.cap.release()
