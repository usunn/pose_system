# input_handler.py
# 각종 영상 정보 소스를 통일된 인터페이스로 뽑아주는 모듈

import cv2
from config import FRAME_WIDTH, FRAME_HEIGHT

class InputHandler:
    def __init__(self, source=0, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        """
        영상 소스 초기화
        :param source: int(웹캠 인덱스) 또는 str(동영상 파일 경로)
        """
        self.cap = cv2.VideoCapture(source)
        # 카메라가 안 열리면 경고
        if not self.cap.isOpened():
            print(f"⚠️ 카메라 열기 실패 (source={source})")
        # 캡처 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
    def is_opened(self):
        return self.cap.isOpened()

    def get_frame(self):
        """
        한 프레임을 읽어서 반환합니다.
        읽기 실패 시 None을 반환합니다.
        :return: BGR 이미지(np.ndarray) 또는 None
        """
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def release(self):
        """
        캡처 리소스를 해제합니다.
        """
        if self.cap.isOpened():
            self.cap.release()
