"""虚拟隔空触控示例（MediaPipe + OpenCV）。

功能：
1. 打开摄像头实时预览。
2. 在画面上绘制若干“无效”的半透明悬浮按钮。
3. 当用户食指指尖进入按钮区域时，按钮显示 hover 状态。

运行环境：
- Windows 10/11
- Python 3.9+

依赖：
- opencv-python
- mediapipe
- numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FloatingButton:
    """描述一个半透明悬浮按钮。"""

    label: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, point: Tuple[int, int]) -> bool:
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


class VirtualTouchDemo:
    """隔空虚拟触控演示应用。"""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.window_name = "Virtual Touch Demo (MediaPipe)"
        self.preferred_camera_size = (1280, 720)

        # 在预览上放置 4 个无效按钮。
        self.buttons: List[FloatingButton] = [
            FloatingButton("Music", 40, 60, 150, 70),
            FloatingButton("Lights", 220, 60, 150, 70),
            FloatingButton("Fan", 400, 60, 150, 70),
            FloatingButton("Scene", 580, 60, 150, 70),
        ]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def _fit_size(src_w: int, src_h: int, max_w: int, max_h: int) -> Tuple[int, int]:
        """按原始宽高比缩放到目标范围内，避免预览画面被拉伸。"""
        if src_w <= 0 or src_h <= 0:
            return max_w, max_h

        scale = min(max_w / src_w, max_h / src_h)
        scale = max(scale, 0.1)
        return int(src_w * scale), int(src_h * scale)

    def _configure_camera_size(self, cap: cv2.VideoCapture) -> Tuple[int, int]:
        """尝试设置更高分辨率；返回实际生效的分辨率。"""
        target_w, target_h = self.preferred_camera_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return actual_w, actual_h

    def _draw_buttons(
        self, frame: np.ndarray, fingertip: Tuple[int, int] | None
    ) -> np.ndarray:
        """绘制半透明按钮，并根据食指位置显示 hover 状态。"""
        overlay = frame.copy()

        for btn in self.buttons:
            is_hover = fingertip is not None and btn.contains(fingertip)

            # 普通态：淡蓝色；Hover 态：亮绿色。
            if is_hover:
                fill_color = (50, 220, 120)
                border_color = (40, 255, 160)
                text_color = (20, 40, 20)
                alpha = 0.40
            else:
                fill_color = (180, 120, 60)
                border_color = (235, 180, 120)
                text_color = (255, 255, 255)
                alpha = 0.25

            x1, y1 = btn.x, btn.y
            x2, y2 = btn.x + btn.w, btn.y + btn.h

            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, thickness=-1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # 边框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness=2)
            cv2.putText(
                frame,
                btn.label,
                (x1 + 18, y1 + 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                text_color,
                2,
                cv2.LINE_AA,
            )

            if is_hover:
                cv2.putText(
                    frame,
                    "HOVER",
                    (x1 + 18, y1 + btn.h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (10, 70, 10),
                    1,
                    cv2.LINE_AA,
                )

        return frame

    @staticmethod
    def _extract_index_fingertip(
        hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int]:
        """将食指指尖 landmark 转为像素坐标。"""
        h, w, _ = frame_shape
        fingertip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        return int(fingertip.x * w), int(fingertip.y * h)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查设备是否被占用。")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cam_w, cam_h = self._configure_camera_size(cap)
        win_w, win_h = self._fit_size(cam_w, cam_h, max_w=1000, max_h=700)
        cv2.resizeWindow(self.window_name, win_w, win_h)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)

                index_fingertip: Tuple[int, int] | None = None

                if result.multi_hand_landmarks:
                    hand_lms = result.multi_hand_landmarks[0]
                    index_fingertip = self._extract_index_fingertip(hand_lms, frame.shape)

                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

                    cv2.circle(frame, index_fingertip, 10, (0, 255, 255), -1)
                    cv2.putText(
                        frame,
                        f"Index: {index_fingertip[0]}, {index_fingertip[1]}",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (80, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )

                frame = self._draw_buttons(frame, index_fingertip)

                cv2.putText(
                    frame,
                    "Demo only: hover detection without click action | Press Q to quit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Camera: {cam_w}x{cam_h}",
                    (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (210, 210, 210),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


if __name__ == "__main__":
    app = VirtualTouchDemo(camera_index=0)
    app.run()
