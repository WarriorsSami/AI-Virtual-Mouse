from enum import IntEnum

import cv2
import mediapipe as mp
import numpy as np

import time
import math

FUXIA = (255, 0, 255)
LIGHT_GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

wImg, hImg = 1000, 800
fpsX, fpsY = 10, 70


class Fingers(IntEnum):
    THUMB = 4,
    INDEX = 8,
    MIDDLE = 12,
    RING = 16,
    PINKY = 20


class HandDetector(object):
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         self.detection_conf, self.track_conf)

        self.mp_draw = mp.solutions.drawing_utils
        # thumb - index - middle - ring - pinky
        self.tip_ids = [4, 8, 12, 16, 20]

        self.results = None
        self.lm_list = []

    def find_hands(self, image, draw=True):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return image

    def fingers_up(self):
        fingers = []

        # Thumb
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for uid in range(1, 5):
            if self.lm_list[self.tip_ids[uid]][2] < self.lm_list[self.tip_ids[uid] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, image, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(image, (x1, y1), (x2, y2), FUXIA, t)
            cv2.circle(image, (x1, y1), r, FUXIA, cv2.FILLED)
            cv2.circle(image, (x2, y2), r, FUXIA, cv2.FILLED)
            cv2.circle(image, (cx, cy), r, BLUE, cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, image, [x1, y1, x2, y2, cx, cy]

    def find_position(self, image, hand_no=0, draw=True):
        x_list = []
        y_list = []
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for (uid, lm) in enumerate(my_hand.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([uid, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 5, FUXIA, cv2.FILLED)

        x_min, x_max = min(x_list, default=0), max(x_list, default=0)
        y_min, y_max = min(y_list, default=0), max(y_list, default=0)
        bbox = x_min, y_min, x_max, y_max

        if draw:
            cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max - 20, y_max - 20), LIGHT_GREEN, 2)

        return self.lm_list, bbox


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, image = cap.read()
        image = cv2.resize(image, (wImg, hImg))

        image = detector.find_hands(image)
        lm_list, bbox = detector.find_position(image)
        if len(lm_list):
            print(detector.fingers_up())

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, str(int(fps)), (fpsX, fpsY), cv2.FONT_HERSHEY_PLAIN, 3, FUXIA, 3)
        cv2.imshow("Image", image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
