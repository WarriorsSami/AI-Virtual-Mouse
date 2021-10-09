import time

import autopy
import cv2
import numpy as np

from HandTrackingModule import FUXIA, LIGHT_GREEN
from HandTrackingModule import HandDetector, Fingers
from HandTrackingModule import fpsX, fpsY

wCam, hCam = 600, 420
frameR = 100  # Frame Reduction
wScr, hScr = autopy.screen.size()
THRESHOLD = 50
SMOOTHEN = 6.9

pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

# print(wScr, hScr)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HandDetector(max_hands=1)

while True:
    #  1. Find hand landmarks
    success, image = cap.read()
    # image = cv2.resize(image, (wImg, hImg))

    image = detector.find_hands(image)
    lm_list, bbox = detector.find_position(image)

    #  2. Find the tips of index and middle fingers
    if len(lm_list):
        x1, y1 = lm_list[Fingers.INDEX][1:]
        x2, y2 = lm_list[Fingers.MIDDLE][1:]
        # print(x1, y1, x2, y2)

        #  3. Check which fingers are up
        fingers = detector.fingers_up()
        # print(fingers)
        if 1 not in fingers:
            break
        cv2.rectangle(image, (frameR, frameR), (wCam - frameR, hCam - frameR), FUXIA, 2)

        #  4. Only index finger is up => Moving Mode
        if fingers[1] and fingers[2] == 0:
            #  5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            #  6. Smoothen values
            cLocX = pLocX + (x3 - pLocX) / SMOOTHEN
            cLocY = pLocY + (y3 - pLocY) / SMOOTHEN

            #  7. Move mouse
            autopy.mouse.move(wScr - cLocX, cLocY)
            cv2.circle(image, (x1, y1), 15, FUXIA, cv2.FILLED)

            pLocX, pLocY = cLocX, cLocY

        #  8. Both index and middle fingers are up => Clicking Mode
        if fingers[1] and fingers[2]:
            #  9. Find distance between fingers
            length, image, info_line = detector.find_distance(Fingers.INDEX, Fingers.MIDDLE, image)
            # print(length)

            # 10. Enable click event if distance is less than a certain value
            if length < THRESHOLD:
                cv2.circle(image, (info_line[4], info_line[5]), 15, LIGHT_GREEN, cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, str(int(fps)), (fpsX, fpsY), cv2.FONT_HERSHEY_PLAIN, 3, FUXIA, 3)

    # 12. Display
    cv2.imshow("Image", image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
