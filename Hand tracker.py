import cv2
import numpy as np
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, landmarks in enumerate(hand_landmarks.landmark):
                #print(id,landmarks)
                h,w,c = img.shape
                cx, cy = int(landmarks.x*w), int(landmarks.y*h)
                print(id, cx, cy)
                cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, hand_landmarks,mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)        
    cv2.imshow("Result",img)
    cv2.waitKey(1) 