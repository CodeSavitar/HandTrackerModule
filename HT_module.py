import cv2
import numpy as np
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

        return img   

    def findPosition(self, img, handno=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handno]

            for id, landmarks in enumerate(myHand.landmark):
                #print(id,landmarks)
                h,w,c = img.shape
                cx, cy = int(landmarks.x*w), int(landmarks.y*h)
                #print(id, cx, cy)
                landmark_list.append([id, cx, cy])
                cv2.circle(img, (cx,cy), 8, (0,0,204), cv2.FILLED)

        return landmark_list        

def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:

        success,img = cap.read()
        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)
        if len(landmark_list) != 0:
            print(landmark_list[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)        
        cv2.imshow("Result",img)
        cv2.waitKey(1) 


if __name__ == "__main__":
    main()