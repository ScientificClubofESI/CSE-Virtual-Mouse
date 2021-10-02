import cv2
import mediapipe as mp
import math

class hand_detector():
    def __init__(self, mode=False, maxHands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5, ):
        self.mode = mode
        self.maxHands = maxHands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        #importing the models from mediapipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # intializing the detection model
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.min_detection_confidence, self.min_tracking_confidence)

        # ids of the tips of each finger
        self.tip_ids = [4, 8, 12, 16, 20]

    def detect_hand(self, img):
            # the frame from opencv has coloration of type BGR
            # mediapipe works with RGB so we have to convert
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #detect the hand (points d'articulations)
            self.results = self.hands.process(imgRGB)
            # print(results.multi_hand_landmarks) => a dictionary
            # that contains each point d'articulation with its id, x_coordinate and y_coordinate

            #drawing the points
            if self.results.multi_hand_landmarks:
                for landmark in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, landmark, self.mp_hands.HAND_CONNECTIONS)

            return img

    def find_fingers_positions(self, img, draw=True):
        x_coords = []
        y_coords = []
        #bbox = []
        self.landmarks = []

        # results.multi_hand_landmarks is a list of 
        # the positions of all the 20 points d'articulation (id, x, y, z)
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            for id, landmark in enumerate(hand.landmark):
                h, w, c = img.shape
                # get the real coordinates of each landmark (multiply by image height and width)
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_coords.append(x)
                y_coords.append(y)

                self.landmarks.append([id, x, y])
                if draw:
                    cv2.circle(img, (x, y), 5, (255, 0, 255), cv2.FILLED)

        return self.landmarks

    def fingers_up(self):
        fingers = []

        # detect if thumb is up
        # if x_coords of the 4th articulation point is bigger than the x_coords of the 3rd point (see hand diagram)
        if self.landmarks[self.tip_ids[0]][1] > self.landmarks[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else :
            fingers.append(0)

        # other fingers
        # if the y_coords of the tip is less than the y_coords of the (id-2)th articulation point
        for id in range(1,5):
            if self.landmarks[self.tip_ids[id]][2] < self.landmarks[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, id1, id2, img, draw=True, radius=15, thic=3):
        x1, y1 = self.landmarks[id1][1:] # x_coords and y_coords of the 1st articulation
        x2, y2 = self.landmarks[id2][1:] # x_coords and y_coords of the 2nd articulation
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), thic)
            cv2.circle(img, (x1, y1), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), radius, (0, 0, 255), cv2.FILLED)

        distance = math.hypot(x2 - x1, y2 - y1) # distance between 2 points in space

        return distance, img, [x1, y1, x2, y2, cx, cy]