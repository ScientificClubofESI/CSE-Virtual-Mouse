import cv2
import numpy as np
import hand_detector_module as hdm
import time
import autopy

# global variables #
cam_width, cam_height = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
clickable_dist = 22

prev_x_coords, prev_y_coords = 0, 0
current_x_coords, current_y_coords = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = hdm.hand_detector()
screen_width, screen_height = autopy.screen.size()

# loop until exit button
while True:
    # 1) find the landmarks
    success, img = cap.read()
    img = detector.detect_hand(img)
    landmarks = detector.find_fingers_positions(img)

    # 2) get the tip of the index and the small finger
    if len(landmarks) != 0:
        x1, y1 = landmarks[8][1:]
        x2, y2 = landmarks[20][1:]

        # 3) get fingers state (up or down)
        fingers = detector.fingers_up()
        cv2.rectangle(img, (frameR, frameR), (cam_width - frameR, cam_height - frameR), (255, 0, 255), 2)

        # 4) execution based on fingers state

        # 4-1) moving mode (only index finger is up)
        if(fingers[1] == 1 and fingers[2] == 0):
            # convert coordinates
            # mapping from image coords to screen coords
            x3 = np.interp(x1, (frameR, cam_width-frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, cam_height - frameR), (0, screen_height))

            # smooting the values
            current_x_coords = prev_x_coords + (x3 - prev_x_coords) / smoothening
            current_y_coords = prev_y_coords + (y3 - prev_y_coords) / smoothening

            # moving the mouse
            autopy.mouse.move(screen_width - current_x_coords, current_y_coords)
            cv2.circle(img, (x1,y1), 15, (255, 0, 255), cv2.FILLED)
            prev_x_coords, prev_y_coords = current_x_coords, current_y_coords

        # 4-2) clicking mode (index up and middle finger up and distance between them < clickable_dist)
        if(fingers[1] == 1 and fingers[2] == 1):
            length, img, _ = detector.find_distance(8, 12, img)
            #print(length)
            if(length <= clickable_dist):
                autopy.mouse.click()
        # 4-3) scrolling down (thumb up)
        if(fingers[0] == 1): 
            autopy.key.toggle(autopy.key.Code.DOWN_ARROW, down=True)
        else:
            autopy.key.toggle(autopy.key.Code.DOWN_ARROW, down=False)   

        # 4-4) scrolling up (small finger up) 
        if(fingers[4] == 1): 
            autopy.key.toggle(autopy.key.Code.UP_ARROW, down=True)
        else:
            autopy.key.toggle(autopy.key.Code.UP_ARROW, down=False)            

    img = cv2.flip(img, 1)
    cv2.imshow('Tracking', img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()