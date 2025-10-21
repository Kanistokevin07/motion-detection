import cv2
import time
import datetime
import os
import numpy as np

BASE_SAVE_DIR = "motion_snapshots"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

session_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SESSION_DIR = os.path.join(BASE_SAVE_DIR, session_time)
os.makedirs(SESSION_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
time.sleep(2)

ret, frame1 = cap.read()
ret, frame2 = cap.read()
motion_counter = 0

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1200:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame1, timestamp, (10, frame1.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    display_frame = frame1.copy()
    if motion_detected:
        overlay = display_frame.copy()
        alpha = 0.4
        cv2.rectangle(overlay, (0,0), (display_frame.shape[1], display_frame.shape[0]), (0,0,255), -1)
        display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
        cv2.putText(display_frame, "MOTION DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        motion_counter += 1
        filename = os.path.join(SESSION_DIR, f"motion_{motion_counter}_{int(time.time())}.jpg")
        cv2.imwrite(filename, frame1)
        print(f"[INFO] Motion detected! Saved snapshot: {filename}")

    cv2.imshow("Smart Motion Detection", display_frame)
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
