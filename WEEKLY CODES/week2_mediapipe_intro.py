import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    status = "Hand Detected" if hand_detected else "No Hand Found"
    color = (0, 255, 0) if hand_detected else (0, 0, 255)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, "Week 2 - MediaPipe Intro", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Week 2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
