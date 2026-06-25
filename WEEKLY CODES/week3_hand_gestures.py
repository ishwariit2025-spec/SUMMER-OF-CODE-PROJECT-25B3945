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

def get_landmark_list(frame, hand_landmarks):
    h, w, _ = frame.shape
    lm_list = []
    for idx, lm in enumerate(hand_landmarks.landmark):
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        lm_list.append([idx, cx, cy])
    return lm_list

def fingers_up(lm_list):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)
    for tip in tips[1:]:
        fingers.append(1 if lm_list[tip][2] < lm_list[tip - 2][2] else 0)
    return fingers

def classify_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"
    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"
    if fingers == [0, 1, 0, 0, 0]:
        return "POINTING"
    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE"
    if fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP"
    if fingers == [0, 0, 0, 0, 1]:
        return "PINKY UP"
    if fingers == [1, 1, 0, 0, 1]:
        return "ROCK ON"
    return "UNKNOWN"

cap = cv2.VideoCapture(0)
prev_time = 0
last_gesture = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "NO HAND"
    lm_list = []

    if results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        lm_list = get_landmark_list(frame, hand_lm)

        if lm_list:
            for tip_idx in [4, 8, 12, 16, 20]:
                _, cx, cy = lm_list[tip_idx]
                cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            fingers = fingers_up(lm_list)
            gesture = classify_gesture(fingers)

            if gesture != last_gesture:
                print(f"Fingers: {fingers}  ->  {gesture}")
                last_gesture = gesture

            labels = ["T", "I", "M", "R", "P"]
            for i, (label, state) in enumerate(zip(labels, fingers)):
                color = (0, 255, 0) if state else (0, 0, 255)
                cv2.putText(frame, label, (20 + i * 35, height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    color = (0, 255, 0) if gesture not in ["NO HAND", "UNKNOWN"] else (0, 0, 255)
    cv2.putText(frame, gesture, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"FPS: {int(fps)}", (width - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "Week 3 - Gesture Recognition", (20, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Week 3 - Hand Gestures", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
