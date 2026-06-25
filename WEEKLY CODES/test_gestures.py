import cv2
import mediapipe as mp
import time
import gestures

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def build_landmark_list(frame, hand_landmarks):
    h, w, _ = frame.shape
    return [[i, int(lm.x * w), int(lm.y * h)] for i, lm in enumerate(hand_landmarks.landmark)]

cap = cv2.VideoCapture(0)
prev_time = 0
gesture_counts = {}
last_gesture = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "NONE"
    lm_list = []
    hand_label = "Right"
    fingers = [0, 0, 0, 0, 0]

    if results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]
        if results.multi_handedness:
            hand_label = results.multi_handedness[0].classification[0].label

        lm_list = build_landmark_list(frame, hand_lm)
        gesture = gestures.classify(lm_list, hand_label)
        fingers = gestures.fingers_up(lm_list, hand_label)

        gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        if gesture != last_gesture:
            print(f"-> {gesture} ({hand_label})")
            last_gesture = gesture

        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        for tip in [4, 8, 12, 16, 20]:
            _, cx, cy = lm_list[tip]
            cv2.circle(frame, (cx, cy), 11, (220, 0, 220), cv2.FILLED)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    color = (0, 255, 0) if gesture not in ("NONE", "UNKNOWN") else (80, 80, 80)
    cv2.putText(frame, gesture, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
    cv2.putText(frame, f"FPS: {int(fps)}", (width - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 0), 2)
    cv2.putText(frame, f"Hand: {hand_label}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

    finger_names = ["T", "I", "M", "R", "P"]
    for i, (name, state) in enumerate(zip(finger_names, fingers)):
        color = (0, 255, 0) if state else (0, 0, 180)
        cv2.putText(frame, name, (20 + i * 35, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, "Week 4 - Gesture Module Test", (20, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("Week 4 - Gesture Module Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nGesture Statistics:")
for g, count in sorted(gesture_counts.items(), key=lambda x: -x[1]):
    print(f"  {g}: {count} frames")
