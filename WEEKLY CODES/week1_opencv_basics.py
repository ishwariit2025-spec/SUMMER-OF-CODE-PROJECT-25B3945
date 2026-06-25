import cv2
import time

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 255, 0), 2)
    cv2.circle(frame, (width // 2, height // 2), 50, (255, 0, 0), 2)
    cv2.line(frame, (10, 10), (width // 2, height // 2), (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Week 1 - OpenCV Basics", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Week 1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
