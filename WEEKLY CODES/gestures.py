import math

def _distance(lm_list, p1, p2):
    x1, y1 = lm_list[p1][1], lm_list[p1][2]
    x2, y2 = lm_list[p2][1], lm_list[p2][2]
    return math.hypot(x2 - x1, y2 - y1)

def _hand_scale(lm_list):
    ref = _distance(lm_list, 0, 9)
    return ref if ref != 0 else 1

def _norm_dist(lm_list, p1, p2):
    return _distance(lm_list, p1, p2) / _hand_scale(lm_list)

def fingers_up(lm_list, hand_label="Right"):
    if not lm_list or len(lm_list) < 21:
        return [0, 0, 0, 0, 0]

    fingers = []

    if hand_label == "Right":
        fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)
    else:
        fingers.append(1 if lm_list[4][1] < lm_list[3][1] else 0)

    for tip in [8, 12, 16, 20]:
        pip = tip - 2
        fingers.append(1 if lm_list[tip][2] < lm_list[pip][2] else 0)

    return fingers

def classify(lm_list, hand_label="Right"):
    if not lm_list or len(lm_list) < 21:
        return "NONE"

    fingers = fingers_up(lm_list, hand_label)
    thumb_index_dist = _norm_dist(lm_list, 4, 8)

    if thumb_index_dist < 0.30:
        return "PINCH"

    if thumb_index_dist < 0.40 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
        return "OK"

    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"
    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN_PALM"
    if fingers == [0, 1, 0, 0, 0]:
        return "POINT"
    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE"
    if fingers == [1, 0, 0, 0, 0]:
        return "THUMB"
    if fingers == [0, 0, 0, 0, 1]:
        return "PINKY"
    if fingers == [1, 1, 0, 0, 1]:
        return "ROCK_ON"
    if fingers == [1, 1, 0, 0, 0]:
        return "GUN"

    return "UNKNOWN"
