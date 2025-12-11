import cv2
import mediapipe as mp
import numpy as np
import json
import os

GESTURE_FILE = "hand_gestures.json"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Normalize landmarks relative to wrist and scale to unit size (distance-invariant)
def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    normalized = [[lm[0] - wrist[0], lm[1] - wrist[1]] for lm in landmarks]
    norm = np.linalg.norm(normalized)
    if norm == 0:
        return normalized
    return (np.array(normalized) / norm).tolist()

# Load saved gestures from file
def load_gestures():
    if os.path.exists(GESTURE_FILE):
        with open(GESTURE_FILE, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}

# Save a gesture to file
def save_gesture(name, landmarks):
    gestures = load_gestures()
    gestures[name] = landmarks
    with open(GESTURE_FILE, "w") as file:
        json.dump(gestures, file)

# Match current landmarks to saved gestures
def match_gesture(current, known_gestures, tolerance=1.0):
    min_distance = float("inf")
    matched_name = "None"
    for name, stored in known_gestures.items():
        if len(stored) != len(current):
            continue
        distance = sum(
            np.linalg.norm(np.array(a) - np.array(b))
            for a, b in zip(current, stored)
        )
        if distance < min_distance and distance < tolerance:
            min_distance = distance
            matched_name = name
    return matched_name

# Webcam capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", 480, 360)
cv2.namedWindow("Gesture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture", 400, 100)

saved_gestures = load_gestures()
tolerance_factor = 0.6

# For stabilizing output
prev_gesture = "None"
stable_gesture = "None"
frames_stable = 0
frames_required = 5

print("Press 'r' to record a new gesture, 'ESC' to quit")

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]  # X and Y only

    if landmark_list:
        saved_gestures = load_gestures()
        normalized = normalize_landmarks(landmark_list)
        current_gesture = match_gesture(normalized, saved_gestures, tolerance_factor)

        if current_gesture == prev_gesture:
            frames_stable += 1
        else:
            frames_stable = 0
        prev_gesture = current_gesture

        if frames_stable >= frames_required:
            stable_gesture = current_gesture
    else:
        stable_gesture = "None"
        frames_stable = 0

    # Display webcam feed
    cv2.imshow("Webcam", frame)

    # Display recognized gesture
    gesture_display = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(gesture_display, f"Gesture: {stable_gesture}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Gesture", gesture_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        if landmark_list:
            name = input("Enter gesture name: ").strip()
            normalized = normalize_landmarks(landmark_list)
            save_gesture(name, normalized)
            print(f"Gesture '{name}' saved.")
        else:
            print("No hand detected. Try again.")
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()



#It’s easy to underestimate Linear Algebra -- until you see it running a real-time gesture recognition system.