import cv2
import mediapipe as mp
from collections import deque
import time

class HandTracker:
    def __init__(self):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Incompatible MediaPipe build detected. "
                "This app requires the legacy Solutions API. "
                "Run: pip install mediapipe==0.10.14"
            )

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.gesture_history = deque(maxlen=5)
        self.last_trigger_time = 0
        self.cooldown = 1.0 # seconds

    def get_gesture(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        current_gesture = None
        landmarks = None
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            lm = landmarks.landmark
            
            # Helper to check if finger is extended
            # Tips: 8, 12, 16, 20. MCPs: 5, 9, 13, 17
            def is_ext(tip, mcp): return lm[tip].y < lm[mcp].y
            
            idx_ext = is_ext(8, 5)
            mid_ext = is_ext(12, 9)
            rng_ext = is_ext(16, 13)
            pnk_ext = is_ext(20, 17)
            thumb_up = lm[4].y < lm[2].y and abs(lm[4].x - lm[2].x) < 0.1
            thumb_ext = abs(lm[4].x - lm[2].x) > 0.04 or lm[4].y < lm[3].y

            # Open Palm
            if idx_ext and mid_ext and rng_ext and pnk_ext and thumb_up:
                current_gesture = "OPEN PALM"
            # Peace Sign
            elif idx_ext and mid_ext and not rng_ext and not pnk_ext:
                # Ensure fingers are separated in a V-shape
                dist = ((lm[8].x - lm[12].x)**2 + (lm[8].y - lm[12].y)**2)**0.5
                if dist > 0.04: # Threshold for separation
                    current_gesture = "PEACE"
            # Thumbs Up
            elif thumb_up and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
                current_gesture = "THUMBS UP"
            # Fist
            elif not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not thumb_up:
                current_gesture = "FIST"
            # Number gestures (1-5) for quick filter selection
            elif idx_ext and not mid_ext and not rng_ext and not pnk_ext:
                current_gesture = "NUM_1"
            elif idx_ext and mid_ext and not rng_ext and not pnk_ext:
                current_gesture = "NUM_2"
            elif idx_ext and mid_ext and rng_ext and not pnk_ext:
                current_gesture = "NUM_3"
            elif idx_ext and mid_ext and rng_ext and pnk_ext and not thumb_ext:
                current_gesture = "NUM_4"
            elif idx_ext and mid_ext and rng_ext and pnk_ext and thumb_ext:
                current_gesture = "NUM_5"

        self.gesture_history.append(current_gesture)
        
        # Smoothing: check if same gesture appears 4/5 times
        triggered_gesture = None
        if len(self.gesture_history) == 5:
            for g in ["OPEN PALM", "PEACE", "THUMBS UP", "FIST", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5"]:
                if self.gesture_history.count(g) >= 4:
                    triggered_gesture = g
                    break
        
        action_triggered = False
        now = time.time()
        if triggered_gesture and (now - self.last_trigger_time > self.cooldown):
            # Only trigger PEACE, THUMBS UP, FIST once. OPEN PALM is state-based.
            if triggered_gesture != "OPEN PALM":
                self.last_trigger_time = now
            action_triggered = True
            
        return triggered_gesture, landmarks, action_triggered

    def draw_debug(self, frame, landmarks):
        if landmarks:
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame