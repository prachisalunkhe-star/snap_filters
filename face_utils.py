import cv2
import mediapipe as mp
import numpy as np
import math

class FaceDetector:
    def __init__(self, max_faces=5):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Incompatible MediaPipe build detected. "
                "This app requires the legacy Solutions API. "
                "Run: pip install mediapipe==0.10.14"
            )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_face_data(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        face_data_list = []
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Landmarks indices:
                # Forehead: 10, Nose Tip: 1, Chin: 152, Left Cheek: 234, Right Cheek: 454
                # Left Eye Center: 468, Right Eye Center: 473
                
                def get_pt(idx):
                    lm = landmarks.landmark[idx]
                    return int(lm.x * w), int(lm.y * h)

                l_eye = get_pt(468)
                r_eye = get_pt(473)
                nose = get_pt(1)
                forehead = get_pt(10)
                l_cheek = get_pt(234)
                r_cheek = get_pt(454)
                chin = get_pt(152)
                
                # Face width (distance between cheeks)
                face_width = math.hypot(r_cheek[0] - l_cheek[0], r_cheek[1] - l_cheek[1])
                
                # Roll angle based on eyes
                angle = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
                
                # Bounding box for effects
                all_x = [lm.x * w for lm in landmarks.landmark]
                all_y = [lm.y * h for lm in landmarks.landmark]
                bbox = (int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y)))

                face_data = {
                    "left_eye": l_eye,
                    "right_eye": r_eye,
                    "nose_tip": nose,
                    "forehead": forehead,
                    "left_cheek": l_cheek,
                    "right_cheek": r_cheek,
                    "chin": chin,
                    "face_width": face_width,
                    "face_tilt_angle": angle,
                    "bbox": bbox,
                    "raw_landmarks": landmarks
                }
                face_data_list.append(face_data)
                
        return face_data_list

    def draw_debug(self, frame, face_data_list):
        for fd in face_data_list:
            for key in ["left_eye", "right_eye", "nose_tip", "forehead", "chin"]:
                cv2.circle(frame, fd[key], 3, (0, 255, 0), -1)
            bx1, by1, bx2, by2 = fd["bbox"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
        return frame

    def close(self):
        self.face_mesh.close()