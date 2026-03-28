import cv2
import numpy as np
import os
from PIL import Image

class FilterManager:
    def __init__(self, assets_dir):
        self.assets_dir = assets_dir
        self.filters = [
            {"name": "No Filter", "type": "none"},
            {"name": "Sunglasses", "type": "png", "asset": "sunglasses.png", "scale": 1.1, "anchor": "eye_center", "y_offset_ratio": -0.05},
            {"name": "Dog Ears", "type": "png", "asset": "dog_ears.png", "scale": 1.5, "anchor": "forehead", "y_offset_ratio": -0.55},
            {"name": "Cat Ears", "type": "png", "asset": "cat_ears.png", "scale": 1.3, "anchor": "forehead", "y_offset_ratio": -0.55},
            {"name": "Crown", "type": "png", "asset": "crown.png", "scale": 1.2, "anchor": "forehead", "y_offset_ratio": -0.45},
            {"name": "Rainbow", "type": "png", "asset": "rainbow.png", "scale": 0.9, "anchor": "forehead", "y_offset_ratio": -0.8},
            {"name": "Flower Crown", "type": "png", "asset": "flower_crown.png", "scale": 1.4, "anchor": "forehead", "y_offset_ratio": -0.35},
            {"name": "Bunny Ears", "type": "png", "asset": "bunny_ears.png", "scale": 1.6, "anchor": "forehead", "y_offset_ratio": -0.6},
            {"name": "Fire Halo", "type": "png", "asset": "fire_halo.png", "scale": 1.3, "anchor": "forehead", "y_offset_ratio": -0.75},
            {"name": "Nerd Glasses", "type": "png", "asset": "nerd_glasses.png", "scale": 1.0, "anchor": "eye_center", "y_offset_ratio": -0.05},
            {"name": "Devil Horns", "type": "png", "asset": "devil_horns.png", "scale": 1.2, "anchor": "forehead", "y_offset_ratio": -0.6},
            {"name": "Pirate Hat", "type": "png", "asset": "pirate_hat.png", "scale": 1.8, "anchor": "forehead", "y_offset_ratio": -0.55},
            {"name": "Blush Cheeks", "type": "drawn_blush"},
            {"name": "Pixel Glasses", "type": "drawn_pixel"},
            {"name": "Distortion", "type": "warp"},
            {"name": "Beauty", "type": "beauty"}
        ]
        self.thumbnails = self._gen_thumbs()
        self.loaded_assets = {}
        self._load_assets()

    def _load_assets(self):
        for f in self.filters:
            if f["type"] == "png":
                path = os.path.join(self.assets_dir, f["asset"])
                if os.path.exists(path):
                    self.loaded_assets[f["asset"]] = Image.open(path).convert("RGBA")

    def _gen_thumbs(self):
        thumbs = []
        for f in self.filters:
            t = np.zeros((60, 60, 3), dtype=np.uint8)
            cv2.putText(t, f["name"][:5], (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            thumbs.append(t)
        return thumbs

    def apply(self, index, frame, face_data_list):
        if index == 0 or not face_data_list: return frame
        f_conf = self.filters[index]
        
        for face in face_data_list:
            if f_conf["type"] == "png":
                frame = self._overlay_png(frame, face, f_conf)
            elif f_conf["type"] == "drawn_blush":
                frame = self._apply_blush(frame, face)
            elif f_conf["type"] == "drawn_pixel":
                frame = self._apply_pixel_glasses(frame, face)
            elif f_conf["type"] == "warp":
                frame = self._apply_warp(frame, face)
            elif f_conf["type"] == "beauty":
                frame = self._apply_beauty(frame, face)
        return frame

    def _overlay_png(self, frame, face, conf):
        asset_name = conf["asset"]
        if asset_name not in self.loaded_assets: return frame
        
        pil_img = self.loaded_assets[asset_name]
        w_size = int(face["face_width"] * conf["scale"])
        ratio = w_size / float(pil_img.size[0])
        h_size = int(float(pil_img.size[1]) * ratio)
        
        if w_size < 1 or h_size < 1: return frame
        
        resized = pil_img.resize((w_size, h_size), Image.Resampling.LANCZOS)
        rotated = resized.rotate(-face["face_tilt_angle"], expand=True)
        
        eye_center = (
            (face["left_eye"][0] + face["right_eye"][0]) // 2,
            (face["left_eye"][1] + face["right_eye"][1]) // 2
        )
        anchor_kind = conf.get("anchor", "eye_center")
        if anchor_kind == "forehead":
            anchor = face["forehead"]
        elif anchor_kind == "nose":
            anchor = face["nose_tip"]
        else:
            anchor = eye_center

        y_off = int(conf.get("y_offset_ratio", 0.0) * face["face_width"])
        pos = (anchor[0] - rotated.size[0] // 2, anchor[1] - rotated.size[1] // 2 + y_off)
        frame_h, frame_w = frame.shape[:2]
        if pos[0] > frame_w or pos[1] > frame_h:
            return frame
        if pos[0] + rotated.size[0] < 0 or pos[1] + rotated.size[1] < 0:
            return frame
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        frame_pil.paste(rotated, pos, rotated)
        return cv2.cvtColor(np.array(frame_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _apply_blush(self, frame, face):
        overlay = frame.copy()
        for cheek in [face["left_cheek"], face["right_cheek"]]:
            cv2.ellipse(overlay, cheek, (int(face["face_width"]*0.15), int(face["face_width"]*0.1)), 
                        int(face["face_tilt_angle"]), 0, 360, (150, 100, 255), -1)
        return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    def _apply_pixel_glasses(self, frame, face):
        overlay = frame.copy()
        w = int(face["face_width"] * 0.3)
        h = int(w * 0.6)
        for eye in [face["left_eye"], face["right_eye"]]:
            x, y = eye[0] - w//2, eye[1] - h//2
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (10, 10, 10), -1)
            cv2.rectangle(overlay, (x+2, y+2), (x+8, y+8), (240, 240, 240), -1)
        return cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    def _apply_warp(self, frame, face):
        x1, y1, x2, y2 = face["bbox"]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return frame
        
        rows, cols = roi.shape[:2]
        map_x, map_y = np.zeros(roi.shape[:2], np.float32), np.zeros(roi.shape[:2], np.float32)
        
        for i in range(rows):
            for j in range(cols):
                map_x[i, j] = j + 15 * np.sin(2 * np.pi * i / 60.0)
                map_y[i, j] = i
        
        warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = warped
        return frame

    def _apply_beauty(self, frame, face):
        x1, y1, x2, y2 = face["bbox"]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return frame
        
        # Skin smoothing
        smooth = cv2.bilateralFilter(roi, 9, 75, 75)
        
        # Saturation boost
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        skin_mask = cv2.inRange(hsv, lower, upper)
        
        hsv[:,:,1] = np.where(skin_mask > 0, np.minimum(hsv[:,:,1] * 1.25, 255), hsv[:,:,1])
        frame[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame