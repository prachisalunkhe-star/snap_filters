import cv2
import numpy as np

class Sidebar:
    def __init__(self, width=160, button_height=80):
        self.width = width
        self.btn_h = button_height
        self.bg_color = (30, 30, 30)
        self.hover_idx = -1

    def draw(self, filters, thumbnails, active_idx, offset=0):
        canvas = np.full((len(filters) * self.btn_h, self.width, 3), self.bg_color, dtype=np.uint8)
        
        for i, f in enumerate(filters):
            y_top = i * self.btn_h
            # Hover effect
            if i == self.hover_idx:
                cv2.rectangle(canvas, (0, y_top), (self.width, y_top + self.btn_h), (50, 50, 50), -1)
            
            # Active highlight
            if i == active_idx:
                cv2.rectangle(canvas, (2, y_top + 2), (self.width - 2, y_top + self.btn_h - 2), (255, 255, 0), 3)

            # Thumbnail
            tx = (self.width - 60) // 2
            ty = y_top + 5
            canvas[ty:ty+60, tx:tx+60] = thumbnails[i]
            
            # Label
            text = f["name"]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.35
            (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
            cv2.putText(canvas, text, ( (self.width - tw)//2, y_top + 75), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Separator
            cv2.line(canvas, (10, y_top + self.btn_h - 1), (self.width - 10, y_top + self.btn_h - 1), (60, 60, 60), 1)
            
        # Return only the visible part based on offset
        h_limit = 720 # Default frame height
        if offset + h_limit > canvas.shape[0]:
            offset = canvas.shape[0] - h_limit
        
        return canvas[offset:offset+h_limit, :]

    def update_hover(self, mx, my):
        if 0 <= mx < self.width:
            self.hover_idx = my // self.btn_h
        else:
            self.hover_idx = -1
        return self.hover_idx