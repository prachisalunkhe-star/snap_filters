import sys
import cv2
import numpy as np
import os
import time
from datetime import datetime

# Local imports from the same directory
from face_utils import FaceDetector
from hand_tracker import HandTracker
from sidebar import Sidebar
from filter_manager import FilterManager

def main():
    print("[INIT] Starting Snap Filter AI...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, "filters", "assets")
    SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
    PICTURES_SNAP_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "snap")
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs(PICTURES_SNAP_DIR, exist_ok=True)

    # 1. Robust Camera Initialization
    print("[STEP 1] Initializing Camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARN] Camera index 0 failed. Trying index 1...")
        sys.stdout.flush()
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("[FATAL] No webcam detected! Check connections or permissions.")
        sys.exit(1)

    # Standard HD Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 2. Component Initialization with error trapping
    print("[STEP 2] Loading AI Models (this may take a moment)...")
    try:
        # Initialize Face Mesh
        detector = FaceDetector(max_faces=2)
        # Initialize Filter Logic
        filter_mgr = FilterManager(ASSETS_DIR)
        # Initialize Hand Tracking
        tracker = HandTracker()
        sidebar = Sidebar()
        print("[SUCCESS] All AI modules loaded.")
    except Exception as e:
        print(f"[FATAL] Failed to initialize AI modules: {e}")
        return

    active_idx = 0
    paused = False
    sidebar_offset = 0
    flash_frames = 0
    review_mode = False
    pending_capture = None
    countdown_start = None
    countdown_seconds = 3
    status_text = ""
    status_until = 0.0

    ui_state = {
        "capture_rect": None,
        "save_rect": None,
        "discard_rect": None
    }
    click_flags = {
        "capture": False,
        "save": False,
        "discard": False
    }

    def in_rect(px, py, rect):
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        return x1 <= px <= x2 and y1 <= py <= y2

    def save_frame_to_pictures(img):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(PICTURES_SNAP_DIR, f"snap_{ts}.png")
        cv2.imwrite(out_path, img)
        return out_path

    def draw_button(canvas, rect, label, color, icon=None):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (230, 230, 230), 2)
        if icon == "camera":
            # Draw a simple camera glyph on the left side.
            icon_h = max(18, (y2 - y1) - 20)
            icon_w = int(icon_h * 1.2)
            ix1 = x1 + 10
            iy1 = y1 + ((y2 - y1) - icon_h) // 2
            ix2 = ix1 + icon_w
            iy2 = iy1 + icon_h
            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (28, 28, 28), -1)
            cv2.rectangle(canvas, (ix1 + 6, iy1 - 6), (ix1 + 24, iy1), (28, 28, 28), -1)
            lens_r = max(4, icon_h // 4)
            cv2.circle(canvas, (ix1 + icon_w // 2, iy1 + icon_h // 2), lens_r, (220, 220, 220), 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            tx = ix2 + 10
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        else:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    def mouse_callback(event, x, y, flags, param):
        nonlocal active_idx, sidebar_offset
        if event == cv2.EVENT_LBUTTONDOWN:
            if review_mode:
                if in_rect(x, y, ui_state["save_rect"]):
                    click_flags["save"] = True
                    return
                if in_rect(x, y, ui_state["discard_rect"]):
                    click_flags["discard"] = True
                    return
            elif in_rect(x, y, ui_state["capture_rect"]):
                click_flags["capture"] = True
                return

            # Handle clicking on sidebar
            if x < sidebar.width:
                idx = (y + sidebar_offset) // sidebar.btn_h
                if 0 <= idx < len(filter_mgr.filters):
                    active_idx = idx
        # Handle scrolling
        elif event == cv2.EVENT_MOUSEWHEEL:
            max_offset = max(0, len(filter_mgr.filters) * sidebar.btn_h - 720)
            if flags > 0:
                sidebar_offset = max(0, sidebar_offset - 40)
            else:
                sidebar_offset = min(max_offset, sidebar_offset + 40)
        sidebar.update_hover(x, y + sidebar_offset)

    cv2.namedWindow("Snap Filter")
    cv2.setMouseCallback("Snap Filter", mouse_callback)

    print("[STEP 3] Entering Main Loop. Press 'Q' to quit.")
    sys.stdout.flush()
    
    retry_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            retry_count += 1
            if retry_count > 30:
                print("[FATAL] Lost camera connection.")
                break
            time.sleep(0.1)
            continue
        
        retry_count = 0
        frame = cv2.flip(frame, 1)
        h_f, w_f = frame.shape[:2]
        display_frame = frame.copy()
        
        # Run hand gestures only when not reviewing capture
        gesture = None
        triggered = False
        if not review_mode:
            gesture, hand_lms, triggered = tracker.get_gesture(frame)
            paused = gesture == "OPEN PALM"
            if triggered:
                if gesture == "THUMBS UP":
                    active_idx = (active_idx + 1) % len(filter_mgr.filters)
                elif gesture == "FIST":
                    active_idx = 0
                elif gesture and gesture.startswith("NUM_"):
                    num = int(gesture.split("_")[1])
                    active_idx = min(num, len(filter_mgr.filters) - 1)
                    status_text = f"Filter {active_idx}: {filter_mgr.filters[active_idx]['name']}"
                    status_until = time.time() + 1.5
                elif gesture == "PEACE" and countdown_start is None:
                    countdown_start = time.time()
                    status_text = "Timer started (3s)"
                    status_until = time.time() + 1.5

        face_data = detector.get_face_data(frame)

        # Apply Filter
        if not paused:
            display_frame = filter_mgr.apply(active_idx, display_frame, face_data)

        # Timer capture flow (PEACE gesture)
        if countdown_start is not None and not review_mode:
            elapsed = time.time() - countdown_start
            if elapsed >= countdown_seconds:
                pending_capture = display_frame.copy()
                review_mode = True
                countdown_start = None
                flash_frames = 2
            else:
                remaining = max(1, countdown_seconds - int(elapsed))
                cv2.putText(
                    display_frame,
                    f"CAPTURE IN {remaining}",
                    (40, 80),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.3,
                    (0, 215, 255),
                    3
                )

        # UI Rendering
        sb_img = sidebar.draw(filter_mgr.filters, filter_mgr.thumbnails, active_idx, sidebar_offset)
        sidebar_full = np.full((h_f, sidebar.width, 3), (30, 30, 30), dtype=np.uint8)
        # Ensure sidebar doesn't exceed frame height
        sb_h = min(sb_img.shape[0], h_f)
        sidebar_full[:sb_h, :] = sb_img[:sb_h, :]

        right_panel = pending_capture.copy() if (review_mode and pending_capture is not None) else display_frame
        combined = np.hstack([sidebar_full, right_panel])

        # Click-triggered actions
        if click_flags["capture"] and not review_mode:
            pending_capture = display_frame.copy()
            review_mode = True
            countdown_start = None
            click_flags["capture"] = False
            flash_frames = 2
        if click_flags["save"] and review_mode and pending_capture is not None:
            save_path = save_frame_to_pictures(pending_capture)
            review_mode = False
            pending_capture = None
            click_flags["save"] = False
            status_text = f"Saved: {save_path}"
            status_until = time.time() + 2.5
        if click_flags["discard"] and review_mode:
            review_mode = False
            pending_capture = None
            click_flags["discard"] = False
            status_text = "Capture discarded"
            status_until = time.time() + 1.8

        if paused:
            cv2.putText(combined, "|| PAUSED", (sidebar.width + 50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 200, 255), 3)

        # Capture and review controls
        if review_mode:
            overlay = combined.copy()
            cv2.rectangle(overlay, (sidebar.width + 20, 20), (sidebar.width + w_f - 20, 120), (20, 20, 20), -1)
            combined = cv2.addWeighted(overlay, 0.35, combined, 0.65, 0)
            cv2.putText(combined, "Capture Review", (sidebar.width + 40, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, "Save or Discard", (sidebar.width + 40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)

            save_rect = (sidebar.width + w_f - 280, 36, sidebar.width + w_f - 150, 86)
            discard_rect = (sidebar.width + w_f - 140, 36, sidebar.width + w_f - 20, 86)
            ui_state["save_rect"] = save_rect
            ui_state["discard_rect"] = discard_rect
            ui_state["capture_rect"] = None
            draw_button(combined, save_rect, "Save", (110, 220, 140))
            draw_button(combined, discard_rect, "Discard", (120, 170, 255))
        else:
            btn_w = 170
            btn_h = 50
            cap_x1 = sidebar.width + (w_f - btn_w) // 2
            cap_y1 = h_f - btn_h - 20
            cap_rect = (cap_x1, cap_y1, cap_x1 + btn_w, cap_y1 + btn_h)
            ui_state["capture_rect"] = cap_rect
            ui_state["save_rect"] = None
            ui_state["discard_rect"] = None
            draw_button(combined, cap_rect, "Capture", (120, 170, 255), icon="camera")

        if flash_frames > 0:
            combined[:, sidebar.width:] = 255
            flash_frames -= 1

        if status_text and time.time() < status_until:
            cv2.putText(
                combined,
                status_text,
                (sidebar.width + 20, h_f - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Snap Filter", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not review_mode:
            pending_capture = display_frame.copy()
            review_mode = True
            countdown_start = None
            flash_frames = 2
        elif key == ord('s') and review_mode and pending_capture is not None:
            save_path = save_frame_to_pictures(pending_capture)
            review_mode = False
            pending_capture = None
            status_text = f"Saved: {save_path}"
            status_until = time.time() + 2.5
        elif key == ord('d') and review_mode:
            review_mode = False
            pending_capture = None
            status_text = "Capture discarded"
            status_until = time.time() + 1.8
        elif ord('0') <= key <= ord('9'):
            val = key - ord('0')
            if val < len(filter_mgr.filters):
                active_idx = val

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()