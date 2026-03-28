# Snap Filter AI

Real-time face filter application using MediaPipe and OpenCV.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If you already installed dependencies earlier, reinstall MediaPipe explicitly:
   ```bash
   pip uninstall -y mediapipe
   pip install mediapipe==0.10.14
   ```
2. Generate placeholder assets:
   ```bash
   python generate_assets.py
   ```
3. Run application:
   ```bash
   python main.py
   ```

## Gesture Guide
| Gesture | Action |
| --- | --- |
| Open Palm | Pause/Freeze Filter |
| Peace Sign | Start 3-second timer and capture |
| Thumbs Up | Next Filter |
| Fist | Reset to No Filter |
| 1 to 5 Fingers | Select filter number 1 to 5 |

## Capture Workflow
- Click the `Capture` button (or press `C`) to open review mode.
- In review mode, choose `Save` or `Discard` (or use `S` / `D` keys).
- Saved images go to your Pictures folder at: `~/Pictures/snap` (Windows: `C:\Users\<you>\Pictures\snap`).

## Adding Custom Filters
1. Add your `.png` overlay to `filters/assets/`.
2. Open `filters/filter_manager.py`.
3. Append a new dictionary to `self.filters` with the asset name and desired anchor point (`eyes`, `nose`, or `forehead`).