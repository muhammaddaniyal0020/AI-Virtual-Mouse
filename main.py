"""
main.py
-------
AI Virtual Mouse - control your mouse cursor with hand gestures using a
webcam, OpenCV, and MediaPipe Hands.

Features:
    * Smooth (EMA-based) cursor movement with dead-zone filtering
    * Full-monitor coordinate mapping via numpy interpolation
    * Left click, right click, double click
    * Click / double-click / screenshot cooldowns to prevent repeats
    * Drag & drop (pinch thumb + index to grab, release to drop)
    * Scroll up / down
    * Timestamped screenshots saved to a "Screenshots/" folder
    * Real-time FPS counter and on-screen status overlay
    * Defensive error handling so the app never crashes on camera
      loss or missing hand detections

Run with:
    python main.py
Press 'q' with the video window focused to quit.
"""

import os
import time
import random
from datetime import datetime

import cv2
import mediapipe as mp
import pyautogui
import util
from pynput.mouse import Button, Controller

mouse = Controller()

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
SMOOTHING = 0.7                 # EMA smoothing factor (0 = none, closer to 1 = smoother/laggier)
FRAME_MARGIN = 100              # px inset from the camera frame edges used when mapping to the screen
CLICK_COOLDOWN = 0.5            # seconds between left/right clicks
DOUBLE_CLICK_COOLDOWN = 1.0     # seconds between double clicks
SCREENSHOT_COOLDOWN = 2.0       # seconds between screenshots
SCROLL_COOLDOWN = 0.15          # seconds between individual scroll "ticks"
SCROLL_SPEED = 40               # scroll units applied per tick
DEAD_ZONE = 4                   # px; ignore cursor moves smaller than this to kill jitter
SCROLL_DEAD_ZONE_PX = 8         # px; vertical movement needed to register a scroll tick
PINCH_THRESHOLD = 40            # thumb-tip/index-tip distance (0-1000 scale) below which counts as a pinch
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
SCREENSHOT_DIR = "Screenshots"

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False  # gestures already stay within the frame; avoid corner-triggered exceptions

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    max_num_hands=1,
)


# --------------------------------------------------------------------------
# LANDMARK / GESTURE HELPERS
# --------------------------------------------------------------------------
def find_finger_tip(processed):
    """
    Return the index-finger-tip landmark of the first detected hand,
    or None if no hand was detected.
    """
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None


def is_move_gesture(landmark_list, thumb_index_dist):
    """Index finger extended while the thumb rests near the index base."""
    return (
        thumb_index_dist is not None
        and thumb_index_dist < 50
        and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
    )


def is_left_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
        and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90
        and thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
        and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
        and thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
        and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
        and thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50
        and util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50
        and thumb_index_dist < 50
    )


def is_pinching(landmark_list):
    """Thumb tip close to index tip -> used to trigger drag & drop."""
    pinch_dist = util.get_distance([landmark_list[4], landmark_list[8]])
    return pinch_dist is not None and pinch_dist < PINCH_THRESHOLD


def is_scroll_pose(landmark_list):
    """Index, middle and ring fingers extended while the pinky is curled."""
    index_extended = util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
    middle_extended = util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90
    ring_extended = util.get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 90
    pinky_curled = util.get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) < 50
    return index_extended and middle_extended and ring_extended and pinky_curled


# --------------------------------------------------------------------------
# CURSOR / ACTION HANDLERS
# --------------------------------------------------------------------------
def compute_target_cursor(index_finger_tip):
    """
    Map the (normalized) index-finger-tip position to full-monitor
    screen coordinates using numpy interpolation, insetting the usable
    camera frame by FRAME_MARGIN so the whole screen is reachable
    without requiring the finger to reach the extreme edges.
    """
    raw_x = index_finger_tip.x * CAMERA_WIDTH
    raw_y = index_finger_tip.y * CAMERA_HEIGHT

    screen_x = util.interpolate(
        raw_x, [FRAME_MARGIN, CAMERA_WIDTH - FRAME_MARGIN], [0, SCREEN_WIDTH]
    )
    screen_y = util.interpolate(
        raw_y, [FRAME_MARGIN, CAMERA_HEIGHT - FRAME_MARGIN], [0, SCREEN_HEIGHT]
    )

    screen_x = util.clamp(screen_x, 0, SCREEN_WIDTH - 1)
    screen_y = util.clamp(screen_y, 0, SCREEN_HEIGHT - 1)
    return screen_x, screen_y


def move_mouse(state, index_finger_tip):
    """Smoothly move the OS cursor toward the mapped target position."""
    if index_finger_tip is None:
        return

    target = compute_target_cursor(index_finger_tip)
    smoothed = util.smooth_cursor(state["prev_cursor"], target, SMOOTHING)

    if state["prev_cursor"] is not None:
        dx = smoothed[0] - state["prev_cursor"][0]
        dy = smoothed[1] - state["prev_cursor"][1]
        if abs(dx) < DEAD_ZONE and abs(dy) < DEAD_ZONE:
            return  # inside the dead zone; skip to avoid jitter

    pyautogui.moveTo(int(smoothed[0]), int(smoothed[1]))
    state["prev_cursor"] = smoothed


def take_screenshot(frame):
    """Save a timestamped screenshot into the Screenshots/ directory."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(SCREENSHOT_DIR, f"Screenshot_{timestamp}.png")
    try:
        image = pyautogui.screenshot()
        image.save(filename)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Failed to save screenshot: {exc}")
    cv2.putText(frame, "Screenshot Taken", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def start_drag(state, frame):
    if not state["dragging"]:
        mouse.press(Button.left)
        state["dragging"] = True
    cv2.putText(frame, "Gesture : Dragging", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


def stop_drag(state):
    if state["dragging"]:
        mouse.release(Button.left)
        state["dragging"] = False


def handle_scroll(state, landmark_list, frame):
    """
    While the scroll pose is held, compare the current vertical
    position of the middle-finger tip against the previous frame to
    decide whether to scroll up or down.
    """
    current_y = landmark_list[12][1] * CAMERA_HEIGHT
    now = time.time()
    direction_label = None

    if state["prev_scroll_y"] is not None and now - state["last_scroll_time"] > SCROLL_COOLDOWN:
        delta = current_y - state["prev_scroll_y"]
        if delta < -SCROLL_DEAD_ZONE_PX:
            mouse.scroll(0, SCROLL_SPEED / 40)  # positive -> scroll up
            direction_label = "Scroll Up"
            state["last_scroll_time"] = now
        elif delta > SCROLL_DEAD_ZONE_PX:
            mouse.scroll(0, -SCROLL_SPEED / 40)  # negative -> scroll down
            direction_label = "Scroll Down"
            state["last_scroll_time"] = now

    state["prev_scroll_y"] = current_y

    if direction_label:
        cv2.putText(frame, f"Gesture : {direction_label}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 200), 2)
    else:
        cv2.putText(frame, "Gesture : Scroll Ready", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 200), 2)


# --------------------------------------------------------------------------
# MAIN GESTURE DISPATCH
# --------------------------------------------------------------------------
def detect_gesture(frame, landmark_list, processed, state):
    """
    Inspect the current frame's hand landmarks and route to the
    appropriate action (move / click / drag / scroll / screenshot),
    respecting per-gesture cooldowns. Updates `state["gesture_label"]`
    for the on-screen overlay.
    """
    now = time.time()
    state["gesture_label"] = "None"

    if len(landmark_list) < 21:
        state["gesture_label"] = "No Hand Detected"
        stop_drag(state)
        state["prev_cursor"] = None
        state["prev_scroll_y"] = None
        return

    index_finger_tip = find_finger_tip(processed)
    thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])
    pinching = is_pinching(landmark_list)

    # Release an active drag as soon as the pinch is let go, regardless
    # of whatever gesture is detected next.
    if state["dragging"] and not pinching:
        stop_drag(state)

    if pinching:
        state["gesture_label"] = "Dragging"
        start_drag(state, frame)
        if index_finger_tip is not None:
            move_mouse(state, index_finger_tip)  # allow dragging while moving

    elif is_move_gesture(landmark_list, thumb_index_dist):
        state["gesture_label"] = "Moving"
        move_mouse(state, index_finger_tip)
        cv2.putText(frame, "Gesture : Moving", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    elif is_scroll_pose(landmark_list):
        state["gesture_label"] = "Scrolling"
        handle_scroll(state, landmark_list, frame)

    elif is_left_click(landmark_list, thumb_index_dist):
        if now - state["last_click_time"] > CLICK_COOLDOWN:
            mouse.press(Button.left)
            mouse.release(Button.left)
            state["last_click_time"] = now
        state["gesture_label"] = "Left Click"
        cv2.putText(frame, "Gesture : Left Click", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    elif is_right_click(landmark_list, thumb_index_dist):
        if now - state["last_click_time"] > CLICK_COOLDOWN:
            mouse.press(Button.right)
            mouse.release(Button.right)
            state["last_click_time"] = now
        state["gesture_label"] = "Right Click"
        cv2.putText(frame, "Gesture : Right Click", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    elif is_double_click(landmark_list, thumb_index_dist):
        if now - state["last_double_click_time"] > DOUBLE_CLICK_COOLDOWN:
            pyautogui.doubleClick()
            state["last_double_click_time"] = now
        state["gesture_label"] = "Double Click"
        cv2.putText(frame, "Gesture : Double Click", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    elif is_screenshot(landmark_list, thumb_index_dist):
        if now - state["last_screenshot_time"] > SCREENSHOT_COOLDOWN:
            take_screenshot(frame)
            state["last_screenshot_time"] = now
        state["gesture_label"] = "Screenshot"

    if state["gesture_label"] not in ("Dragging", "Moving", "Scrolling"):
        state["prev_cursor"] = None
    if state["gesture_label"] != "Scrolling":
        state["prev_scroll_y"] = None


# --------------------------------------------------------------------------
# OVERLAY / DRAWING
# --------------------------------------------------------------------------
def draw_tracking_visuals(frame, hand_landmarks, landmark_list):
    """Draw hand landmarks, connections, thumb/index tips, and a tracking box."""
    mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    if len(landmark_list) >= 9:
        thumb_px = (int(landmark_list[4][0] * CAMERA_WIDTH), int(landmark_list[4][1] * CAMERA_HEIGHT))
        index_px = (int(landmark_list[8][0] * CAMERA_WIDTH), int(landmark_list[8][1] * CAMERA_HEIGHT))
        cv2.circle(frame, thumb_px, 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, index_px, 10, (0, 255, 255), cv2.FILLED)

    xs = [lm[0] * CAMERA_WIDTH for lm in landmark_list]
    ys = [lm[1] * CAMERA_HEIGHT for lm in landmark_list]
    if xs and ys:
        pad = 20
        x1, y1 = int(min(xs) - pad), int(min(ys) - pad)
        x2, y2 = int(max(xs) + pad), int(max(ys) + pad)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)


def draw_overlay(frame, fps, state, tracking_active):
    """Render the professional status overlay: FPS, gesture, cursor/tracking status, resolution."""
    overlay_lines = [
        f"FPS: {fps:.1f}",
        f"Gesture: {state['gesture_label']}",
        f"Cursor: {'Dragging' if state['dragging'] else 'Active' if tracking_active else 'Idle'}",
        f"Tracking: {'Locked' if tracking_active else 'Searching...'}",
        f"Screen Res: {SCREEN_WIDTH}x{SCREEN_HEIGHT}",
    ]

    y = frame.shape[0] - 15 - (len(overlay_lines) - 1) * 22
    for line in overlay_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open the camera. Check that it is connected and not in use by another app.")
        return

    state = {
        "prev_cursor": None,
        "dragging": False,
        "last_click_time": 0.0,
        "last_double_click_time": 0.0,
        "last_screenshot_time": 0.0,
        "last_scroll_time": 0.0,
        "prev_scroll_y": None,
        "gesture_label": "None",
    }

    prev_frame_time = time.time()
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 30  # ~1s of dropped frames at 30 FPS before giving up

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                print("[WARN] Failed to read frame from camera; retrying...")
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                    print("[ERROR] Camera appears disconnected. Exiting gracefully.")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            try:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = hands.process(frame_rgb)

                landmark_list = []
                tracking_active = False

                if processed.multi_hand_landmarks:
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    for lm in hand_landmarks.landmark:
                        landmark_list.append((lm.x, lm.y))
                    draw_tracking_visuals(frame, hand_landmarks, landmark_list)
                    tracking_active = True
                else:
                    state["prev_cursor"] = None
                    state["prev_scroll_y"] = None
                    stop_drag(state)

                detect_gesture(frame, landmark_list, processed, state)

                now = time.time()
                elapsed = now - prev_frame_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                prev_frame_time = now

                draw_overlay(frame, fps, state, tracking_active)
                cv2.imshow("AI Virtual Mouse", frame)

            except Exception as exc:
                # Defensive: never let a single bad frame crash the app.
                print(f"[WARN] Skipped a frame due to an error: {exc}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        stop_drag(state)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
