import cv2
import mediapipe as mp
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0)
}

radius = 25
circle_positions = None
selected_color = None
drawing_mode = False
prev_x, prev_y = None, None
spock_gesture_active = False 

cap = cv2.VideoCapture(0)
cv2.namedWindow('Air Canvas', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Air Canvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

canvas = None
mask = None

def is_finger_extended(p1, p2, threshold=30):
    """Check if finger is extended based on pixel distance"""
    distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return distance > threshold


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if canvas is None:
        h, w, _ = frame.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        num_colors = len(colors)
        margin = 50
        start_x = margin
        end_x = w - margin
        spacing = (end_x - start_x) / (num_colors - 1)
        circle_positions = {}
        y_position = margin + radius + 20
        for i, color_name in enumerate(colors.keys()):
            x = int(start_x + i * spacing)
            circle_positions[color_name] = (x, y_position)

    for color_name, pos in circle_positions.items():
        cv2.circle(frame, pos, radius, colors[color_name], -1)
        if selected_color == colors[color_name]:
            cv2.circle(frame, pos, radius + 5, (255,255,255), 3)

    results = hands.process(rgb_frame)
    left_hand_open = False
    right_hand_tip = None
    right_hand_mcp = None
    spock_detected = False

    if results.multi_hand_landmarks:
        handedness_list = results.multi_handedness
    
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = handedness_list[i].classification[0].label

            if handedness == 'Left':
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

                h, w = frame.shape[:2]
                index_tip_x = int(index_tip.x * w)
                index_tip_y = int(index_tip.y * h)
                middle_tip_x = int(middle_tip.x * w)
                middle_tip_y = int(middle_tip.y * h)
                ring_tip_x = int(ring_tip.x * w)
                ring_tip_y = int(ring_tip.y * h)
                pinky_tip_x = int(pinky_tip.x * w)
                pinky_tip_y = int(pinky_tip.y * h)
                thumb_tip_x = int(thumb_tip.x * w)
                thumb_tip_y = int(thumb_tip.y * h)
                thumb_mcp_x = int(thumb_mcp.x * w)
                thumb_mcp_y = int(thumb_mcp.y * h)

                d_index_middle = math.sqrt((index_tip_x - middle_tip_x)**2 + (index_tip_y - middle_tip_y)**2)
                d_ring_pinky = math.sqrt((ring_tip_x - pinky_tip_x)**2 + (ring_tip_y - pinky_tip_y)**2)
                d_middle_ring = math.sqrt((middle_tip_x - ring_tip_x)**2 + (middle_tip_y - ring_tip_y)**2)
                thumb_extended = is_finger_extended((thumb_tip_x, thumb_tip_y), (thumb_mcp_x, thumb_mcp_y), 20)

                spock_gesture = (
                    d_index_middle < 30 and   
                    d_ring_pinky < 30 and    
                    d_middle_ring > 50 and  
                    thumb_extended            
                )

                if spock_gesture:
                    spock_detected = True
                else:
                    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    tip_x, tip_y = int(tip.x * w), int(tip.y * h)
                    mcp_x, mcp_y = int(mcp.x * w), int(mcp.y * h)
                    if is_finger_extended((tip_x, tip_y), (mcp_x, mcp_y)):
                        left_hand_open = True

            elif handedness == 'Right':
                tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                h, w = frame.shape[:2]
                tip_x, tip_y = int(tip.x * w), int(tip.y * h)
                mcp_x, mcp_y = int(mcp.x * w), int(mcp.y * h)
                right_hand_tip = (tip_x, tip_y)
                right_hand_mcp = (mcp_x, mcp_y)
                cv2.circle(frame, (tip_x, tip_y), 12, (255,255,255), -1)

    if spock_detected:
        if not spock_gesture_active:
            canvas = np.zeros_like(canvas)
            mask = np.zeros_like(mask)
            spock_gesture_active = True
    else:
        spock_gesture_active = False

    if right_hand_tip is not None:
        color_selected = False
        for color_name, pos in circle_positions.items():
            cx, cy = pos
            if (right_hand_tip[0]-cx)**2 + (right_hand_tip[1]-cy)**2 <= radius**2:
                selected_color = colors[color_name]
                color_selected = True
                break
        
        if not color_selected:
            drawing_mode = is_finger_extended(right_hand_tip, right_hand_mcp)
        else:
            drawing_mode = False

        if left_hand_open and drawing_mode and selected_color is not None:
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), right_hand_tip, selected_color, 8)
                cv2.line(mask, (prev_x, prev_y), right_hand_tip, 255, 8)
            prev_x, prev_y = right_hand_tip
        else:
            prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    frame[mask != 0] = canvas[mask != 0]

    cv2.putText(frame, "Color Palette", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    cv2.putText(frame, "Press 'd' to clear | 's' to save | 'q' to quit", (30, h-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('Air Canvas', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        canvas = np.zeros_like(canvas)
        mask = np.zeros_like(mask)
    elif key == ord('s'):
        if canvas is not None and mask is not None:
            white_bg = np.ones_like(canvas) * 255
            white_bg[mask != 0] = canvas[mask != 0]
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                title="Save drawing"
            )
            if file_path:
                cv2.imwrite(file_path, white_bg)

cap.release()
cv2.destroyAllWindows()
hands.close()
root.destroy()