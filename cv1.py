import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import cv2
from PIL import Image

# Giving different arrays to handle color points of different colors
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colors
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Canvas setup
paintWindow = np.zeros((800, 1200, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (50, 1), (200, 100), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (220, 1), (370, 100), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (540, 100), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (560, 1), (710, 100), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (730, 1), (880, 100), (0, 255, 255), 2)

cv2.putText(paintWindow, "CLEAR", (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (430, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (610, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (760, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    height, width, _ = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw the buttons
    frame = cv2.rectangle(frame, (50, 1), (200, 100), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (220, 1), (370, 100), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (540, 100), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (560, 1), (710, 100), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (730, 1), (880, 100), (0, 255, 255), 2)
    cv2.putText(frame, "CLEAR", (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (430, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (610, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (760, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post-process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * width)
                lmy = int(lm.y * height)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        # Thumb proximity for new points
        if abs(thumb[1] - center[1]) < 30:
            bpoints.append(deque(maxlen=1024))
            blue_index += 1
            gpoints.append(deque(maxlen=1024))
            green_index += 1
            rpoints.append(deque(maxlen=1024))
            red_index += 1
            ypoints.append(deque(maxlen=1024))
            yellow_index += 1

        elif center[1] <= 100:
            if 50 <= center[0] <= 200:  # Clear Button
                bpoints = [deque(maxlen=1024)]
                gpoints = [deque(maxlen=1024)]
                rpoints = [deque(maxlen=1024)]
                ypoints = [deque(maxlen=1024)]
                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
                paintWindow[101:, :, :] = 255
            elif 220 <= center[0] <= 370:
                colorIndex = 0  # Blue
            elif 390 <= center[0] <= 540:
                colorIndex = 1  # Green
            elif 560 <= center[0] <= 710:
                colorIndex = 2  # Red
            elif 730 <= center[0] <= 880:
                colorIndex = 3  # Yellow
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # Save the signature when 's' is pressed
    key = cv2.waitKey(1)
    if key == ord('s'):
        cropped_paint_window = paintWindow[101:, :, :]
        filename = "signature.png"
        cv2.imwrite(filename, cropped_paint_window)
        print(f"Signature saved as {filename}.")

    # Exit the program when 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAll


# Step 1: Load the saved signature
signature = cv2.imread("signature.png", cv2.IMREAD_COLOR)

# Step 2: Convert to Grayscale
gray_signature = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Thresholding for Binary Image
_, binary_signature = cv2.threshold(gray_signature, 200, 255, cv2.THRESH_BINARY)

# Step 4: Crop Unnecessary Margins
contours, _ = cv2.findContours(binary_signature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # Find the largest contour (assuming it's the signature)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped_signature = binary_signature[y:y+h, x:x+w]

    # Step 5: Smooth and Denoise
    # Apply Gaussian Blur
    blurred_signature = cv2.GaussianBlur(cropped_signature, (3, 3), 0)

    # Apply Morphological Operations (Dilate to Thicken Lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_signature = cv2.dilate(blurred_signature, kernel, iterations=1)

    # Step 6: Save as Compressed PNG
    final_image_path = "signature_final_optimized.png"
    final_signature_pil = Image.fromarray(final_signature)
    final_signature_pil.save(final_image_path, optimize=True)
    print(f"Final optimized signature saved as '{final_image_path}'.")
else:
    print("No contours found. Please check the input signature image.")

