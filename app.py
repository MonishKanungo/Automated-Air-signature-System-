import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from PIL import Image
import warnings
import os 
from PyPDF2 import PdfReader, PdfWriter
import io
from reportlab.pdfgen import canvas



# Suppress warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Helper function to optimize signature
def optimize_signature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)
    return Image.fromarray(img_thresh)

# Streamlit UI
st.title("Automated Signature System")
st.image("logo.png", width=1000)
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        position: fixed;
        top: 10px;  /* Adjust the vertical placement */
        left: 10px; /* Adjust the horizontal placement */
        z-index: 1000; /* Ensures the logo stays on top */
    }
    .logo-container img {
        width: 150px; /* Adjust the width of the logo */
        height: auto; /* Maintain aspect ratio */
    }
    </style>
    <div class="logo-container">
        <img src="https://via.placeholder.com/150" alt="logo.png">
    </div>
    """,
    unsafe_allow_html=True
)


# Initialize variables for Air Canvas
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

paintWindow = np.zeros((800, 1200, 3), dtype=np.uint8) + 255
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

if st.sidebar.button("Start Air Canvas"):
    st.write("Press 'S' to save and 'Q' to close the window")

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

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
    cv2.destroyAllWindows()

# Helper function to optimize the signature
def optimize_signature(signature_path):
    try:
        # Step 1: Load the saved signature
        signature = cv2.imread(signature_path, cv2.IMREAD_COLOR)

        # Step 2: Convert to Grayscale
        gray_signature = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply Thresholding for Binary Image
        _, binary_signature = cv2.threshold(gray_signature, 200, 255, cv2.THRESH_BINARY)

        # Step 4: Crop Unnecessary Margins
        contours, _ = cv2.findContours(binary_signature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour (assuming it's the signature)
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped_signature = binary_signature[y : y + h, x : x + w]

            # Step 5: Smooth and Denoise
            # Apply Gaussian Blur
            blurred_signature = cv2.GaussianBlur(cropped_signature, (3, 3), 0)

            # Apply Morphological Operations (Dilate to Thicken Lines)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            final_signature = cv2.dilate(blurred_signature, kernel, iterations=1)

            # Convert to PIL Image
            final_signature_pil = Image.fromarray(final_signature)

            return final_signature_pil
        else:
            raise ValueError("No contours found. Please check the input signature image.")
    except Exception as e:
        raise RuntimeError(f"Error during optimization: {e}")





# Optimize button logic
if st.sidebar.button("Optimize Signature"):
    try:
        signature_path = "signature.png"  # Ensure the signature is saved in this path beforehand
        optimized_img = optimize_signature(signature_path)
        st.image(optimized_img, caption="Optimized Signature", use_column_width=True)

        # Save the optimized image for download
        optimized_img.save("signature_final_optimized.png", optimize=True)
        st.success("Signature optimized successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

# Download button logic



def add_signature_to_pdf(input_pdf_path, output_pdf_path, signature_image_path, margin=(10, 10)):
    """
    Adds a signature to the bottom-right corner of a PDF with a specified margin.

    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_pdf_path (str): Path to save the signed PDF.
        signature_image_path (str): Path to the signature image file.
        margin (tuple): Margin (x_margin, y_margin) from the bottom-right corner.
    """
    # Load the input PDF
    pdf_reader = PdfReader(input_pdf_path)
    pdf_writer = PdfWriter()

    # Open the signature image
    signature = Image.open(signature_image_path)
    signature_width, signature_height = 200, 90  # Adjust size of the signature as needed

    # Process each page of the PDF
    for page in pdf_reader.pages:
        # Get the page dimensions and convert to float
        page_width = float(page.mediabox[2])  # Top-right x-coordinate
        page_height = float(page.mediabox[3])  # Top-right y-coordinate

        # Calculate the position for the signature
        x_position = page_width - signature_width - margin[0]
        y_position = margin[1]

        # Create a new PDF overlay
        signature_pdf = io.BytesIO()
        c = canvas.Canvas(signature_pdf, pagesize=(page_width, page_height))
        c.drawImage(signature_image_path, x_position, y_position, width=signature_width, height=signature_height)
        c.save()

        # Load the overlay as a PDF
        signature_pdf.seek(0)
        signature_overlay = PdfReader(signature_pdf)

        # Merge the overlay with the current page
        page.merge_page(signature_overlay.pages[0])
        pdf_writer.add_page(page)

    # Save the final PDF
    with open(output_pdf_path, "wb") as output_file:
        pdf_writer.write(output_file)

# Streamlit App
st.sidebar.title("Upload and Sign PDF")

# Upload PDF file
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Add signature button
if st.sidebar.button("Add Signature to PDF"):
    if uploaded_pdf:
        # Save the uploaded PDF temporarily
        temp_pdf_path = "temp_uploaded_document.pdf"
        with open(temp_pdf_path, "wb") as temp_pdf:
            temp_pdf.write(uploaded_pdf.read())

        # Check if signature image exists
        signature_image_path = "signature_final_optimized.png"
        if not os.path.exists(signature_image_path):
            st.error("Signature image not found! Please upload or optimize your signature first.")
        else:
            try:
                # Define the output PDF path
                output_pdf_path = "signed_document.pdf"

                # Add signature to the PDF
                margin = (20, 20)  # Margin from bottom-right corner
                add_signature_to_pdf(temp_pdf_path, output_pdf_path, signature_image_path, margin)
                st.success("Signature added to the PDF successfully!")

                # Offer the signed PDF for download
                with open(output_pdf_path, "rb") as signed_pdf:
                    st.download_button(
                        label="Download Signed PDF",
                        data=signed_pdf,
                        file_name="signed_document.pdf",
                        mime="application/pdf",
                    )
            except Exception as e:
                st.error(f"Error while adding signature: {e}")
    else:
        st.error("Please upload a PDF file first.")





