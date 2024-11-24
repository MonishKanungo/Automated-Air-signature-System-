import cv2
from PIL import Image

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
