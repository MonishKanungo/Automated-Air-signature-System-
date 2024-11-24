import cv2

# Load the saved signature
signature = cv2.imread("signature.png")

# Convert to grayscale
gray_signature = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)

# Remove noise
blurred_signature = cv2.GaussianBlur(gray_signature, (5, 5), 0)

# Apply binary thresholding
_, binary_signature = cv2.threshold(blurred_signature, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Define a kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Dilate to strengthen the strokes
dilated_signature = cv2.dilate(binary_signature, kernel, iterations=1)

# Erode to refine the edges
optimized_signature = cv2.erode(dilated_signature, kernel, iterations=1)

# Save the optimized signature
cv2.imwrite("optimized_signature.png", optimized_signature)
print("Optimized signature saved as 'optimized_signature.png'")
