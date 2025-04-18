import cv2
import os

# Set the directory where captured images will be saved.
save_dir = "captured_images/object_eyedrop" 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Open the Logitech C270 (typically at index 0; change if needed)
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Adjust width if needed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Adjust height if needed

print("Live capture started. Press SPACE to capture an image, ESC to quit.")

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Display the live video feed
    cv2.imshow("Live Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # If SPACEBAR is pressed, capture and save the image.
    if key == 32:
        img_count += 1
        filename = os.path.join(save_dir, f"img_{img_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Captured: {filename}")
    
    # If ESC is pressed, exit the capture loop.
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

