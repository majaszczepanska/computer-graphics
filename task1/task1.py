from pathlib import Path
import cv2
import matplotlib.pyplot as plt

#load the image from file
image_path = Path(__file__).parent / "view1.jpg"
img_bgr = cv2.imread(str(image_path))

if img_bgr is None:
    raise FileNotFoundError("Image not found: 'view1.jpg'!")

#convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#show the original image
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

#resize image to 50% and convert to grayscale
height, width = img_bgr.shape[:2]
new_size = (int(width * 0.5), int(height * 0.5))
img_resized = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

#rotate image by 90 degrees
img_rotated = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

#show the result image
plt.figure(figsize=(6, 6))
plt.imshow(img_rotated, cmap="gray")
plt.title("Result (50% size, grayscale, rotated 90°)")
plt.axis("off")
plt.show()

#print the image array (numbers / matrix)
print("Image shape (height, width):", img_rotated.shape)
print("Image matrix (pixel values 0-255):")
print(img_rotated)