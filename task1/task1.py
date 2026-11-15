from pathlib import Path
import cv2
import matplotlib.pyplot as plt

#import image for this task
image_path = Path(__file__).parent / "view1.jpg"
img_bgr = cv2.imread(str(image_path))

if img_bgr is None:
    raise FileNotFoundError("File not found: 'view1.jpg'!")

#conver bgr to rgb (matplotlib))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#show original image
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("original")
plt.axis("off")
plt.show()