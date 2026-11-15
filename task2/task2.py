import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load remote image from URL
url = "https://upload.wikimedia.org/wikipedia/commons/5/50/Shieldgreen%2C_a_house_in_the_forest_-_geograph.org.uk_-_6848009.jpg"
req = urllib.request.Request(
    url,
    headers={"User-Agent": "GKiPO-StudentProject/1.0 (https://github.com/maja)"}
)
with urllib.request.urlopen(req) as response:
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)

img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise ValueError("Failed to decode image from URL!")

# convert BGR to RGB and grayscale
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# plot image, overall histogram and RGB channels histogram
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
axes[1].plot(hist_gray, color="black")
axes[1].set_xlim([0, 256])
axes[1].set_title("Overall Histogram (Luminance)")
axes[1].set_xlabel("Pixel Value (0-255)")
axes[1].set_ylabel("Pixel Count")

for idx, col in enumerate(("r", "g", "b")):
    hist_ch = cv2.calcHist([img_rgb], [idx], None, [256], [0, 256])
    axes[2].plot(hist_ch, color=col, label=col.upper())
axes[2].set_xlim([0, 256])
axes[2].set_title("RGB Channels Histogram")
axes[2].set_xlabel("Pixel Value (0-255)")
axes[2].legend()

plt.tight_layout()
plt.show()