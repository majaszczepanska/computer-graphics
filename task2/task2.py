import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histograms(img_bgr, title_prefix=""):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"{title_prefix}Image")
    axes[0].axis("off")

    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    axes[1].plot(hist_gray, color="black")
    axes[1].set_xlim([0, 256])
    axes[1].set_title(f"{title_prefix}Overall Histogram (Luminance)")
    axes[1].set_xlabel("Pixel Value (0-255)")
    axes[1].set_ylabel("Pixel Count")

    for idx, col in enumerate(("r", "g", "b")):
        hist_ch = cv2.calcHist([img_rgb], [idx], None, [256], [0, 256])
        axes[2].plot(hist_ch, color=col, label=col.upper())
    axes[2].set_xlim([0, 256])
    axes[2].set_title(f"{title_prefix}RGB Channels Histogram")
    axes[2].set_xlabel("Pixel Value (0-255)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def evaluate_quality(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size

    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    dark_ratio = float(np.sum(gray < 50) / total_pixels)
    bright_ratio = float(np.sum(gray > 205) / total_pixels)
    dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))

    issues = []
    if mean_brightness < 65 or dark_ratio > 0.50:
        issues.append("underexposed (too dark)")
    if mean_brightness > 190 or bright_ratio > 0.50:
        issues.append("overexposed (too bright)")
    if std_contrast < 35 or dynamic_range < 90:
        issues.append("low contrast (narrow tonal range)")

    return {
        "is_good": len(issues) == 0,
        "issues": issues,
        "mean_brightness": round(mean_brightness, 2),
        "std_contrast": round(std_contrast, 2),
        "dark_pct": round(dark_ratio * 100, 1),
        "bright_pct": round(bright_ratio * 100, 1),
        "dynamic_range": round(dynamic_range, 1),
    }


url = "https://upload.wikimedia.org/wikipedia/commons/5/50/Shieldgreen%2C_a_house_in_the_forest_-_geograph.org.uk_-_6848009.jpg"
req = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"}
)
with urllib.request.urlopen(req) as response:
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)

img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise ValueError("Failed to decode image from URL!")

report = evaluate_quality(img_bgr)
print("=== IMAGE QUALITY REPORT ===")
print(f"Mean brightness: {report['mean_brightness']} / 255")
print(f"Contrast (std dev): {report['std_contrast']}")
print(f"Dynamic range (P5-P95): {report['dynamic_range']}")
print(f"Dark pixels (<50): {report['dark_pct']}% | Bright pixels (>205): {report['bright_pct']}%")

if report["is_good"]:
    print("Verdict: Good exposure and contrast.")
else:
    print("Verdict: Defective image ->", ", ".join(report["issues"]))

plot_histograms(img_bgr)