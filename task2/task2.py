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


def fix_quality(img_bgr, report):
    # convert to LAB to enhance luminance without distorting colors
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # gamma correction for underexposed or overexposed images
    if "underexposed (too dark)" in report["issues"]:
        gamma = 0.65
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        l_channel = cv2.LUT(l_channel, lut)
    elif "overexposed (too bright)" in report["issues"]:
        gamma = 1.5
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        l_channel = cv2.LUT(l_channel, lut)

    # local contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    merged_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


# 1. load remote image from URL
url = "https://upload.wikimedia.org/wikipedia/commons/d/df/Underexposed_%E2%80%93_Pulau_Ubin_%284440132168%29.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original"
req = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"}
)
with urllib.request.urlopen(req) as response:
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)

img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise ValueError("Failed to decode image from URL!")

# 2 & 3. evaluate image quality based on histogram statistics
report = evaluate_quality(img_bgr)
print("=== IMAGE QUALITY REPORT ===")
print(f"Mean brightness: {report['mean_brightness']} / 255")
print(f"Contrast (std dev): {report['std_contrast']}")
print(f"Dynamic range (P5-P95): {report['dynamic_range']}")
print(f"Dark pixels (<50): {report['dark_pct']}% | Bright pixels (>205): {report['bright_pct']}%")

if report["is_good"]:
    print("Verdict: Good exposure and contrast.")
    plot_histograms(img_bgr, title_prefix="[Original - OK] ")
else:
    print("Verdict: Defective image ->", ", ".join(report["issues"]))
    plot_histograms(img_bgr, title_prefix="[Before Fix] ")

    # bonus: automatic image quality enhancement
    img_fixed = fix_quality(img_bgr, report)
    fixed_report = evaluate_quality(img_fixed)
    print("\n=== AFTER AUTO-ENHANCEMENT ===")
    print(f"New brightness: {fixed_report['mean_brightness']} | New contrast: {fixed_report['std_contrast']}")
    plot_histograms(img_fixed, title_prefix="[After Fix] ")