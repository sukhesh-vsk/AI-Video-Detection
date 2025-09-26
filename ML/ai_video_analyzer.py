import cv2
import numpy as np
import pywt
import os
import matplotlib.pyplot as plt

# ---------------------------
# Feature Extraction Function
# ---------------------------
def noise_features(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 1. Residual Variance (noise measure)
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    residual = cv2.subtract(img, denoised)
    res_var = np.var(residual) / (np.var(img) + 1e-6)

    # 2. Wavelet Detail Energy
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    wavelet_energy = (np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)) / img.size

    # 3. FFT High-Frequency Ratio
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    h, w = img.shape
    crow, ccol = h // 2, w // 2
    radius = min(h, w) // 8  # low-frequency radius

    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    low_energy = np.sum(magnitude_spectrum * mask)
    total_energy = np.sum(magnitude_spectrum)
    high_energy = total_energy - low_energy

    fft_ratio = high_energy / (low_energy + 1e-6)

    return {
        "residual_variance": float(res_var),
        "wavelet_energy": float(wavelet_energy),
        "fft_high_ratio": float(fft_ratio)
    }

# ---------------------------
# Rule-Based Frame Classifier
# ---------------------------
def classify_frame(features):
    # Thresholds are empirical; tune based on real data
    if features["residual_variance"] < 0.005 and features["wavelet_energy"] < 50:
        return "AI"
    elif features["fft_high_ratio"] > 10:
        return "AI"
    else:
        return "Real"

# ---------------------------
# Process Video & Visualize
# ---------------------------
def analyze_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)

    results = []
    features_list = []
    frame_idx_list = []

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            feats = noise_features(frame)
            label = classify_frame(feats)

            results.append(label)
            features_list.append(feats)
            frame_idx_list.append(frame_idx)

            print(f"Frame {frame_idx}: {label}")
            yield f"Frame {frame_idx}: {label}"  # log output

        frame_idx += 1

    cap.release()

    ai_count = results.count("AI")
    real_count = results.count("Real")
    confidence = ai_count / len(results) if results else 0
    final_label = "AI Generated" if confidence > 0.6 else "Real"

    residuals = [f["residual_variance"] for f in features_list]
    wavelets = [f["wavelet_energy"] for f in features_list]
    fft_ratios = [f["fft_high_ratio"] for f in features_list]
    colors = ['red' if lbl == "AI" else 'green' for lbl in results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].scatter(frame_idx_list, residuals, c=colors, s=40)
    axes[0].set_ylabel("Residual Variance")
    axes[0].set_title("Residual Noise per Frame")

    axes[1].scatter(frame_idx_list, wavelets, c=colors, s=40)
    axes[1].set_ylabel("Wavelet Energy")
    axes[1].set_title("Wavelet Detail Energy per Frame")

    axes[2].scatter(frame_idx_list, fft_ratios, c=colors, s=40)
    axes[2].set_ylabel("FFT High-Freq Ratio")
    axes[2].set_xlabel("Frame Index")
    axes[2].set_title("High-Frequency Ratio per Frame")

    axes[0].scatter([], [], c='red', label='AI')
    axes[0].scatter([], [], c='green', label='Real')
    axes[0].legend(loc='upper right')

    plt.tight_layout()

    print({
        "final_label": final_label,
        "confidence": confidence,
        "ai_count": ai_count,
        "real_count": real_count,
        "plot": fig
    })

    yield {
        "final_label": final_label,
        "confidence": confidence,
        "ai_count": ai_count,
        "real_count": real_count,
        "plot": fig
    }
    
# ---------------------------
# Run on a Video
# ---------------------------
if __name__ == "__main__":
    video_path = "../Datasets/ai/ai (1).mp4"
    final_label, confidence, ai_count, real_count = analyze_video(video_path)

    print("\n=== Final Decision ===")
    print(f"Video: {video_path}")
    print(f"Result: {final_label}")
    print(f"AI Frames: {ai_count}, Real Frames: {real_count}")
    print(f"Confidence: {confidence*100:.2f}%")


################################################
################################################
################################################

# import cv2
# import numpy as np
# import pywt
# import os

# # ---------------------------
# # Feature Extraction Function
# # ---------------------------
# def noise_features(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

#     # 1. Residual Variance (noise measure)
#     denoised = cv2.GaussianBlur(img, (3, 3), 0)
#     residual = cv2.subtract(img, denoised)
#     res_var = np.var(residual) / (np.var(img) + 1e-6)

#     # 2. Wavelet Detail Energy
#     coeffs2 = pywt.dwt2(img, 'haar')
#     LL, (LH, HL, HH) = coeffs2
#     wavelet_energy = (np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)) / img.size

#     # 3. FFT High-Frequency Ratio
#     f = np.fft.fft2(img)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = np.abs(fshift)

#     h, w = img.shape
#     crow, ccol = h // 2, w // 2
#     radius = min(h, w) // 8  # low-frequency radius

#     mask = np.zeros((h, w), np.uint8)
#     cv2.circle(mask, (ccol, crow), radius, 1, -1)

#     low_energy = np.sum(magnitude_spectrum * mask)
#     total_energy = np.sum(magnitude_spectrum)
#     high_energy = total_energy - low_energy

#     fft_ratio = high_energy / (low_energy + 1e-6)

#     return {
#         "residual_variance": float(res_var),
#         "wavelet_energy": float(wavelet_energy),
#         "fft_high_ratio": float(fft_ratio)
#     }

# # ---------------------------
# # Rule-Based Frame Classifier
# # ---------------------------
# def classify_frame(features):
#     # Thresholds are empirical; tune based on real data
#     if features["residual_variance"] < 0.005 and features["wavelet_energy"] < 50:
#         return "AI"
#     elif features["fft_high_ratio"] > 10:
#         return "AI"
#     else:
#         return "Real"

# # ---------------------------
# # Process Video
# # ---------------------------
# def analyze_video(video_path, frame_skip=10):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     results = []
#     frame_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_idx % frame_skip == 0:  # sample frames
#             feats = noise_features(frame)
#             label = classify_frame(feats)
#             results.append(label)

#             print(f"Frame {frame_idx}: {feats} → {label}")

#         frame_idx += 1

#     cap.release()

#     # Aggregate results
#     ai_count = results.count("AI")
#     real_count = results.count("Real")
#     confidence = ai_count / len(results) if results else 0

#     final_label = "AI Generated" if confidence > 0.6 else "Real"
#     return final_label, confidence, ai_count, real_count

# # ---------------------------
# # Run on a Video
# # ---------------------------
# if __name__ == "__main__":
#     # video_path = "datasets/ai_video_2.mp4"
#     video_path = "../Datasets/ai/ai (1).mp4"
#     final_label, confidence, ai_count, real_count = analyze_video(video_path)

#     print("\n=== Final Decision ===")
#     print(f"Video: {video_path}")
#     print(f"Result: {final_label}")
#     print(f"AI Frames: {ai_count}, Real Frames: {real_count}")
#     print(f"Confidence: {confidence*100:.2f}%")


################################################
################################################
################################################

'''
import cv2
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import pywt

# ------------------------------
# 1. Extract Frames
# ------------------------------
def extract_frames(video_path, output_folder="frames", max_frames=20):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    saved = 0
    frames_list = []
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = f"{output_folder}/frame_{frame_no}.jpg"
        cv2.imwrite(frame_path, frame)
        frames_list.append(frame_path)
        saved += 1
        frame_no += 10  # skip some frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    cap.release()
    print(f"✅ Extracted {saved} frames into '{output_folder}'")
    return frames_list

# ------------------------------
# 2. Metadata Analysis (ffprobe)
# ------------------------------
def get_metadata(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_format", "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("📊 Video Metadata:\n", result.stdout)

def noise_features(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, 0).astype(np.float32)

    # ---------------------------
    # 1. Residual Variance
    # ---------------------------
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    residual = cv2.subtract(img, denoised)
    res_var = np.var(residual) / (np.var(img) + 1e-6)  # normalized

    # ---------------------------
    # 2. Wavelet Detail Energy
    # ---------------------------
    coeffs2 = pywt.dwt2(img, 'haar')   # single-level 2D wavelet
    LL, (LH, HL, HH) = coeffs2
    wavelet_energy = (np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)) / (img.size)

    # ---------------------------
    # 3. FFT High-Frequency Ratio
    # ---------------------------
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    h, w = img.shape
    crow, ccol = h//2, w//2
    radius = min(h, w) // 8   # low frequency radius

    # Create mask for low-frequencies
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    low_energy = np.sum(magnitude_spectrum * mask)
    total_energy = np.sum(magnitude_spectrum)
    high_energy = total_energy - low_energy

    fft_ratio = high_energy / (low_energy + 1e-6)

    # ---------------------------
    # Return all features
    # ---------------------------
    return {
        "residual_variance": float(res_var),
        "wavelet_energy": float(wavelet_energy),
        "fft_high_ratio": float(fft_ratio)
    }

# ------------------------------
# 3. Noise Residual + Variance
# ------------------------------
def noise_variance(image_path):
    img = cv2.imread(image_path, 0)
    denoised = cv2.GaussianBlur(img, (3,3), 0)
    residual = cv2.subtract(img, denoised)
    var = np.var(residual)   # variance of noise
    return var

# ------------------------------
# 4. Frequency Analysis (FFT High-Freq Energy)
# ------------------------------
def fft_energy(image_path):
    img = cv2.imread(image_path, 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # focus on high frequencies (corners of spectrum)
    h, w = magnitude.shape
    quarter = int(min(h, w) * 0.25)
    high_freq = magnitude[:quarter, :quarter].sum() + \
                magnitude[:quarter, -quarter:].sum() + \
                magnitude[-quarter:, :quarter].sum() + \
                magnitude[-quarter:, -quarter:].sum()
    return high_freq / (h * w)  # normalize

# ------------------------------
# 5. Rule-Based Classifier for a Frame
# ------------------------------
def classify_frame(image_path, noise_thresh=50, fft_thresh=5):
    noise_var = noise_variance(image_path)
    fft_val = fft_energy(image_path)

    print(f"Frame: {image_path}")
    print(f" - Noise Variance: {noise_var:.2f}")
    print(f" - FFT High-Freq Energy: {fft_val:.2f}")

    if noise_var < noise_thresh and fft_val > fft_thresh:
        print(" 🟥 Likely AI-generated")
        return 1
    else:
        print(" 🟩 Likely Real")
        return 0

# ------------------------------
# 6. Classify Whole Video
# ------------------------------
def classify_video(video_path, max_frames=20):
    frames = extract_frames(video_path, max_frames=max_frames)
    get_metadata(video_path)

    noise_vals, fft_vals = [], []
    results = []

    for f in frames:
        noice_feat = noise_features(f)
        noise_var = noise_variance(f)
        fft_val = fft_energy(f)
        noise_vals.append(noise_var)
        fft_vals.append(fft_val)
        print("Noice Features: ", noice_feat)
    print("FFT vals: ", fft_vals)
    print("Noice vals: ", noise_vals)
    # Adaptive thresholds: median values
    noise_thresh = np.median(noise_vals) * 0.8
    fft_thresh = np.median(fft_vals) * 1.2

    print(f"\nDynamic Noise Threshold: {noise_thresh:.2f}")
    print(f"Dynamic FFT Threshold: {fft_thresh:.2f}")

    ai_count, real_count = 0, 0
    for i, f in enumerate(frames):
        if noise_vals[i] < noise_thresh and fft_vals[i] > fft_thresh:
            print(f"Frame {i}: 🟥 AI")
            ai_count += 1
        else:
            print(f"Frame {i}: 🟩 Real")
            real_count += 1

    print("\n📊 Summary:")
    print(f"AI-like frames: {ai_count}")
    print(f"Real-like frames: {real_count}")

    if ai_count > real_count:
        print("🚨 Final Decision: Likely AI-Generated Video")
    else:
        print("✅ Final Decision: Likely Real Video")


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    # video_file = "datasets/real_video_1.mp4"   # 🔹 Change your video here
    video_file = "../Datasets/ai/ai (12).mp4"   # 🔹 Change your video here
    classify_video(video_file, max_frames=30)
'''
