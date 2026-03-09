import cv2
import numpy as np
import pywt
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uuid
from .utils import noise_features
from .vit_classifier import classify_frame_vit
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(BASE_DIR, "../outputs")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Process Video & Visualize
# ---------------------------
def analyze_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    results = []
    features_list = []
    frame_idx_list = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # ViT Prediction (with score support)
            label = classify_frame_vit(frame)

            # Logging for SSE
            yield f"Frame {frame_idx}: {label}"

            results.append(label)
            feats = noise_features(frame)
            features_list.append(feats)
            frame_idx_list.append(frame_idx)

        frame_idx += 1

    cap.release()

    # ✅ Dynamic label handling
    # ai_labels = [lbl for lbl in set(results) if "ai" in lbl.lower()]
    # real_labels = [lbl for lbl in set(results) if "real" in lbl.lower()]

    # ai_count = sum(lbl in ai_labels for lbl in results)
    # real_count = sum(lbl in real_labels for lbl in results)

    # confidence = ai_count / len(results) if results else 0
    # final_label = ai_labels[0].upper() if confidence > 0.5 else real_labels[0].upper()
    ai_count = results.count("AI")
    real_count = results.count("Real")

    final_label = "AI" if ai_count > real_count else "Real"
    confidence = ai_count / len(results)

    # ✅ Visualization Data
    residuals = [f["residual_variance"] for f in features_list]
    wavelets = [f["wavelet_energy"] for f in features_list]
    fft_ratios = [f["fft_high_ratio"] for f in features_list]
    colors = ['red' if lbl in results else 'green' for lbl in results]

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

    output_dir = os.path.join(BASE_DIR, "../outputs")
    os.makedirs(output_dir, exist_ok=True)

    plot_filename = f"{uuid.uuid4().hex}.png"
    # fig_path = os.path.join(output_dir, plot_filename)
    fig_path = os.path.join("/home/vsk/Code/AI-Video-Detection/frontend/public/plots", plot_filename)

    plt.savefig(fig_path)  # ✅ SAFE now
    time.sleep(2)
    plt.close(fig)         # ✅ Required to prevent crash

    result_dict = {
        "final_label": final_label,
        "confidence": confidence,
        "ai_count": ai_count,
        "real_count": real_count,
        "plot": plot_filename
    }

    yield result_dict

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

