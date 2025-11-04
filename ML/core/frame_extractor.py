import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_frames_from_video(video_path, output_dir, frame_skip=20):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)

    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        idx += 1

    cap.release()
    print(f"{saved} frames saved → {output_dir}")


def build_dataset(src_root="../Datasets", dest_root="../frames_dataset"):
    src_root = os.path.join(BASE_DIR, src_root)
    dest_root = os.path.join(BASE_DIR, dest_root)

    categories = ["ai", "real"]
    print("Current Working Directory:", os.getcwd())

    for cat in categories:
        src_dir = os.path.join(src_root, cat)
        dest_dir = os.path.join(dest_root, cat)

        print("Src: ", src_dir)
        print("Dest: ", dest_dir)

        os.makedirs(dest_dir, exist_ok=True)

        for file in os.listdir(src_dir):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                print(f"Extracting frames from {file}...")
                print(os.path.join(dest_dir, os.path.splitext(file)[0]))
                extract_frames_from_video(
                    os.path.join(src_dir, file),
                    dest_dir
                )

if __name__ == "__main__":
    build_dataset()
