from flask import Flask, render_template, request, redirect, url_for, Response, send_file
import os, uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from threading import Thread
import queue
from flask_cors import CORS

from core.analyzer import analyze_video

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SERVER_NAME"] = "localhost:5000" 
CORS(app)

# --- Globals ---
log_queue = queue.Queue()
last_result = {}
processing = False

print("Before")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    global processing
    if "video" not in request.files:
        return {"error": "No video uploaded"}, 400

    file = request.files["video"]
    if file.filename == "":
        return {"error": "No filename"}, 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    def run_analysis():
        global last_result, processing
        with app.app_context():
            processing = True
            for log in analyze_video(video_path):
                if isinstance(log, str):
                    log_queue.put(log)  # Send log
                elif isinstance(log, dict):  # Final result
                    print(f"Log: {log}")
                    print(f"Plot Log: {log["plot"]}")
                    last_result = {
                        "label": log["final_label"],
                        "confidence": round(log["confidence"] * 100, 2),
                        "ai_count": log["ai_count"],
                        "real_count": log["real_count"],
                        "video_url": f"/static/uploads/{filename}",
                        "plot_image": log["plot"]
                    }
            processing = False
            log_queue.put("=== Processing Complete ===")

    Thread(target=run_analysis).start()
    return {"status": "processing"}


@app.route("/result", methods=["GET"])
def get_result():
    global last_result, processing
    if processing:
        return {"status": "processing"}
    if last_result:
        return last_result
    return {"status": "no result yet"}

@app.route("/plot/<filename>")
def get_plot(filename):
    return send_file(os.path.join(BASE_DIR, "outputs", filename))

@app.route("/plot")
def generate_plot():
    global last_result

    if not last_result:
        return "No result", 400

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(last_result["frames"], last_result["residuals"], c='cyan')
    axes[1].plot(last_result["frames"], last_result["wavelets"], c='yellow')
    axes[2].plot(last_result["frames"], last_result["fft_ratios"], c='white')

    plt.tight_layout()

    output_dir = os.path.join(BASE_DIR, "static/results")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, "result_plot.png")
    plt.savefig(plot_path)
    plt.close()

    return send_file(plot_path, mimetype="image/png")

@app.route("/logs")
def logs():
    def generate():
        while processing or not log_queue.empty():
            try:
                msg = log_queue.get(timeout=1)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                continue
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    print("Inside")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print("After")
    app.run(debug=True, threaded=True)
