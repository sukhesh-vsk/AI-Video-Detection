from flask import Flask, render_template, request, redirect, url_for, Response
import os, uuid, matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from threading import Thread
import queue
from flask_cors import CORS

from core.analyzer import analyze_video

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SERVER_NAME"] = "localhost:5000" 
CORS(app)

# --- Globals ---
log_queue = queue.Queue()
last_result = {}
processing = False

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
                    plot_filename = f"{uuid.uuid4().hex}.png"
                    plot_path = os.path.join(RESULT_FOLDER, plot_filename)
                    log["plot"].savefig(plot_path)
                    plt.close(log["plot"])
                    last_result = {
                        "label": log["final_label"],
                        "confidence": round(log["confidence"] * 100, 2),
                        "ai_count": log["ai_count"],
                        "real_count": log["real_count"],
                        "video_url": f"/static/uploads/{filename}",
                        "plot_image": f"/static/results/{plot_filename}"
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
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True, threaded=True)
