"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function UploadPage() {
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoURL, setVideoURL] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setVideoURL(URL.createObjectURL(file)); // Create a temporary URL for preview
    } else {
      setSelectedFile(null);
      setVideoURL(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    setLoading(true);
    setLogs([]);
    setProgress(0);

    // Upload video
    await fetch("http://localhost:5000/analyze", {
      method: "POST",
      body: formData,
    });

    // Start listening to logs (SSE)
    const eventSource = new EventSource("http://localhost:5000/logs");
    eventSource.onmessage = (event) => {
      setLogs((prev) => [...prev, event.data]);
      if (event.data.includes("Complete")) {
        eventSource.close();
        router.push("/result"); // Navigate to Result page
      } else {
        setProgress((p) => Math.min(p + 5, 100));
      }
    };
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 text-white p-6">
      <div className="w-full max-w-3xl bg-gray-900 rounded-3xl shadow-2xl p-8">
        <h1 className="text-4xl font-bold text-center text-blue-400">AI Video Detection</h1>
        <p className="text-gray-300 text-center mt-2">Upload a video to detect if it's AI-generated</p>

        <form onSubmit={handleUpload} className="mt-6 flex flex-col items-center gap-4">
        <label className="relative w-full cursor-pointer">
          <input
            type="file"
            name="video"
            accept="video/*"
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            onChange={handleFileChange}
          />
          <div className="w-full px-6 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-purple-600 hover:to-blue-600 text-white rounded-lg shadow-lg text-center font-semibold transition-all duration-300">
            Browse Video
          </div>
        </label>

        {selectedFile && (
          <div className="mt-4 text-center">
            <p className="text-white font-medium">Selected File: {selectedFile.name}</p>
            <video
              src={videoURL}
              controls
              className="mt-2 w-full max-w-md rounded-lg shadow-lg"
            />
          </div>
        )}

          <button
            type="submit"
            className="w-1/2 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-purple-500 hover:to-blue-500 text-white font-semibold mt-8 py-2 px-6 rounded-lg shadow-lg transition-all duration-300"
          >
            Analyze
          </button>
        </form>

        {loading && (
          <div className="mt-8 space-y-6">
            {/* Spinner */}
            <div className="flex items-center justify-center gap-4">
              <div className="w-8 h-8 border-4 border-t-blue-500 border-gray-700 rounded-full animate-spin"></div>
              <span className="text-lg text-gray-200">Analyzing video…</span>
            </div>

            {/* Progress Bar */}
            <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden mt-4">
              <div
                className="h-4 bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>

            {/* Logs */}
            <div className="bg-gray-800 p-4 rounded-lg h-48 overflow-y-scroll font-mono text-green-400">
              {logs.length === 0 ? (
                <p>Waiting for updates...</p>
              ) : (
                logs.map((log, idx) => <div key={idx}>{log}</div>)
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
