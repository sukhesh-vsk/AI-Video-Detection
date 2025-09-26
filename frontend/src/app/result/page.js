"use client";

import { useEffect, useState } from "react";

export default function ResultPage() {
  const [result, setResult] = useState(null);

  useEffect(() => {
    const fetchResult = async () => {
      const res = await fetch("http://localhost:5000/result");
      const data = await res.json();
      setResult(data);
    };
    fetchResult();
  }, []);

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
        <p className="text-lg animate-pulse">Fetching results…</p>
      </div>
    );
  }

  const getConfidenceColor = (conf) => {
    if (conf < 40) return "text-green-400";      // mostly real
    if (conf < 60) return "text-yellow-400";     // moderate
    if (conf < 80) return "text-orange-500";     // leaning AI
    return "text-red-500";                        // mostly AI
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-start bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 p-6">
      <div className="w-full max-w-4xl bg-gray-900 rounded-3xl shadow-2xl p-8 mt-8">
        <h2 className="text-4xl font-bold text-center text-blue-400 mb-4">Analysis Result</h2>

        <div className="flex flex-col md:flex-row items-center justify-around gap-6 mt-6">
            {/* Stats */}
            <div className="bg-gray-800 p-6 rounded-xl shadow-lg flex flex-col gap-3 w-full md:w-1/3">
            <p className={`text-lg ${result.label === "AI Generated" ? "text-red-500" : "text-green-400"}`}>
              <strong className="text-blue-400">Label:</strong> {result.label}
            </p>
            <p className={`text-lg ${getConfidenceColor(result.confidence)}`}>
              <strong className="text-blue-400">Confidence:</strong> {result.confidence}%
            </p>
            <p className="text-lg text-white">
              <strong className="text-blue-400">AI Frames:</strong> {result.ai_count}
            </p>
            <p className="text-lg text-white">
              <strong className="text-blue-400">Real Frames:</strong> {result.real_count}
            </p>
          </div>

          {/* Video */}
          <div className="w-full md:w-1/2">
            <video
              src={`http://localhost:5000${result.video_url}`}
              controls
              className="rounded-xl shadow-lg w-full"
              type="video/mp4"
            />
          </div>
        </div>

        {/* Plot Image */}
        <div className="mt-6 w-full flex justify-center">
          <img
            src={`http://localhost:5000${result.plot_image}`}
            alt="Result Plot"
            className="rounded-xl shadow-lg w-full md:w-2/3"
          />
        </div>
      </div>
    </div>
  );
}
