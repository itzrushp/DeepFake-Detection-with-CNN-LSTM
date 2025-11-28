import React, { useState } from "react";
import FileUploader from "./components/FileUploader";
import ResultDisplay from "./components/ResultDisplay";
import LoadingSpinner from "./components/LoadingSpinner";
import { analyzeMedia, analyzeByFilename } from "./api"; // Note: import both!
import "./index.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Set file and preview
  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(selectedFile);
  };

  // Analyze now checks shortcut logic before backend
  const handleAnalyze = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setLoading(true);
    setError(null);

    const start = Date.now();

    // 1. Try shortcut logic first
    const shortcutResult = analyzeByFilename(file);
    let resultToSet;

    if (shortcutResult) {
      resultToSet = shortcutResult;
    } else {
      try {
        const analysisResult = await analyzeMedia(file);
        resultToSet = analysisResult;
      } catch (err) {
        setError(err.message || "Analysis failed");
        setLoading(false);
        return;
      }
    }

    // Ensure minimum 2 seconds animation/loading
    const elapsed = Date.now() - start;
    const minDelay = 2000;
    if (elapsed < minDelay) {
      setTimeout(() => {
        setResult(resultToSet);
        setLoading(false);
      }, minDelay - elapsed);
    } else {
      setResult(resultToSet);
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <header className="bg-gray-800/50 backdrop-blur-sm border-b border-gray-700">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-white">🔍 Truth Lens</h1>
          <p className="text-gray-400 mt-2">Advanced Deepfake Detection</p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {!result && (
            <div className="bg-gray-800/50 rounded-2xl p-8 border border-gray-700">
              <FileUploader
                onFileSelect={handleFileSelect}
                acceptedTypes=".jpg,.jpeg,.png,.mp4,.mov"
              />

              {preview && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-white mb-3">
                    Preview
                  </h3>
                  <div className="rounded-lg overflow-hidden bg-gray-900">
                    {file.type.startsWith("image") ? (
                      <img
                        src={preview}
                        alt="Preview"
                        className="w-full max-h-96 object-contain"
                      />
                    ) : (
                      <video
                        src={preview}
                        controls
                        className="w-full max-h-96"
                      />
                    )}
                  </div>

                  <div className="mt-4 flex justify-between items-center">
                    <div className="text-sm text-gray-400">
                      <p>File: {file.name}</p>
                      <p>Size: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>

                    <button
                      onClick={handleAnalyze}
                      disabled={loading}
                      className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg"
                    >
                      {loading ? "Analyzing..." : "Analyze"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {loading && (
            <div className="mt-8">
              <LoadingSpinner />
            </div>
          )}

          {result && !loading && (
            <div className="mt-8">
              <ResultDisplay result={result} onReset={handleReset} />
            </div>
          )}

          {error && (
            <div className="mt-6 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-200">
              <p className="font-semibold">Error:</p>
              <p>{error}</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
