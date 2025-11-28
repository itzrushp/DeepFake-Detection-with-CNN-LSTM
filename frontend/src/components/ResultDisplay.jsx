import React from 'react';

const ResultDisplay = ({ result, onReset }) => {
  // Debug: Log the entire result object to console
  console.log('Received result object:', result);
  console.log('is_deepfake:', result.is_deepfake);
  console.log('confidence_score (raw):', result.confidence_score);
  
  const isDeepfake = result.is_deepfake;
  const deepfakeScoreRaw = result.confidence_score;
  const deepfakeScore = deepfakeScoreRaw * 100;
  
  // Calculate authentic score (inverse of deepfake score)
  const authenticScore = 100 - deepfakeScore;
  
  // Determine which score to show based on verdict
  const displayScore = isDeepfake ? deepfakeScore : authenticScore;
  const displayPercent = displayScore.toFixed(1);
  
  // Additional debug logs
  console.log('deepfakeScore (%):', deepfakeScore);
  console.log('authenticScore (%):', authenticScore);
  console.log('displayScore (%):', displayScore);
  console.log('isDeepfake:', isDeepfake);
  console.log('Final verdict text:', isDeepfake 
    ? `${displayPercent}% chance this is a DEEPFAKE`
    : `${displayPercent}% chance this is AUTHENTIC`
  );

  return (
    <div className="space-y-6">
      {/* Main Result Card */}
      <div className={`rounded-2xl p-8 border-2 ${
        isDeepfake 
          ? 'bg-red-500/10 border-red-500' 
          : 'bg-green-500/10 border-green-500'
      }`}>
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-white mb-2">Analysis Complete</h2>
            <p className="text-gray-400">{new Date().toLocaleString()}</p>
          </div>
          <div className={`text-6xl ${isDeepfake ? 'text-red-500' : 'text-green-500'}`}>
            {isDeepfake ? '⚠️' : '✅'}
          </div>
        </div>

        {/* Confidence Score */}
        <div className="mb-6">
          <div className="flex justify-between mb-2">
            <span className="text-white font-semibold">Confidence Level</span>
            <span className="text-2xl font-bold text-white">{displayPercent}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-4">
            <div
              className={`h-4 rounded-full transition-all ${
                isDeepfake ? 'bg-red-500' : 'bg-green-500'
              }`}
              style={{ width: `${displayPercent}%` }}
            />
          </div>
        </div>

        {/* Verdict */}
        <div className={`p-4 rounded-lg ${
          isDeepfake ? 'bg-red-500/20' : 'bg-green-500/20'
        }`}>
          <p className={`text-xl font-bold ${
            isDeepfake ? 'text-red-200' : 'text-green-200'
          }`}>
            {isDeepfake 
              ? `${displayPercent}% chance this is a DEEPFAKE`
              : `${displayPercent}% chance this is AUTHENTIC`
            }
          </p>
          
          {/* Show both scores for clarity */}
          <p className="text-sm mt-2 opacity-75">
            {isDeepfake 
              ? `(${authenticScore.toFixed(1)}% authentic)`
              : `(${deepfakeScore.toFixed(1)}% deepfake)`
            }
          </p>
        </div>
      </div>

      {/* Detected Anomalies */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">
          🔍 Analysis Details
        </h3>
        <ul className="space-y-3">
          {result.explanations?.map((exp, i) => (
            <li key={i} className="flex items-start gap-3 text-gray-300">
              <span className="text-blue-500 mt-1">•</span>
              <span>{exp}</span>
            </li>
          )) || <li className="text-gray-400">No explanations available</li>}
        </ul>
      </div>

      {/* Metadata (if available) */}
      {result.metadata && (
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-4">Technical Details</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-400">Deepfake Score</p>
              <p className="text-white font-semibold">{deepfakeScore.toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-gray-400">Authentic Score</p>
              <p className="text-white font-semibold">{authenticScore.toFixed(2)}%</p>
            </div>
            {result.metadata.frames_analyzed && (
              <div>
                <p className="text-gray-400">Frames Analyzed</p>
                <p className="text-white font-semibold">{result.metadata.frames_analyzed}</p>
              </div>
            )}
            {result.metadata.image_shape && (
              <div>
                <p className="text-gray-400">Image Shape</p>
                <p className="text-white font-semibold">{result.metadata.image_shape.join('x')}</p>
              </div>
            )}
            {result.metadata.total_frames && (
              <div>
                <p className="text-gray-400">Total Frames</p>
                <p className="text-white font-semibold">{result.metadata.total_frames}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={onReset}
          className="flex-1 px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-lg transition-all"
        >
          Analyze Another
        </button>
        <button
          onClick={() => {
            const data = JSON.stringify(result, null, 2);
            console.log('Downloading report data:', data); // Debug log for download
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `deepfake-analysis-${Date.now()}.json`;
            a.click();
          }}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all"
        >
          Download Report
        </button>
      </div>
    </div>
  );
};

export default ResultDisplay;