import React from 'react';

const LoadingSpinner = () => {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="w-24 h-24 border-4 border-gray-700 border-t-blue-500 rounded-full animate-spin"></div>
      <div className="mt-6 text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Analyzing Media...</h3>
        <p className="text-gray-400 text-sm">This may take a few moments</p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
