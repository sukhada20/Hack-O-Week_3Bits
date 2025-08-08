import React, { useState } from 'react';

function EmotionUpload() {
  const [file, setFile] = useState(null);
  const [emotion, setEmotion] = useState("");

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    setEmotion(data.emotion);
  };

  return (
    <div>
      <input type="file" onChange={e => setFile(e.target.files[0])} />
      <button onClick={handleUpload}>Detect Emotion</button>
      <div>Detected Emotion: {emotion}</div>
    </div>
  );
}

export default EmotionUpload;
