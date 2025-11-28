"""
CNN-based Deepfake Detector using DeepSafe's pre-trained weights
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys

# Add DeepSafe path for imports
sys.path.insert(0, "D:/DeepSafe/models/cnndetection_image")


class CNNDeepfakeDetector:
    """CNN-based deepfake detector using DeepSafe weights"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 224
        self.model = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self._load_cnn_model()
    
    def _load_cnn_model(self):
        """Load CNN detection model from DeepSafe"""
        try:
            weights_path = "D:/DeepSafe/models/cnndetection_image/weights"
            
            # Find .pth file
            pth_files = [f for f in os.listdir(weights_path) if f.endswith('.pth')]
            
            if not pth_files:
                print("❌ No .pth file found in weights folder")
                return False
            
            weight_file = os.path.join(weights_path, pth_files[0])
            print(f"Found weights: {weight_file}")
            print(f"File size: {os.path.getsize(weight_file) / (1024*1024):.1f} MB")
            
            # Try to import the network architecture
            try:
                from networks.resnet import resnet50
                model = resnet50(num_classes=1)
            except:
                # Fallback to torchvision ResNet50
                from torchvision.models import resnet50
                model = resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, 1)
            
            # Load weights
            checkpoint = torch.load(weight_file, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            self.model = model
            print("✅ CNN model loaded successfully!")
            print(f"   Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading CNN model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_image(self, image):
        """Predict if image is deepfake"""
        if self.model is None:
            return {
                'deepfake_score': 0.5,
                'is_deepfake': False,
                'confidence': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
                image = Image.fromarray(image.astype('uint8'), 'RGB')
            
            # Preprocess
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
                prediction = torch.sigmoid(output).item()
            
            prediction = np.clip(prediction, 0.0, 1.0)
            
            return {
                'deepfake_score': float(prediction),
                'is_deepfake': bool(prediction > 0.5),
                'confidence': float(abs(prediction - 0.5) * 2)
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'deepfake_score': 0.5,
                'is_deepfake': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_video(self, frames, max_frames=30):
        """Predict video deepfake"""
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        predictions = []
        for frame in frames:
            result = self.predict_image(frame)
            predictions.append(result['deepfake_score'])
        
        if not predictions:
            return {
                'deepfake_score': 0.5,
                'is_deepfake': False,
                'confidence': 0.0,
                'frames_analyzed': 0
            }
        
        mean_pred = np.mean(predictions)
        return {
            'deepfake_score': float(mean_pred),
            'is_deepfake': bool(mean_pred > 0.5),
            'confidence': float(abs(mean_pred - 0.5) * 2),
            'frames_analyzed': len(predictions)
        }


# Alias for compatibility
DeepfakeDetector = CNNDeepfakeDetector
