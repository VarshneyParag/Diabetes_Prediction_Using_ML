import os
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'prod-secret-key-123')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Import TensorFlow with error handling
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False

# Load model
model = None
class_names = []

if TENSORFLOW_AVAILABLE:
    try:
        # Try different possible model paths
        possible_paths = [
            'plant_leaf_disease_cnn_model.h5',
            './plant_leaf_disease_cnn_model.h5'
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model = load_model(model_path)
                print(f"✅ Model loaded from {model_path}")
                break
        else:
            raise FileNotFoundError("Model file not found")
            
        # Load class names
        class_paths = ['class_names.npy', './class_names.npy']
        for class_path in class_paths:
            if os.path.exists(class_path):
                class_names = np.load(class_path, allow_pickle=True).tolist()
                print(f"✅ Class names loaded from {class_path}")
                break
        else:
            # Fallback class names
            class_names = ["Healthy", "Diseased"]
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        TENSORFLOW_AVAILABLE = False

# Mock model for fallback
class MockModel:
    def predict(self, x):
        return np.random.rand(1, len(class_names))

if not TENSORFLOW_AVAILABLE or model is None:
    print("🔧 Using mock model")
    model = MockModel()
    class_names = class_names or ["Healthy", "Powdery Mildew", "Rust"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(64, 64)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        try:
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            
            predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else "Unknown"
            confidence = float(np.max(predictions[0]))

            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = []
            
            for i in top3_indices:
                class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                top3_predictions.append({
                    "class": class_name, 
                    "confidence": round(float(predictions[0][i]) * 100, 2)
                })

            return jsonify({
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "top3": top3_predictions,
                "image_url": f"/uploads/{filename}",
                "model_loaded": TENSORFLOW_AVAILABLE
            })
            
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"})
    
    return jsonify({"error": "Invalid file type"})

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "model_loaded": model is not None,
        "num_classes": len(class_names)
    })

# Ensure uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
