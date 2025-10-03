import os
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class names for demonstration
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", 
    "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew",
    "Cherry___healthy", "Corn___Cercospora_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot",
    "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper_bell___Bacterial_spot", "Pepper_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
    "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

print("✅ Using lightweight version without TensorFlow")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_leaf_health(img_path):
    """Simple color-based analysis for demonstration"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return "Unknown", 50.0
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Calculate green color percentage (healthy indicator)
        green_lower = np.array([36, 25, 25])
        green_upper = np.array([86, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_percentage = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])
        
        # Calculate yellow/brown color percentage (disease indicator)
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_percentage = np.sum(yellow_mask > 0) / (img.shape[0] * img.shape[1])
        
        # Simple health assessment
        if green_percentage > 0.6 and yellow_percentage < 0.1:
            # Likely healthy
            healthy_classes = [cls for cls in class_names if 'healthy' in cls.lower()]
            main_class = random.choice(healthy_classes) if healthy_classes else class_names[3]  # Apple_healthy
            confidence = min(95.0, 70.0 + (green_percentage * 25))
        else:
            # Likely diseased
            disease_classes = [cls for cls in class_names if 'healthy' not in cls.lower()]
            main_class = random.choice(disease_classes) if disease_classes else class_names[0]  # Apple_scab
            confidence = min(90.0, 60.0 + (yellow_percentage * 30))
            
        return main_class, confidence
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return "Unknown", 50.0

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
            # Use simple color analysis
            predicted_class, confidence = analyze_leaf_health(filepath)
            
            # Generate top 3 predictions
            top3_predictions = []
            top3_predictions.append({
                "class": predicted_class, 
                "confidence": round(confidence, 2)
            })
            
            # Add 2 more random predictions
            other_classes = [cls for cls in class_names if cls != predicted_class]
            for _ in range(2):
                if other_classes:
                    cls = random.choice(other_classes)
                    other_classes.remove(cls)
                    top3_predictions.append({
                        "class": cls, 
                        "confidence": round(random.uniform(10, confidence - 5), 2)
                    })

            result = {
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "top3": top3_predictions,
                "image_url": f"/uploads/{filename}",
                "demo_mode": True,
                "message": "Using color-based analysis (Demo Mode)"
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"})
    else:
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, or JPEG."})

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "mode": "lightweight",
        "num_classes": len(class_names)
    })

# Create uploads directory
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
