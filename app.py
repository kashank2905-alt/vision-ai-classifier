from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)

# Yahan hum duniya ka sab se behtareen Zero-Shot Image model (CLIP) load kar rahe hain
print("Loading Advanced Context AI (OpenAI CLIP)... (This may take a minute to download)")
image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Hum AI ko apni marzi ki categories de rahe hain jin mein se usay chuna hai
        candidate_labels = [
            "liquid wastewater, flowing water, or fluid",
            "solid waste, plastic, metal, or solid object",
            "gas, smoke, exhaust, or vapor"
        ]
        
        # AI tasveer ko in 3 categories mein judge karega
        results = image_classifier(image, candidate_labels=candidate_labels)
        
        # Sab se high probability wala result nikalna
        best_match = results[0]
        predicted_text = best_match['label']
        confidence_score = round(best_match['score'] * 100, 1)
        
        # UI par dikhane ke liye khubsurat formatting
        if "liquid" in predicted_text:
            final_state = "Liquid Wastewater 💧"
        elif "gas" in predicted_text:
            final_state = "Gas / Smoke 💨"
        else:
            final_state = "Solid Waste 🪨"
            
        return jsonify({
            "label": final_state,
            "confidence": f"{confidence_score}%",
            "message": "Context perfectly understood by Advanced AI!"
        })
    except Exception as e:
        return jsonify({"error": "Error processing image."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8501)