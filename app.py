from flask import Flask, render_template, request, jsonify
import requests
import base64

app = Flask(__name__)

# APNA HUGGING FACE TOKEN YAHAN DAALEIN
HF_API_TOKEN = "YOUR_HUGGING_FACE_TOKEN_HERE" 
API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    image_bytes = file.read()
    
    try:
        candidate_labels = [
            "liquid wastewater, flowing water, or fluid",
            "solid waste, plastic, metal, or solid object",
            "gas, smoke, exhaust, or vapor"
        ]
        
        payload = {
            "inputs": base64.b64encode(image_bytes).decode('utf-8'),
            "parameters": {"candidate_labels": candidate_labels}
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        results = response.json()
        
        # 🔥 SAFE CHECK: Agar Hugging Face se koi error ya loading status aaye
        if isinstance(results, dict) and "error" in results:
            return jsonify({
                "error": "AI Brain is warming up! Please wait 10-15 seconds and upload again.",
                "details": results["error"]
            }), 503
            
        # Agar response sahi hai (List format mein hai)
        best_match = results[0]
        predicted_text = best_match['label']
        confidence_score = round(best_match['score'] * 100, 1)
        
        if "liquid" in predicted_text:
            final_state = "Liquid Wastewater 💧"
        elif "gas" in predicted_text:
            final_state = "Gas / Smoke 💨"
        else:
            final_state = "Solid Waste 🪨"
            
        return jsonify({
            "label": final_state,
            "confidence": f"{confidence_score}%",
            "message": "Analyzed successfully!"
        })
        
    except Exception as e:
        return jsonify({"error": "Something went wrong on the server. Try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)
