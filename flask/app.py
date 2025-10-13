import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

try:
    model_path = r"C:\Users\USER\3d-printing-model\flask\3d_printer.pkl"
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"ERROR: Model file not found at path: {model_path}")
    model = None


try:
    scaler_path = r"C:\Users\USER\3d-printing-model\flask\Min_max_scaler.pkl"
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print(f"WARNING: Scaler file not found at path: {scaler_path}")
    scaler = None


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/result', methods=['GET'])
def result_page():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template("result.html", prediction_text="Error: Model not loaded. Check server logs.")

    try:
        # All 11 features
        features_name = [
            'layer_height', 'wall_thickness', 'infill_density', 'infill_pattern',
            'nozzle_temperature', 'bed_temperature', 'print_speed', 'fan_speed',
            'roughness', 'tension_strenght', 'elongation'
        ]

        # Collect input values from form
        input_features = []
        for name in features_name:
            value = request.form.get(name, 0)
            if name == 'infill_pattern':
                value = int(value)  # categorical 0/1
            else:
                value = float(value)
            input_features.append(value)

        features_value = np.array([input_features])  # shape (1,11)

        # Apply scaler
        if scaler is not None:
            features_value = scaler.transform(features_value)

        # Predict
        prediction = model.predict(features_value)
        output = prediction[0]

        # Get probabilities if available
        probabilities = model.predict_proba(features_value)[0] if hasattr(model, "predict_proba") else None

        # Map output to material
        if output == 1:
            prediction_text = (
                "The Suggested Material is ABS. "
                "Acrylonitrile butadiene styrene (ABS) is a common thermoplastic polymer used in 3D printing."
            )
        else:
            prediction_text = (
                "The Suggested Material is PLA. "
                "PLA (polylactic acid) is a thermoplastic made from renewable resources such as corn starch or sugar cane."
            )

        return render_template("result.html", prediction_text=prediction_text, probabilities=probabilities)

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
