import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__, template_folder='templates')

try:
    model_path = r"C:\Users\USER\3d-printing-model\flask\3d_print.pkl"
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"ERROR: Model file not found at path: {model_path}")
    model = None



@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/result', methods=['GET'])
def result_page():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template("result.html", prediction_text="Error: Model is not loaded. Please check server logs.")

    try:
        features_name = [
            'layer_height', 'wall_thickness', 'infill_density', 'infill_pattern',
            'nozzle_temperature', 'bed_temperature', 'print_speed', 'fan_speed',
            'roughness', 'tension_strenght', 'elongation'
        ]
        
        input_features = [float(request.form.get(name, 0)) for name in features_name]
        
        features_value = [np.array(input_features)]

        prediction = model.predict(features_value)
        output = prediction[0]
        
        if output == 1:
            prediction_text = (
                "The Suggested Material is ABS. "
                "(Acrylonitrile butadiene styrene is a common thermoplastic polymer "
                "typically used for injection molding applications)"
            )
        elif output == 0:
            prediction_text = (
                "The Suggested Material is PLA. "
                "(PLA, also known as polylactic acid or polylactide, is a thermoplastic "
                "made from renewable resources such as corn starch, tapioca roots, or "
                "sugar cane, unlike other industrial materials made primarily from petroleum)"
            )
        else:
            prediction_text = (
                "The given values do not match the range of values of the model. "
                "Try giving the values in the mentioned range."
            )
        
        return render_template("result.html", prediction_text=prediction_text)
    
    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

