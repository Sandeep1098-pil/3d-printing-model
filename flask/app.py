from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('3d_printer.pkl')
scaler = joblib.load('Min_max_scaler.pkl')

# Features in exact order
model_features = [
    'layer_height', 'wall_thickness', 'infill_density',
    'nozzle_temperature', 'bed_temperature', 'print_speed',
    'fan_speed', 'roughness', 'tension_strenght', 'elongation',
    'infill_pattern_honeycomb'
]

# Map boolean prediction to material name
def map_material(pred):
    return 'PLA' if pred == True else 'ABS'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        user_input = {key: float(request.form[key]) for key in model_features}
        user_input['infill_pattern_honeycomb'] = int(user_input['infill_pattern_honeycomb'])

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input], columns=model_features)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        raw_prediction = model.predict(input_scaled)[0]
        prediction = map_material(raw_prediction)

        # Optional description
        description_dict = {
            'PLA': 'Good for beginners, low warping, biodegradable.',
            'ABS': 'Strong, heat-resistant, requires heated bed.',
        }
        description = description_dict.get(prediction, '')

        # Render result
        return render_template(
            'result.html',
            material=prediction,
            description=description,
            inputs=user_input
        )

    except Exception as e:
        return render_template(
            'result.html',
            material=f"Error: {str(e)}",
            description='',
            inputs=None
        )

if __name__ == "__main__":
    app.run(debug=True)
