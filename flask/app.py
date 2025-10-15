from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('3d_printer.pkl')
scaler = joblib.load('Min_max_scaler.pkl')


model_features = [
    'layer_height', 'wall_thickness', 'infill_density',
    'nozzle_temperature', 'bed_temperature', 'print_speed',
    'fan_speed', 'roughness', 'tension_strenght', 'elongation',
    'infill_pattern_honeycomb'
]

def map_material(pred):
    return 'PLA' if pred == True else 'ABS'


@app.route('/')
def index():
    # Landing page
    return render_template('index.html')

@app.route('/home')
def home():
    # Prediction page
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {key: float(request.form[key]) for key in model_features}
        user_input['infill_pattern_honeycomb'] = int(user_input['infill_pattern_honeycomb'])
        input_df = pd.DataFrame([user_input], columns=model_features)
        input_scaled = scaler.transform(input_df)
        raw_prediction = model.predict(input_scaled)[0]
        prediction = map_material(raw_prediction)

        description_dict = {
            'PLA': 'Good for beginners, low warping, biodegradable.',
            'ABS': 'Strong, heat-resistant, requires heated bed.'
        }
        description = description_dict.get(prediction, '')

        return render_template('result.html',
                               material=prediction,
                               description=description,
                               inputs=user_input)

    except Exception as e:
        return render_template('result.html',
                               material=f"Error: {str(e)}",
                               description='',
                               inputs=None)

if __name__ == "__main__":
    app.run(debug=True)
