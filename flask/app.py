from flask import Flask, render_template, request, redirect, url_for, make_response
import pandas as pd
import joblib
import sqlite3
import os
import datetime
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-key-123')
app.config['JWT_ALGORITHM'] = 'HS256'

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'users.db')


def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        '''
    )
    conn.commit()
    conn.close()


def get_user(username):
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, username, password_hash FROM users WHERE username=?', (username,))
    row = cur.fetchone()
    conn.close()
    return row


def create_user(username, password):
    password_hash = generate_password_hash(password)
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
        (username, password_hash, datetime.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def create_token(username):
    payload = {
        'sub': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm=app.config['JWT_ALGORITHM'])
    if isinstance(token, bytes):
        token = token.decode('utf-8')
    return token


def decode_token(token):
    try:
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=[app.config['JWT_ALGORITHM']])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_current_user():
    token = request.cookies.get('access_token')
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    return payload.get('sub')


def login_required(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not get_current_user():
            return redirect(url_for('auth', next=request.path))
        return func(*args, **kwargs)

    return wrapper


init_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, '3d_printer.pkl')
)

scaler = joblib.load(
    os.path.join(BASE_DIR, 'Min_max_scaler.pkl')
)


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
    if not get_current_user():
        return redirect(url_for('auth'))
    return render_template('index.html')

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if get_current_user():
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            return render_template('auth.html', error='Username and password are required.', action=action)

        if action == 'signin':

            if get_user(username):
                return render_template(
                    'auth.html',
                    error='Username already exists.',
                    action=action
                )

            if len(password) < 8:
                return render_template(
                    'auth.html',
                    error='Password must be at least 8 characters.',
                    action=action
                )

            create_user(username, password)

            token = create_token(username)

            response = make_response(
                redirect(url_for('index'))
            )

            response.set_cookie(
                'access_token',
                token,
                httponly=True,
                samesite='Lax'
            )

            return response
                

        if action == 'login':
            user = get_user(username)
            if not user or not check_password_hash(user[2], password):
                return render_template('auth.html', error='Invalid username or password.', action=action)
            token = create_token(username)
            response = make_response(redirect(url_for('index')))
            response.set_cookie('access_token',token,httponly=True,secure=True,samesite='Lax')
            return response

    action = request.args.get('action', 'signin')
    return render_template('auth.html', action=action)

@app.route('/logout')
def logout():
    response = make_response(redirect(url_for('auth')))
    response.set_cookie('access_token', '', expires=0)
    return response

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
@login_required
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
