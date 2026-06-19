# 3D Printing Material Prediction Web App

A Flask-based web application that predicts the best 3D printing filament material using a machine learning model. The app includes secure JWT authentication, SQLite user storage, and a polished login/register flow.

## What this project includes

- Flask web app with model-driven material prediction
- Secure login and registration using JWT stored in cookies
- SQLite database for user credentials
- Tabbed login/register page with username/password/confirm-password support
- Model inference using pre-trained `3d_printer.pkl` and `Min_max_scaler.pkl`
- GitHub Actions workflow for CI and deployment readiness
- `.gitignore` to protect credentials, local environment files, and model/database artifacts

## Repository structure

```
3d-printing-model/
├─ .github/
│  └─ workflows/
│     └─ python-app.yml
├─ flask/
│  ├─ static/
│  │  ├─ css/
│  │  │  └─ auth.css
│  │  ├─ images/
│  │  └─ js/
│  │     └─ auth.js
│  ├─ templates/
│  │  ├─ about.html
│  │  ├─ auth.html
│  │  ├─ home.html
│  │  ├─ index.html
│  │  └─ result.html
│  ├─ 3d_printer.pkl
│  ├─ Min_max_scaler.pkl
│  ├─ app.py
│  ├─ requirements.txt
│  └─ users.db  # runtime-generated
├─ dataset/
├─ Output video/
├─ training/
└─ .gitignore
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd 3d-printing-model/flask
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   python app.py
   ```
5. Open `http://127.0.0.1:5000` in your browser.

## Usage

- Visit `/auth` for login or registration.
- After signing in, access `/home` to submit print parameters.
- The app predicts PLA or ABS and shows a result page.
- Use `/logout` to end the session.

## Available routes

- `GET /` — landing page
- `GET /auth` — login/register screen
- `POST /auth` — authenticate or register user
- `GET /home` — prediction page (requires login)
- `GET /about` — about page (requires login)
- `POST /predict` — model prediction route (requires login)
- `GET /logout` — sign out

## GitHub Actions deployment

A GitHub Actions workflow file is added at:
- `.github/workflows/python-app.yml`

It performs:
- repository checkout
- Python setup
- dependency installation
- simple Python compile check

Push to `main` or `master` to trigger CI.

## Security and .gitignore

A `.gitignore` file was added to protect sensitive files and local artifacts:
- `.venv/`
- `__pycache__/`
- `.env`
- `*.db`
- `*.pkl`
- `*.key`, `*.pem`
- `secrets.json`, `credentials.json`
- `.idea/`, `.vscode/`

## Notes

- `users.db` is generated automatically when the app first runs.
- If you want to deploy to GitHub Pages or another platform, adjust the workflow to your chosen provider.

