
# 3D Printer Material Prediction Using Machine Learning

A machine learning model for predicting the optimal 3D printer material based on design specifications and printing requirements. By analyzing material properties, print parameters, and desired outcomes, this project aims to recommend suitable printing materials, aiding designers, engineers, and manufacturers in selecting the most appropriate materials for 3D printing applications.


##  Features

- **Layer Height:** Thickness of each printed layer, affecting surface quality and print time.  
- **Wall Thickness:** Outer shell thickness that determines strength and durability.  
- **Infill Density:** Percentage of internal fill controlling weight, strength, and material use.  
- **Infill Pattern:** Geometric design inside the model that affects strength and flexibility.  
- **Nozzle Temperature:** Heat of the nozzle to properly melt and extrude filament.  
- **Bed Temperature:** Heat of the print bed to improve layer adhesion and prevent warping.  
- **Print Speed:** Speed of the printer head, influencing print time and surface quality.  
- **Material:** Type of filament (e.g., PLA, ABS, PETG) used for printing.  
- **Fan Speed:** Cooling rate of the printed layer to enhance surface finish and bonding.  
- **Roughness:** Surface smoothness level of the final printed part.  
- **Tensile Strength:** Maximum stress the printed part can withstand before breaking.  
- **Elongation:** Measure of how much the printed material can stretch before failure.

## Project Structure
```
3d-printing-model/
│
├─ dataset/
│
├─ flask/
│  ├─ static/
│  │  ├─ css/
│  │  ├─ images/
│  │  └─ js/
│  ├─ templates/
│  ├─ 3d_printer.pkl
│  ├─ app.py
│  └─ Min_max_scaler.pkl
│
├─ Output video/
│
├─ training/
│  ├─ .ipynb_checkpoints/
│  ├─ 3_printing.ipynb
│  └─ my_data.csv
│
└─ External Libraries/
```
## API Reference

## 🌐 Web Routes

GET /                      – Landing page (`index.html`)  
GET /home                  – Prediction form (`home.html`)  
GET /about                 – About page (`about.html`)  
POST /predict              – Submit form data and return prediction
GET /result                – Result page (`result.html`)
## Work Flow

## Work Flow

- User uploads a 3D printing parameter dataset via the web app.
- The system pre-processes the data using the saved Min_max_scaler.pkl.
- User selects the input parameters (like layer height, infill density, nozzle temperature, etc.) for prediction.
- The trained model 3d_printer.pkl predicts the most suitable 3D printing material.
- The prediction result is displayed to the user on the web interface.
- User can view the given parameters on screen with result.

