\# Delivery ETA Prediction \& SLA Risk Analysis



This project predicts food delivery ETA (in minutes) and provides a realistic SLA promise with risk tagging, simulating a real-world delivery analytics use case.



\## Problem Statement

Accurately estimating delivery time is critical for customer trust and operational planning. Underestimation leads to SLA breaches, while overestimation hurts customer experience.  

The goal is to predict ETA reliably and define an SLA buffer based on model behavior and residual risk.



\## Approach

\- Performed data cleaning and feature engineering on delivery, location, and order attributes

\- Trained an XGBoost regression model for ETA prediction

\- Used residual analysis to define SLA buffer instead of fixed thresholds

\- Conducted stress testing across ETA buckets to understand risk zones

\- Implemented risk tagging (High / Moderate / Low) based on prediction ranges



\## Model Performance

\- R² Score: 0.81  

\- Mean Absolute Error (MAE): 3.2 minutes  

\- Root Mean Squared Error (RMSE): 4.03 minutes  

\- Data leakage checks performed (no leakage detected)



\## Application

A Streamlit application is built for real-time inference:

\- User inputs pickup and delivery addresses

\- Coordinates fetched using OpenCage API

\- Distance calculated using geodesic formula

\- ETA predicted using trained model

\- SLA buffer added and risk tag displayed

<img width="838" height="886" alt="App-input" src="https://github.com/user-attachments/assets/df31ec8f-017e-448b-91b7-63e00469bd64" />







\## Tech Stack

\- Python

\- Pandas, NumPy

\- XGBoost, Scikit-learn

\- Streamlit

\- GeoPy



\## Project Structure



delivery-eta-xgboost-streamlit/

│

├── app.py

├── requirements.txt

├── README.md

│

├── artifacts/

│ ├── model\_v1.pkl

│ ├── features\_v1.pkl

│ ├── city\_freq\_map\_v1.pkl

│ ├── city\_freq\_mean\_v1.pkl

│

├── notebooks/

│ └── 01\_understanding\_and\_cleaning\_data.ipynb

│

├── assets/

│ └── app\_screenshot.png





\## Notes

\- Model artifacts are versioned and inference uses the same preprocessing pipeline as training

\- API keys are managed securely using Streamlit secrets



