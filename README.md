# Daily Water Intake Prediction using Machine Learning

This project applies multiple machine learning models to predict **Hydration Level (Good vs Poor)** using a Daily Water Intake dataset.

## Project Overview

The goal of this project is to build and compare machine learning models for **binary classification** in a health/wellness context. The project evaluates classical machine learning approaches alongside an **Artificial Neural Network (ANN)** to understand performance tradeoffs.

## Models Used

- Logistic Regression  
- Random Forest Classifier  
- Artificial Neural Network (ANN) — `MLPClassifier` (scikit-learn)

## Workflow

- Loaded and reviewed the dataset  
- Encoded the target label (Hydration Level) **explicitly**: `Poor = 0`, `Good = 1`  
- One-hot encoded categorical features (Gender, Physical Activity Level, Weather)  
- Standardized numeric features  
- Split data into training and test sets (80/20, stratified, random_state=42)  
- Trained and evaluated multiple models  
- Compared models using accuracy, ROC-AUC, weighted F1-score, and training time  

## Results Summary

Models were trained using an 80/20 train-test split (stratified, random_state=42). Categorical features were one-hot encoded and numeric features were standardized. The target label was encoded as: **Poor = 0, Good = 1**.

### Model Performance (Test Set)

| Model | Test Accuracy | ROC-AUC | Weighted F1 | Training Time (s) |
|------|--------------:|--------:|------------:|------------------:|
| ANN (MLPClassifier) | 0.998000 | 0.999988 | 0.997998 | 7.540646 |
| Logistic Regression | 0.995833 | 0.999948 | 0.995818 | 3.826432 |
| Random Forest | 0.986333 | 0.999024 | 0.986278 | 5.304417 |

### Confusion Matrices (rows=true, cols=pred)

**ANN (MLPClassifier)**  
[[1207, 10], [2, 4781]]

**Logistic Regression**  
[[1193, 24], [1, 4782]]

**Random Forest**  
[[1163, 54], [28, 4755]]

**Overall:** The ANN achieved the best test accuracy and weighted F1-score, while Logistic Regression performed nearly as well. Random Forest performed slightly lower on accuracy but still achieved an excellent ROC-AUC score.

## Tools & Technologies

- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib  
- Git / GitHub  

## Dataset

`Daily_Water_Intake.csv` (place it in the same folder as the script when running locally).  
If you don’t want to commit the dataset to GitHub, keep it locally and add it to `.gitignore`.

## How to Run

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Put `Daily_Water_Intake.csv` in the same folder as:
   - `daily_water_intake_ml_prediction_husandeep_atwal.py`
3. Run:
   - `python daily_water_intake_ml_prediction_husandeep_atwal.py`

## Notes

This project was completed as part of an academic course and demonstrates applied machine learning fundamentals, model evaluation, and comparison of different approaches.
