Author: Gayathri Saxena

# Student Stress Level Prediction using Machine Learning

Author: Gayathri Saxena

## Project Overview
This project predicts student stress levels based on daily habits such as study hours, sleep duration, screen time, and exercise using machine learning.

## Technologies Used
- Python
- pandas
- scikit-learn
- Random Forest Classifier

## Dataset
The dataset (`student_data.csv`) contains student lifestyle data with features:
- study_hours
- sleep_hours
- screen_time
- exercise
- stress_level

## Model Workflow
1. Load and preprocess the dataset
2. Encode categorical variables
3. Train a Random Forest classifier
4. Predict stress level for new inputs

## How to Run
pip install pandas scikit-learn
python stress_prediction.py

## Output
The model prints accuracy and predicted stress level.

## Future Improvements
- Use larger real-world datasets
- Improve accuracy with hyperparameter tuning
- Deploy as a web application
