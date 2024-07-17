import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data (you'll need to replace this with your actual data loading method)
@st.cache_data
def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df.drop('id', axis=1, inplace=True)
    df['bmi'].fillna(df['bmi'].mode()[0], inplace=True)
    return df

df = load_data()

# Preprocess the data
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
df['smoking_status'] = df['smoking_status'].map({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3})
df['work_type'] = df['work_type'].map({'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4})
df['age'] = pd.cut(df['age'], bins=[0, 12, 19, 30, 60, 100], labels=[0, 1, 2, 3, 4])

# Train the model
X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Heart Stroke Prediction')

st.write('Please enter the following information:')

gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.number_input('Age', min_value=0, max_value=100)
hypertension = st.checkbox('Hypertension')
heart_disease = st.checkbox('Heart Disease')
ever_married = st.checkbox('Ever Married')
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Children', 'Govt_job', 'Never_worked'])
Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)
smoking_status = st.selectbox('Smoking Status', ['Formerly smoked', 'Never smoked', 'Smokes', 'Unknown'])

if st.button('Predict'):
    # Prepare input data
    input_data = pd.DataFrame({
        'gender': [0 if gender == 'Female' else 1 if gender == 'Male' else 2],
        'age': [0 if age <= 12 else 1 if age <= 19 else 2 if age <= 30 else 3 if age <= 60 else 4],
        'hypertension': [1 if hypertension else 0],
        'heart_disease': [1 if heart_disease else 0],
        'ever_married': [1 if ever_married else 0],
        'work_type': [0 if work_type == 'Private' else 1 if work_type == 'Self-employed' else 2 if work_type == 'Children' else 3 if work_type == 'Govt_job' else 4],
        'Residence_type': [1 if Residence_type == 'Urban' else 0],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [0 if smoking_status == 'Formerly smoked' else 1 if smoking_status == 'Never smoked' else 2 if smoking_status == 'Smokes' else 3]
    })

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    if prediction[0] == 1:
        st.write('The model predicts a high risk of stroke.')
    else:
        st.write('The model predicts a low risk of stroke.')
    
    st.write(f'Probability of stroke: {probability:.2f}')

st.write('Note: This model is for educational purposes only. Always consult with a healthcare professional for medical advice.')