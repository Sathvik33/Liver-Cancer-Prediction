import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load("models/Cancerpred.joblib")
label_encoders = joblib.load("models/encoders.joblib")
scaler = joblib.load("models/scaler.joblib")


column_order = ['Country', 'Region', 'Population', 'Incidence_Rate', 'Mortality_Rate',
       'Gender', 'Age', 'Alcohol_Consumption', 'Smoking_Status',
       'Hepatitis_B_Status', 'Hepatitis_C_Status', 'Obesity', 'Diabetes',
       'Rural_or_Urban', 'Seafood_Consumption', 'Herbal_Medicine_Use',
       'Healthcare_Access', 'Screening_Availability', 'Treatment_Availability',
       'Liver_Transplant_Access', 'Ethnicity', 'Preventive_Care',
       'Survival_Rate', 'Cost_of_Treatment']

categorical_columns = ['Country', 'Region', 'Gender', 'Smoking_Status', 'Hepatitis_B_Status', 'Hepatitis_C_Status',
                       'Obesity', 'Diabetes', 'Rural_or_Urban', 'Seafood_Consumption',
                       'Herbal_Medicine_Use', 'Healthcare_Access', 'Screening_Availability',
                       'Treatment_Availability', 'Liver_Transplant_Access', 'Ethnicity',
                       'Preventive_Care', 'Alcohol_Consumption']

numerical_columns = ['Population', 'Incidence_Rate', 'Mortality_Rate', 'Age', 'Survival_Rate', 'Cost_of_Treatment']


available_countries = [
    'Nigeria', 'Kenya', 'South Africa', 'Germany', 'France', 'United Kingdom', 'India', 'Pakistan',
    'Bangladesh', 'Brazil', 'Colombia', 'Iran', 'Turkey', 'Egypt', 'Vietnam', 'Thailand',
    'Indonesia', 'China', 'Japan', 'South Korea', 'Mexico', 'Spain', 'United States'
]


country_region_map = {
    'Nigeria': 'Sub-Saharan Africa', 'Kenya': 'Sub-Saharan Africa', 'South Africa': 'Sub-Saharan Africa',
    'Germany': 'Europe', 'France': 'Europe', 'United Kingdom': 'Europe',
    'India': 'South Asia', 'Pakistan': 'South Asia', 'Bangladesh': 'South Asia',
    'Brazil': 'South America', 'Colombia': 'South America',
    'Iran': 'Middle East', 'Turkey': 'Middle East',
    'Egypt': 'North Africa',
    'Vietnam': 'Southeast Asia', 'Thailand': 'Southeast Asia', 'Indonesia': 'Southeast Asia',
    'China': 'Eastern Asia', 'Japan': 'Eastern Asia', 'South Korea': 'Eastern Asia',
    'Mexico': 'Latin America', 'Spain': 'Latin America',
    'United States': 'North America'
}


def preprocess_input(user_inputs):
    processed_inputs = []

    for col in column_order:
        if col in categorical_columns:
            if col in label_encoders:
                encoder = label_encoders[col]
                if user_inputs[col] in encoder.classes_:
                    processed_inputs.append(encoder.transform([user_inputs[col]])[0])
                else:
                    processed_inputs.append(-1)
            else:
                processed_inputs.append(-1)
        else:
            processed_inputs.append(float(user_inputs.get(col, 0)))

    return pd.DataFrame([processed_inputs], columns=column_order)

st.title("Liver Cancer Prediction App")
# st.write("### Countries Available in Dataset:")
# st.write(", ".join(available_countries))

st.write("### Enter Details Below")

user_inputs = {}


user_inputs['Country'] = st.selectbox("Select Country", available_countries)

user_inputs['Region'] = country_region_map[user_inputs['Country']]
st.write(f"Selected Region: **{user_inputs['Region']}**")

user_inputs['Gender'] = st.selectbox("Select Gender", ['Male', 'Female'])
user_inputs['Alcohol_Consumption'] = st.selectbox("Select Alcohol Consumption Level", ['Low', 'Moderate', 'High'])
user_inputs['Smoking_Status'] = st.selectbox("Select Smoking Status", ['Smoker', 'Non-Smoker'])
user_inputs['Hepatitis_B_Status'] = st.selectbox("Select Hepatitis B Status", ['Negative', 'Positive'])
user_inputs['Hepatitis_C_Status'] = st.selectbox("Select Hepatitis C Status", ['Negative', 'Positive'])
user_inputs['Obesity'] = st.selectbox("Select Obesity Status", ['Normal', 'Underweight', 'Obese', 'Overweight'])
user_inputs['Diabetes'] = st.selectbox("Select Diabetes Status", ['Yes', 'No'])
user_inputs['Rural_or_Urban'] = st.selectbox("Select Living Area", ['Rural', 'Urban'])
user_inputs['Seafood_Consumption'] = st.selectbox("Select Seafood Consumption Level", ['Medium', 'High', 'Low'])
user_inputs['Herbal_Medicine_Use'] = st.selectbox("Do you use Herbal Medicine?", ['Yes', 'No'])
user_inputs['Healthcare_Access'] = st.selectbox("Select Healthcare Access Level", ['Poor', 'Good', 'Moderate'])
user_inputs['Screening_Availability'] = st.selectbox("Is Screening Available?", ['Available', 'Not Available'])
user_inputs['Treatment_Availability'] = st.selectbox("Is Treatment Available?", ['Available', 'Not Available'])
user_inputs['Liver_Transplant_Access'] = st.selectbox("Is Liver Transplant Accessible?", ['No', 'Yes'])
user_inputs['Ethnicity'] = st.selectbox("Select Ethnicity", ['Hispanic', 'Mixed', 'African', 'Asian', 'Caucasian'])
user_inputs['Preventive_Care'] = st.selectbox("Select Preventive Care Level", ['Good', 'Moderate', 'Poor'])


for col in numerical_columns:
    user_inputs[col] = st.number_input(f"Enter {col}", value=0.0, format="%.2f")


if st.button("Predict"):
    try:
        processed_input = preprocess_input(user_inputs)
        processed_input_scaled = scaler.transform(processed_input)
        prediction_proba = model.predict_proba(processed_input_scaled)[0][1] * 100
        st.write(f"### Probability of getting liver cancer is: **{prediction_proba:.2f}%**")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
