import streamlit as st
import joblib
import pandas as pd

# to download model related files from hugging face repository
from huggingface_hub import hf_hub_download

# hugging face login profile id
hf_login_id = "JaiBhatia020373"

# set name of the new repository on the Hugging face hub
repo_name = "tourism-package-prediction-model"

# repository type - model repository
repo_type = "model"

repo_id = hf_login_id + "/" + repo_name

# Example: download the joblib file from hugging face repo onto local system
model_path = hf_hub_download(
    repo_id=repo_id,
    filename="tourism-package-prediction-classifier.joblib"
)

print(model_path)

# Load pipeline model from joblib file on local system
pipeline = joblib.load(model_path)

st.title("🏖️ Tourism Package Prediction")
st.write("Predict whether or not the customer will buy the holiday package.")
st.write("Please provide customer details below for prediction.")

# collect user inputs on customer information
age = st.number_input("Age", min_value=18, max_value=100, step=1)
duration_of_pitch = st.number_input("Duration of Pitch", min_value=1, step=1)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
family_size = st.number_input("Family Size", min_value=1, max_value=10, step=1)
contact_type = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
profession = st.selectbox("Profession", ["Salaried", "Self_Employed", "Student", "Retired"])
product_pitched = st.selectbox("Product Pitched", ["Basic","Standard","Deluxe","King","Super Deluxe"])
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1,2,3,4,5])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
city_tier = st.selectbox("City Tier", [1, 2, 3])
number_of_followups = st.number_input("Number of Followups", min_value=0)
number_of_trips = st.number_input("Number of Trips", min_value=0)
number_of_person_visiting = st.number_input("Number of Person Visiting", min_value=1)
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0)
passport = st.selectbox("Passport", ["Yes", "No"])
own_car = st.selectbox("Own Car", ["Yes", "No"])
monthly_income = st.number_input("Monthly Income", min_value=0)

# --- Collect inputs into DataFrame ---
input_data = pd.DataFrame({
    "Age": [age],
    "FamilySize": [family_size],
    "Gender": [gender],
    "MaritalStatus": [marital_status],
    "Profession": [profession],
    "CityTier": [city_tier],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "OwnCar": [own_car],
    "Occupation": [occupation],
    "Designation": [designation],
    "TypeofContact": [contact_type],
    "DurationOfPitch": [duration_of_pitch],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "NumberOfFollowups": [number_of_followups],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MonthlyIncome": [monthly_income] 
})

# --- Prediction ---
if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    result = "Yes (Will take package)" if prediction in [1, "yes"] else "No (Will not take package)"
    st.success(f"Prediction: {result}")
