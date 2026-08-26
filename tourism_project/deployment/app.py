
import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline
model_path = os.path.join(
    os.path.dirname(__file__),
    "tourism_model_v1.joblib"
)

model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction App")

st.write("""
This application predicts whether a customer is likely
to purchase the Wellness Tourism Package based on
customer profile and interaction data.
""")

# =====================================================
# Customer Inputs
# =====================================================

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

TypeofContact = st.selectbox(
    "Type of Contact",
    [
        "Company Invited",
        "Self Enquiry"
    ]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

Gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

NumberOfPersonVisiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    [
        "Not Married",
        "Married",
        "Divorced"
    ]
)

NumberOfTrips = st.number_input(
    "Number Of Trips",
    min_value=0,
    max_value=50,
    value=2
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

NumberOfFollowups = st.number_input(
    "Number Of Followups",
    min_value=0,
    max_value=20,
    value=2
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch",
    min_value=0,
    max_value=200,
    value=20
)

# =====================================================
# Age Group Logic
# =====================================================

if Age <= 25:
    AgeGroup = "18-25"
elif Age <= 35:
    AgeGroup = "26-35"
elif Age <= 45:
    AgeGroup = "36-45"
elif Age <= 55:
    AgeGroup = "46-55"
elif Age <= 65:
    AgeGroup = "56-65"
else:
    AgeGroup = "65+"

# =====================================================
# Create Input DataFrame
# =====================================================

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "AgeGroup": AgeGroup
}])

# =====================================================
# Prediction
# =====================================================

if st.button("Predict Purchase"):

    purchase_probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = (
        purchase_probability >= 0.45
    ).astype(int)

    result = (
        "Likely To Purchase"
        if prediction == 1
        else "Not Likely To Purchase"
    )

    st.subheader("Prediction Result")

    st.success(result)

    st.write(
        f"Purchase Probability: {purchase_probability:.2%}"
    )
