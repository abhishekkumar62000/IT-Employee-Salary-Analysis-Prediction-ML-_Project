import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the saved model, encoder, and scaler
with open('Gaussian_naive_bayes_model.pkl', 'rb') as model_file:
    gnb = pickle.load(model_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set the background image and text color using CSS
def set_background(png_file):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{png_file}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    /* Set text color to black for all text elements */
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
        text-align: center; /* Center align text */
    }}
    .stButton > button {{
        background-color: #4C4C6D; /* Button color */
        color: white; /* Text color for button */
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD; /* Button hover color */
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to read and encode the image file
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Call the function with the background image
image_base64 = get_base64_image("image.jpg")  # Path to your image
set_background(image_base64)

# Streamlit title
st.markdown("<h1 class='title'>Gaussian Naive Bayes Classifier - Income Prediction</h1>", unsafe_allow_html=True)

# Collect user input
age = st.number_input('Age', min_value=0, max_value=100, value=30)
workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
fnlwgt = st.number_input('Final Weight', min_value=0, value=200000)
education = st.selectbox('Education', ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
education_num = st.number_input('Education Num', min_value=0, max_value=20, value=13)
marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox('Occupation', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
sex = st.selectbox('Sex', ['Male', 'Female'])
capital_gain = st.number_input('Capital Gain', min_value=0, value=0)
capital_loss = st.number_input('Capital Loss', min_value=0, value=0)
hours_per_week = st.number_input('Hours per Week', min_value=1, max_value=100, value=40)
native_country = st.selectbox('Native Country', ['United-States', 'Canada', 'Mexico', 'India', 'Germany', 'Philippines', 'Puerto-Rico'])

# Create input dictionary and DataFrame
input_data = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education': education,
    'education_num': education_num,
    'marital_status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'sex': sex,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'hours_per_week': hours_per_week,
    'native_country': native_country
}

input_df = pd.DataFrame([input_data])

# Create a button to make the prediction
if st.button('Predict'):
    # Apply encoder and scaler
    input_df_encoded = encoder.transform(input_df)
    input_df_scaled = scaler.transform(input_df_encoded)

    # Make prediction
    prediction = gnb.predict(input_df_scaled)
    prediction_prob = gnb.predict_proba(input_df_scaled)

    # Display prediction
    if prediction[0] == '>50K':
        st.success("The model predicts that the income is more than 50K.")
    else:
        st.warning("The model predicts that the income is less than or equal to 50K.")

    # Display model confidence
    st.write(f"Model Confidence: {prediction_prob[0][1] * 100:.2f}% for >50K")
