import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model and encoders
model = load_model('model.h5')

with open('one_hot_encoder_Transaction_Type.pkl', 'rb') as file:
    one_hot_encoder_Transaction_Type = pickle.load(file)

with open('one_hot_encoder_Transaction_channel.pkl', 'rb') as file:
    one_hot_encoder_Transaction_channel = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Suspicious Customer Detection")

# User Input
Amount = st.number_input('Amount', min_value=0)
Transaction_Type = st.selectbox('Transaction Type', one_hot_encoder_Transaction_Type.categories_[0])
Daily_Transaction_Count = st.number_input('Daily Transaction Count', min_value=0)
Daily_Transaction_Volume = st.number_input('Daily Transaction Volume', min_value=0)
Monthly_Transaction_Count = st.number_input('Monthly Transaction Count', min_value=0)
Monthly_Transaction_Volume = st.number_input('Monthly Transaction Volume', min_value=0)
Previous_Alert = st.selectbox('Previous Alert', [0, 1])  # Changed to list
Customer_Age = st.slider('Customer Age', 18, 92)
Transaction_Channel = st.selectbox('Transaction Channel', one_hot_encoder_Transaction_channel.categories_[0])
Risk_Score = st.number_input('Risk Score', min_value=0)

# Prepare the input data
input_data = pd.DataFrame({
    'Amount': [Amount],
    'Transaction Type': [Transaction_Type],  # Added Transaction Type to the input DataFrame
    'Daily Transaction Count': [Daily_Transaction_Count],
    'Daily Transaction Volume': [Daily_Transaction_Volume],
    'Monthly Transaction Count': [Monthly_Transaction_Count],
    'Monthly Transaction Volume': [Monthly_Transaction_Volume],
    'Previous Alert': [Previous_Alert],
    'Customer Age': [Customer_Age],
    'Transaction Channel': [Transaction_Channel],  # Added Transaction Channel to the input DataFrame
    'Risk Score': [Risk_Score]
})

# One-hot encode 'Transaction Type'
Transaction_type_encoder = one_hot_encoder_Transaction_Type.transform(input_data[['Transaction Type']]).toarray()
Transaction_type_encoder_df = pd.DataFrame(Transaction_type_encoder, 
                                            columns=one_hot_encoder_Transaction_Type.get_feature_names_out(['Transaction Type']))

# One-hot encode 'Transaction Channel'
Transaction_channel_encoder = one_hot_encoder_Transaction_channel.transform(input_data[['Transaction Channel']]).toarray()
Transaction_channel_encoder_df = pd.DataFrame(Transaction_channel_encoder, 
                                              columns=one_hot_encoder_Transaction_channel.get_feature_names_out(['Transaction Channel']))

# Combine encoded data with the input data
input_data_combined = pd.concat([input_data.reset_index(drop=True), 
                                  Transaction_type_encoder_df, 
                                  Transaction_channel_encoder_df], axis=1)

# Drop the original categorical columns
input_data_final = input_data_combined.drop(columns=['Transaction Type', 'Transaction Channel'])  # Dropped original columns

# Scale the input data
input_scaled = scaler.transform(input_data_final)

# Prediction
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

# Display prediction results
if prediction_prob > 0.5:
    st.write("The customer is suspicious.")
else:
    st.write("The customer is not suspicious.")
