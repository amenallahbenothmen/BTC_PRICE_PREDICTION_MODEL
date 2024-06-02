import numpy as np
import pandas as pd
import streamlit as st
import os
from datetime import datetime
import yaml
from PIL import Image

def welcome():
    return "Welcome All"

def getResult():
    filePath = "C:\\Users\\amenm\\OneDrive\\Desktop\\p2m_final\\BTC_PRICE_PREDICTION_MODEL\\prediction\\result.npy"
    result = np.load(filePath)[0]
    result = float(result)
    formatted_result = f"{result:.3f}"
    return formatted_result

def main():
    st.set_page_config(page_title="Bitcoin Price Predictor", page_icon=":chart_with_upwards_trend:", layout="centered")

    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
            color: #333333;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 15em;
        }
        .stButton>button:hover {
            background-color: #ff6b6b;
            color: white;
        }
        .title {
            font-family: 'Arial', sans-serif;
            font-size: 2.5em;
            color: #333333;
            text-align: center;
            margin-top: 0.5em;
        }
        .sub-title {
            font-family: 'Arial', sans-serif;
            font-size: 1.5em;
            color: #333333;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="title">Bitcoin Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">ML-powered Bitcoin Price Prediction App</div>', unsafe_allow_html=True)
    
    if st.button("Predict"):
        paramsFilePath = "params.yaml"
        with open(paramsFilePath, 'r') as file:
            data = yaml.safe_load(file)
        date = data['CURRENT_DATE']
        current_datetime = datetime.now()
        current_date = current_datetime.date()
        formatted_date = current_date.strftime("%Y-%m-%d")
        if date != formatted_date:
            data['CURRENT_DATE'] = formatted_date
            with open(paramsFilePath, 'w') as file:
                yaml.dump(data, file)
            os.system("python main.py")
        result = getResult()
        st.success(f'The predicted Bitcoin price is: ${result}')
    
    if st.button("About"):
        st.info("This app predicts Bitcoin prices using a Machine Learning model. Built with Streamlit.")

if __name__ == '__main__':
    main()
