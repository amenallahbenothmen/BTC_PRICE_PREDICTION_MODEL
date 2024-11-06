

## Project Overview

The **BTC Price Prediction Model** is a deep learning application designed to forecast Bitcoin prices using historical market data. By implementing bidirectional LSTM neural networks and leveraging MLOps methodologies, the project aims to provide accurate price predictions and streamline the deployment process for continuous integration and delivery.

## Features

- **Historical Data Analysis**: Processes and analyzes historical Bitcoin price data for model training.
- **Bidirectional LSTM Neural Networks**:Utilizes advanced deep learning techniques for time series forecasting.
- **Containerization**: Employs Docker for easy deployment and scalability.
- **Experiment Tracking**: Integrates MLflow for tracking model performance and experiments.
- **Data Version Control**: Uses DVC for efficient data management and reproducibility.



## Prerequisites

- **Python 3.9** 
- **Docker**
- **DVC**
- **MLflow**
- **TensorFlow**
- **Flask**

## Setup and Installation

Follow the steps below to set up and run the application:

### 1. Clone the Repository

```bash
git clone https://github.com/amenallahbenothmen/BTC_PRICE_PREDICTION_MODEL.git
cd BTC_PRICE_PREDICTION_MODEL
```

### 2.Create a Virtual Environment

```bash
conda create -n btc_price_prediction python=3.9
conda activate btc_price_prediction
```


### 3.Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run  app.py
```
