# News Sentiment-Based Stock Price Prediction System

This project implements a sophisticated stock price prediction system that combines news sentiment analysis with historical price data to provide more accurate stock price forecasts. The system uses FAISS (Facebook AI Similarity Search) for efficient similarity search of news headlines, FinBERT for financial sentiment analysis, and LSTM (Long Short-Term Memory) networks for time series prediction.

## Features

- Real-time stock price prediction using LSTM models
- News sentiment analysis using FinBERT
- Efficient similarity search using FAISS
- Interactive web interface
- Support for multiple stock symbols
- Historical price data analysis
- Visualization of predicted vs actual prices

## Tech Stack

- **Backend**: Flask, Python
- **Machine Learning**: 
  - FAISS for similarity search
  - FinBERT for financial sentiment analysis
  - LSTM for time series prediction
  - XGBoost for delta prediction
- **Data Processing**: Pandas, NumPy
- **Stock Data**: yfinance
- **News Data**: Google News RSS feeds
- **Frontend**: HTML/JavaScript with CORS support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ishan0803/Faiss_based_news_sentiment.git
cd Faiss_based_news_sentiment
```

2. Install the required Python packages:
```bash
pip install torch transformers faiss-cpu xgboost tensorflow flask-cors yfinance feedparser pandas numpy matplotlib scikit-learn
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Access the web interface through your browser:
```
http://localhost:5000
```

3. Enter a stock symbol and view predictions based on both technical analysis and news sentiment.

## System Components

### 1. News Sentiment Analysis (`headline_analyzer.py`)
- Uses FinBERT for financial sentiment embedding
- Implements FAISS for efficient similarity search
- Predicts price impact based on news headlines
- Utilizes XGBoost for delta prediction based on similar news patterns

### 2. Price Prediction (`predict_base_price.py`)
- Fetches historical stock data using yfinance
- Implements LSTM model for time series prediction
- Creates and manages prediction datasets
- Visualizes predictions vs actual prices
- Supports customizable time windows and prediction parameters

### 3. Web Interface (`app.py`)
- Flask-based REST API with CORS support
- Handles stock symbol selection
- Manages dataset creation and model training
- Provides prediction endpoints
- Integrates both sentiment and price prediction systems

## Model Configuration

The system can be customized with various parameters:
- Time steps: 100 (default)
- Training epochs: 5 (default)
- Batch size: 16 (default)
- Years of historical data: 5 (default)

Model performance visualization is available in `predicted_vs_actual_prices.png`, showing the comparison between predicted and actual stock prices.

## How It Works

1. **News Analysis**:
   - Fetches relevant news articles using Google News RSS
   - Processes headlines using FinBERT for financial sentiment
   - Uses FAISS to find similar historical headlines
   - Predicts price impact using XGBoost

2. **Price Prediction**:
   - Collects historical price data via yfinance
   - Preprocesses data using MinMaxScaler
   - Trains LSTM model on the processed data
   - Generates future price predictions

3. **Integration**:
   - Combines sentiment analysis with price predictions
   - Provides comprehensive stock movement forecasts
   - Updates predictions with new data

## 🌟 Acknowledgments

- [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone) for financial sentiment analysis
- Facebook's FAISS library for efficient similarity search
- yfinance for providing stock market data
- The open-source community for various tools and libraries used in this project
