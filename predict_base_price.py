import feedparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from urllib.parse import quote_plus
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import os

import json
import requests
import time

def get_stock_news(stock_symbol, total_articles=1000):
    """
    Fetch news articles related to a stock symbol using Google News RSS feeds.
    
    Parameters:
        stock_symbol (str): The stock symbol to search for news about
        total_articles (int): Maximum number of articles to fetch
        api_key (str): Not used in this implementation, kept for compatibility
        
    Returns:
        list: List of dictionaries containing article data
    """
    
    def get_news_from_query(query, max_articles=100):
        search_query = query.replace(' ', '+')
        url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        
        articles = []
        for entry in feed.entries[:max_articles]:
            title = entry.title
            link = entry.link
            published = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else None
            
            articles.append({
                'title': title,
                'published': published,
                'link': link,
                'source': entry.get('source', {}).get('title', 'Unknown') if hasattr(entry, 'source') else 'Unknown',
                'snippet': entry.get('summary', '') if hasattr(entry, 'summary') else ''
            })
        return articles
    
    # List of multiple queries to cover wide news
    queries = [
        f"{stock_symbol} stock",
        f"{stock_symbol} share price",
        f"{stock_symbol} quarterly results",
        f"{stock_symbol} earnings",
        f"{stock_symbol} revenue",
        f"{stock_symbol} profit",
        f"{stock_symbol} market news",
        f"{stock_symbol} news today",
        f"{stock_symbol} business",
        f"{stock_symbol} trading news",
        f"{stock_symbol} financials",
        f"{stock_symbol} investment",
        f"{stock_symbol} stock analysis",
        f"{stock_symbol} NSE news",
        f"{stock_symbol} BSE news",
        f"{stock_symbol} results {datetime.now().year}",
        f"{stock_symbol} forecast",
        f"{stock_symbol} dividend news"
    ]
    
    all_articles = []
    for q in queries:
        print(f"Fetching news for query: {q}")
        articles = get_news_from_query(q, max_articles=100)
        all_articles.extend(articles)
        time.sleep(1)  # Being nice to server
    
    print(f"\nFetched {len(all_articles)} articles before removing duplicates.")
    
    # Remove duplicates based on title
    unique_articles = {}
    for article in all_articles:
        title = article['title'].strip()
        if title not in unique_articles:
            unique_articles[title] = article
    
    all_articles = list(unique_articles.values())
    print(f"After removing duplicates: {len(all_articles)} articles.")
    
    # Sort by date (newest first)
    all_articles = sorted(all_articles, key=lambda x: x['published'] or datetime.min, reverse=True)
    
    # Format articles to match the original function's output structure
    formatted_articles = []
    for article in all_articles[:total_articles]:
        # Format the date in the RFC 2822 format
        formatted_date = article['published'].strftime("%a, %d %b %Y %H:%M:%S GMT") if article['published'] else datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
                
        # Change this line in get_stock_news function
        article_data = {
            "title": article['title'],
            "publishedAt": formatted_date,
            "source": article.get('source', 'Unknown') if isinstance(article.get('source'), str) else article.get('source', {}).get('title', 'Unknown'),
            "url": article['link'],
            "snippet": article.get('snippet', '')
        }
        
        formatted_articles.append(article_data)
    
    # Show a few articles for debugging
    for idx, article in enumerate(formatted_articles[:5]):  # Display only first 5
        print(f"{idx+1}. [{article['publishedAt']}] {article['title']}")
        print(f"    {article['url']}\n")
    
    print(f"Collected {len(formatted_articles)} out of {total_articles} requested articles")
    return formatted_articles

def get_stock_data(stock_symbol, years=5):
    """
    Download historical stock data for a given symbol.
    
    Parameters:
        stock_symbol (str): The stock symbol to download data for
        years (int): Number of years of historical data to fetch
        
    Returns:
        pandas.DataFrame: DataFrame containing stock data
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"Stock data loaded for {stock_symbol}:")
    print(df.head())
    return df

def prepare_data(df, time_steps=100):
    """
    Prepare and scale stock data for LSTM model.
    
    Parameters:
        df (pandas.DataFrame): DataFrame with stock data
        time_steps (int): Number of time steps to use for sequences
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler, split_index
    """
    # Normalize Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps, 3])  # Changed index from 0 to 3 to predict 'Close' price instead of 'Open'
    
    X, y = np.array(X), np.array(y)
    
    # Split into Train & Test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler, split

def build_lstm_model(input_shape, epochs=5, batch_size=16):
    """
    Build and train an LSTM model for stock price prediction.
    
    Parameters:
        input_shape (tuple): Shape of input data (time_steps, features)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tensorflow.keras.Model: Trained LSTM model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=16):
    """
    Train the LSTM model.
    
    Parameters:
        model: The LSTM model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        trained model with history
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )
    return model, history

def generate_predictions(model, X_test, df, scaler, time_steps, split):
    """
    Generate predictions using the trained model.
    
    Parameters:
        model: Trained LSTM model
        X_test: Test data
        df: Original DataFrame with stock data
        scaler: Fitted scaler
        time_steps: Number of time steps used
        split: Index used for train/test split
        
    Returns:
        pandas.DataFrame: DataFrame with dates, actual prices, and predictions
    """
    # Predict on test data
    test_predictions = model.predict(X_test)
    
    # Get actual dates for test period
    test_dates = df.index[time_steps + split:]
    
    # Create full predictions array
    test_predictions_full = np.zeros((len(test_predictions), df.shape[1]))
    test_predictions_full[:, 3] = test_predictions.flatten()  # Changed index from 0 to 3 for Close price
    
    # Inverse transform to get actual price values
    predicted_prices = scaler.inverse_transform(test_predictions_full)[:, 3]  # Changed index from 0 to 3 for Close price
    
    # Get actual prices - Changed from 'Open' to 'Close'
    actual_close_prices = df.iloc[time_steps + split:]['Close'].values.flatten()
    
    # Ensure all arrays have the same length
    min_length = min(len(actual_close_prices), len(test_dates), len(predicted_prices))
    actual_close_prices = actual_close_prices[:min_length]
    test_dates = test_dates[:min_length]
    predicted_prices = predicted_prices[:min_length]
    
    # Create DataFrame with results
    predictions_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actual_close_prices,
        'Predicted': predicted_prices,
        'Difference': actual_close_prices - predicted_prices,
        'Percentage_Diff': ((predicted_prices-actual_close_prices) / predicted_prices) * 100
    })

    plt.figure(figsize=(12,6))
    plt.plot(test_dates, actual_close_prices, label="Actual Close Price", color='blue')  # Changed from Open to Close
    plt.plot(test_dates, predicted_prices, label="Predicted Close Price", color='red')  # Changed from Open to Close
    plt.legend()
    plt.title("Predicted vs Actual Close Prices")  # Changed from Open to Close
    plt.savefig("predicted_vs_actual_prices.png", dpi=300, bbox_inches='tight')
    
    return predictions_df

def process_news_data(articles):
    """
    Process news articles and extract dates.
    
    Parameters:
        articles (list): List of news articles
        
    Returns:
        pandas.DataFrame: DataFrame with processed news data
    """
    news_with_dates = []
    for article in articles:
        try:
            # Parse the date using the RFC 2822 format used in get_stock_news
            published_date = datetime.strptime(article.get('publishedAt'), '%a, %d %b %Y %H:%M:%S %Z')
            # Format to match stock data dates for easy comparison
            formatted_date = published_date.strftime('%Y-%m-%d')
            
            news_with_dates.append({
                'Date': formatted_date,
                'Title': article.get('title'),
                'Source': article.get('source'),
                'URL': article.get('url')
            })
        except (ValueError, TypeError) as e:
            # Try alternative date formats
            try:
                # Try to use pandas' flexible date parser as a fallback
                published_date = pd.to_datetime(article.get('publishedAt'))
                formatted_date = published_date.strftime('%Y-%m-%d')
                
                news_with_dates.append({
                    'Date': formatted_date,
                    'Title': article.get('title'),
                    'Source': article.get('source'),
                    'URL': article.get('url')
                })
            except:
                print(f"Error parsing date: {article.get('publishedAt')}")
                continue
    
    news_df = pd.DataFrame(news_with_dates)
    print(f"Processed {len(news_df)} news articles with valid dates")
    if not news_df.empty:
        print(news_df.head())
    return news_df

def combine_news_and_predictions(news_df, predictions_df):
    """
    Combine news data with price predictions.
    
    Parameters:
        news_df (pandas.DataFrame): DataFrame with news data
        predictions_df (pandas.DataFrame): DataFrame with price predictions
        
    Returns:
        pandas.DataFrame: Combined DataFrame
    """
    # Handle empty dataframes
    if news_df.empty:
        print("No news data to combine")
        return pd.DataFrame()
        
    if predictions_df.empty:
        print("No prediction data to combine")
        return pd.DataFrame()
    
    # Convert Date column to string format for predictions_df
    if isinstance(predictions_df['Date'].iloc[0], pd.Timestamp):
        predictions_df['Date'] = predictions_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Ensure Date column in news_df is also in string format
    news_df['Date'] = pd.to_datetime(news_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Create combined DataFrame
    combined_df = pd.DataFrame()
    for date in news_df['Date'].unique():
        if date in predictions_df['Date'].values:
            # Get price data for the date
            price_row = predictions_df[predictions_df['Date'] == date].iloc[0]
            
            # Get all news articles for the date
            date_articles = news_df[news_df['Date'] == date]
            
            for _, article in date_articles.iterrows():
                row = {
                    'Date': date,
                    'Title': article['Title'],
                    'Source': article['Source'],
                    'URL': article['URL'],
                    'Actual_Price': price_row['Actual'],
                    'Predicted_Price': price_row['Predicted'],
                    'Difference': price_row['Difference'],
                    'Percentage_Diff': price_row['Percentage_Diff']
                }
                combined_df = pd.concat([combined_df, pd.DataFrame([row])], ignore_index=True)
    
    print(f"Found {len(combined_df)} news articles with matching stock price data")
    if not combined_df.empty:
        print(combined_df.head())
    else:
        print("No matching dates found between news and stock prices")
    
    return combined_df

def save_results_to_csv(df, stock_symbol):
    """
    Save results to a CSV file.
    
    Parameters:
        df (pandas.DataFrame): DataFrame to save
        stock_symbol (str): Stock symbol for filename
        
    Returns:
        str: Path to saved file
    """
    if df.empty:
        print(f"No data to save for {stock_symbol}")
        return None
        
    filename = f"{stock_symbol.replace('.', '_')}_predictions.csv"
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
    return filename

def save_model_and_scaler(model, scaler, stock_symbol):
    """
    Save model and scaler to files.
    
    Parameters:
        model: Trained LSTM model
        scaler: Fitted MinMaxScaler
        stock_symbol (str): Stock symbol for filename
        
    Returns:
        tuple: (model_path, scaler_path)
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f"models/{stock_symbol.replace('.', '_')}_model.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    import pickle
    scaler_path = f"models/{stock_symbol.replace('.', '_')}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    return model_path, scaler_path

def load_model_and_scaler(stock_symbol):
    """
    Load model and scaler from files.
    
    Parameters:
        stock_symbol (str): Stock symbol for filename
        
    Returns:
        tuple: (model, scaler)
    """
    import pickle
    
    # Define paths
    model_path = f"models/{stock_symbol.replace('.', '_')}_model.h5"
    scaler_path = f"models/{stock_symbol.replace('.', '_')}_scaler.pkl"
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler file not found for {stock_symbol}")
        return None, None
    
    # Load model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")
    
    return model, scaler

def predict_future_price(stock_symbol, future_date, time_steps=100, model=None, scaler=None):
    """
    Predict the stock price for a future date not in the dataset.
    
    Parameters:
        stock_symbol (str): Stock symbol (e.g., 'INFY.NS')
        future_date (str): Future date to predict price for (format: 'YYYY-MM-DD')
        time_steps (int): Number of time steps used for prediction
        model: Optional pre-loaded model
        scaler: Optional pre-loaded scaler
        
    Returns:
        dict: Prediction results
    """
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler(stock_symbol)
        
    if model is None or scaler is None:
        return {
            "error": f"Model or scaler not found for {stock_symbol}. Please train the model first.",
            "predicted_price": None
        }
    
    # Convert future_date to datetime object
    target_date = datetime.strptime(future_date, '%Y-%m-%d')
    
    # Get the most recent data up to today to use for prediction
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=time_steps*2)).strftime('%Y-%m-%d')
    
    # Fetch the most recent historical data
    print(f"Fetching most recent data from {start_date} to {end_date}")
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Check if we have sufficient data
    if len(data) < time_steps:
        return {
            "error": f"Not enough historical data available for {stock_symbol}. Need at least {time_steps} data points.",
            "predicted_price": None
        }
    
    # Get the needed columns
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Get the most recent time_steps days of data
    recent_data = df.iloc[-time_steps:]
    
    # Scale the data
    scaled_data = scaler.transform(recent_data)
    
    # Reshape for LSTM model input
    x_input = np.array([scaled_data])
    
    # Make prediction
    prediction = model.predict(x_input)
    
    # Create a full row for inverse transform
    pred_full = np.zeros((1, df.shape[1]))
    pred_full[0, 3] = prediction[0][0]  # Changed index from 0 to 3 for Close price
    
    # Inverse transform to get actual price value
    predicted_price = scaler.inverse_transform(pred_full)[0, 3]  # Changed index from 0 to 3 for Close price
    
    return {
        "date": future_date,
        "predicted_price": predicted_price
    }

def create_stock_prediction_model(stock_symbol, years=5, time_steps=100, epochs=5, batch_size=16):
    """
    Create and train a stock prediction model for a given stock symbol.
    
    Parameters:
        stock_symbol (str): The stock symbol to analyze
        years (int): Number of years of historical data to use
        time_steps (int): Number of time steps for LSTM sequences
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (model, scaler, df, predictions_df, model_path)
    """
    
    # Get stock data
    df = get_stock_data(stock_symbol, years)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, split = prepare_data(df, time_steps)
    
    # Build model
    input_shape = (time_steps, df.shape[1])
    model = build_lstm_model(input_shape)
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)
    
    # Generate predictions
    predictions_df = generate_predictions(model, X_test, df, scaler, time_steps, split)
    
    # Save model and scaler
    model_path, scaler_path = save_model_and_scaler(model, scaler, stock_symbol)
    
    return model, scaler, df, predictions_df, model_path

def create_news_analysis_df(stock_symbol, predictions_df, stock_name):
    """
    Create a DataFrame with news analysis for a stock.
    
    Parameters:
        stock_symbol (str): The stock symbol to analyze
        predictions_df (pandas.DataFrame): DataFrame with price predictions
        
    Returns:
        pandas.DataFrame: Combined DataFrame with news and predictions
    """
    # Fetch news articles
    articles = get_stock_news(stock_name)
    print(f"Fetched {len(articles)} news articles")
    
    # Process news data
    news_df = process_news_data(articles)
    
    # Combine news and predictions
    combined_df = combine_news_and_predictions(news_df, predictions_df)
    
    # Save results
    save_results_to_csv(combined_df, stock_symbol)
    
    return combined_df

def create_ds(stock_symbol="INFY.NS", years=5, time_steps=100, epochs=5, batch_size=16, save=True, stock_name="Infosys"):
    """
    Main function to run the stock prediction and news analysis.
    
    Parameters:
        stock_symbol (str): The stock symbol to analyze
        years (int): Number of years of historical data to use
        time_steps (int): Number of time steps for LSTM sequences
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        save (bool): Whether to save the results to CSV
        stock_name (str): Name of the stock for news search
        
    Returns:
        tuple: (model, scaler, combined_df, model_path)
    """
    # Check if model already exists
    model, scaler = load_model_and_scaler(stock_symbol)
    
    if model is None or scaler is None:
        print(f"Training new model for {stock_symbol}")
        # Create and train model
        model, scaler, df, predictions_df, model_path = create_stock_prediction_model(
            stock_symbol, years, time_steps, epochs, batch_size
        )
    else:
        print(f"Using existing model for {stock_symbol}")
        # Get stock data to generate predictions
        df = get_stock_data(stock_symbol, years)
        
        # Prepare data to get split index
        X_train, X_test, y_train, y_test, _, split = prepare_data(df, time_steps)
        
        # Generate predictions
        predictions_df = generate_predictions(model, X_test, df, scaler, time_steps, split)
        model_path = f"models/{stock_symbol.replace('.', '_')}_model.h5"
    
    # Create news analysis DataFrame if save is True
    if save:
        combined_df = create_news_analysis_df(stock_symbol, predictions_df, stock_name)
    else:
        combined_df = None
    
    return model, scaler, combined_df, model_path


if __name__ == "__main__":
    # Create dataset and save model for Infosys
    model, scaler, combined_df, model_path = create_ds("INFY.NS", stock_name="Infosys")
    
    # Example of using the prediction function for a specific date
    prediction = predict_future_price("INFY.NS", "2023-05-15")
    print(f"Predicted price: {prediction['predicted_price']}")