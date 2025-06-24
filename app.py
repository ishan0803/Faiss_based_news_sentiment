from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import datetime
from headline_analyzer import predict_delta_from_input_text
from predict_base_price import create_ds, predict_future_price, load_model_and_scaler

app = Flask(__name__)
CORS(app)
def load_data(file_path):
    """Load CSV data from file."""
    if not os.path.exists(file_path):
        return None
   
    try:
        df = pd.read_csv(file_path)
        # Remove duplicates if any
        df = df.drop_duplicates()
        # Check if required columns exist
        required_columns = ['Title', 'Percentage_Diff']
        if not all(col in df.columns for col in required_columns):
            return None
        return df
    except Exception as e:
        return None
    
@app.route('/stock-symbol', methods=['POST'])
def set_stock_symbol():
    data = request.get_json()
    stock_symbol = data.get('stock_symbol')
    
    if not stock_symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    
    return jsonify({'status': 'success', 'stock_symbol': stock_symbol})

@app.route('/create-dataset', methods=['POST'])
def create_dataset():
    data = request.get_json()
    stock_symbol = data.get('stock_symbol', 'INFY.NS')
    stock_name = data.get('stock_name', 'Infosys')
    years = int(data.get('years', 5))
    time_steps = int(data.get('time_steps', 100))
    epochs = int(data.get('epochs', 5))
    batch_size = int(data.get('batch_size', 16))
    
    try:
        # The updated create_ds function now returns a 4-tuple including the model path
        model, scaler, combined_df, model_path = create_ds(
            stock_symbol, years, time_steps, epochs, batch_size, save=True, stock_name=stock_name
        )
        
        # Return success response with model path information
        return jsonify({
            'status': 'success',
            'message': f'Dataset created for {stock_symbol}',
            'model_path': model_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-price', methods=['POST'])
def predict_price():
    data = request.get_json()
    stock_symbol = data.get('stock_symbol')
    headline = data.get('headline')
    target_date_str = data.get('target_date')  # Should be in format YYYY-MM-DD
    
    if not stock_symbol or not headline or not target_date_str:
        return jsonify({'error': 'Stock symbol, headline, and target date are required'}), 400
    
    # Convert date string to datetime object
    try:
        target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    
    # First try to load the existing model and scaler
    model, scaler = load_model_and_scaler(stock_symbol)
    
    # Check if we have dataset for this stock symbol, if not create one
    file_path = f"{stock_symbol.replace('.', '_')}_predictions.csv"
    
    if not os.path.exists(file_path) or model is None or scaler is None:
        try:
            # Create dataset and model if they don't exist
            print(f"Creating new dataset and model for {stock_symbol}")
            model, scaler, combined_df, _ = create_ds(stock_symbol, stock_name=stock_symbol.split('.')[0])
        except Exception as e:
            return jsonify({'error': f'Failed to create dataset: {str(e)}'}), 500
    else:
        print(f"Using existing model and dataset for {stock_symbol}")
    
    try:
        # Get the base price prediction without headline influence
        # Pass the loaded model and scaler to avoid reloading them
        base_prediction_result = predict_future_price(
            stock_symbol, target_date_str, model=model, scaler=scaler
        )
        
        if "error" in base_prediction_result:
            return jsonify({'error': base_prediction_result["error"]}), 500
            
        base_prediction = base_prediction_result["predicted_price"]
        
        # Load the sentiment model data
        df = load_data(file_path)
        if df is None:
            return jsonify({'error': 'Cannot load prediction data'}), 500
        
        # Predict difference based on headline
        predicted_diff = predict_delta_from_input_text(headline, df)
        
        # Calculate final prediction
        final_prediction = base_prediction - predicted_diff*base_prediction/100
        
        
        # Determine sentiment
        sentiment = "positive" if predicted_diff < 0 else "negative" if predicted_diff > 0 else "neutral"
        
        return jsonify({
            'stock_symbol': stock_symbol,
            'target_date': target_date_str,
            'headline': headline,
            'base_prediction': float(base_prediction),
            'headline_impact': float(predicted_diff*base_prediction/100),
            'final_prediction': float(final_prediction),
            'percentage_change': float(predicted_diff),
            'sentiment': sentiment
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/check-model-status', methods=['GET'])
def check_model_status():
    """
    Check if a model exists for a given stock symbol
    """
    stock_symbol = request.args.get('stock_symbol')
    
    if not stock_symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    
    model_path = f"models/{stock_symbol.replace('.', '_')}_model.h5"
    scaler_path = f"models/{stock_symbol.replace('.', '_')}_scaler.pkl"
    predictions_path = f"{stock_symbol.replace('.', '_')}_predictions.csv"
    
    return jsonify({
        'model_exists': os.path.exists(model_path),
        'scaler_exists': os.path.exists(scaler_path),
        'predictions_exist': os.path.exists(predictions_path)
    })

@app.route('/')
def home():
    return """
    <h1>Stock Price Prediction API</h1>
    <p>Use the following endpoints:</p>
    <ul>
        <li>/stock-symbol - POST to set a stock symbol</li>
        <li>/create-dataset - POST to create a dataset for a stock</li>
        <li>/predict-price - POST to predict price with headline analysis</li>
        <li>/check-model-status - GET to check if a model exists for a stock</li>
    </ul>
    """

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)