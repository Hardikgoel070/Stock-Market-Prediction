# Stock Price Prediction using Sentiment Analysis

## Project Overview
This project aims to predict Netflix (NFLX) stock prices using historical stock data and Twitter sentiment analysis. The goal is to assess whether incorporating sentiment data enhances prediction accuracy.

## Data Sources
- **Netflix Stock Data (2018–2022):** Includes daily Open, High, Low, Close, Adjusted Close prices, and trading volume.
- **Twitter Financial News Sentiment Data:** Contains aggregated daily sentiment scores (P_mean, P_sum) and tweet counts (twt_count).

## Files Overview
- `NFLX.csv`: Raw Netflix stock data.
- `Final_nflx_data_2018-2022.csv`: Cleaned and preprocessed data.
- `Scraping&Sentiment_analysis.ipynb`: Web scraping of financial news and sentiment analysis.
- `Lstm_Sentimental_analysis.ipynb`: Sentiment-based stock price prediction using LSTM.
- `Arima_pred.ipynb`: Stock price prediction using ARIMA.
- `Stock_model_with_twitter.pt`: Pretrained model from sentiment-based LSTM.
- `Final_pred_result.ipynb`: Final stock price predictions.

## Methodology
### 1. CNN-LSTM Model
- Hybrid deep learning model combining convolutional layers (for feature extraction) and LSTM layers (for capturing temporal dependencies).
- **Architecture:**
  - Two CNN layers with max pooling.
  - Two bidirectional LSTM layers with 256 hidden units.
  - Dense layers for regression output.
- **Training:** 50 epochs, batch size of 64, Adam optimizer, MSE loss function.

### 2. ARIMA Model
- Traditional time series forecasting approach.
- **Order:** ARIMA(2,1,3) with seasonal components.
- **Training:** Maximum likelihood estimation.

## Results
### ARIMA Model
- **Open Price Prediction:**
  - Without Twitter: Training MSE = 63.91; Testing MSE = 83.10
  - With Twitter: Training MSE = 59.88; Testing MSE = 83.10
- **Adjusted Close Price Prediction:**
  - Without Twitter: Training MSE = 158.32; Testing MSE = 212.77
  - With Twitter: Training MSE = 158.42; Testing MSE = 212.77

### CNN-LSTM Model
- **Open Price Prediction:**
  - Without Twitter: Training MSE = 1260.21; Testing MSE = 1924.76
  - With Twitter: Training MSE = 177.30; Testing MSE = 2729.50
- **Adjusted Close Price Prediction:**
  - Without Twitter: Training MSE = 1280.14; Testing MSE = 2061.25
  - With Twitter: Training MSE = 303.78; Testing MSE = 2775.36

## Key Observations
1. **ARIMA models** performed more consistently across training and testing sets.
2. **CNN-LSTM models** captured complex patterns but suffered from overfitting.
3. Sentiment data had a marginal impact on **Open price prediction** but minimal effect on **Adjusted Close price prediction**.

## Future Improvements
- Advanced sentiment analysis (e.g., aspect-based analysis).
- Integration of news sentiment or other social media platforms.
- Ensemble methods combining statistical and deep learning models.


