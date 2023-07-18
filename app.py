from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint
from pmdarima.arima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import RandomizedSearchCV
import io
import warnings
import base64
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    input_name = request.form['input_name']
    input_csv = request.files['input_csv']
    input_time = request.form['input_time']
    input_series = request.form['input_series']
    input_forecasting_steps = int(request.form['input_forecasting_steps'])
    seasonality = request.form.get('seasonality', False)
    m = int(request.form.get('m', '0')) if request.form.get('m') != '' else 0
    optimize_models = request.form.get('optimize_models', False)

    # Save the uploaded CSV file to a temporary location
    csv_file_path = 'temp.csv'
    input_csv.save(csv_file_path)

    # Call the program function with the appropriate arguments
    forecast_data = program(csv_file_path, input_time, input_series, seasonality, m, input_forecasting_steps, optimize_models)

    # Remove the temporary CSV file
    os.remove(csv_file_path)

    return render_template('forecast.html', forecast_data=forecast_data)

def extract_time_series(csv_file, m=None, time_column=None, series_column=None):
    """
    Extracts the time series from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        m (int): Seasonality period.
        time_column (str): Name of the time column in the CSV file.
        series_column (str): Name of the series column in the CSV file.

    Returns:
        pandas.DataFrame: Time series DataFrame.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Determine if time column is provided
    if time_column in df.columns:
        # Convert the time column to datetime format
        df[time_column] = pd.to_datetime(df[time_column])

        # Extract the time and series value columns
        time_series = df[[time_column, series_column]].copy()
        time_series.set_index(time_column, inplace=True)
    else:
        # Use the default index as the time column
        time_series = df[[series_column]].copy()
        time_series.index = pd.to_datetime(df.index)

    return time_series

def plot_forecast(time_series, forecasted_values, forecast_steps):
    """
    Plots the original data and forecasted values.

    Args:
        time_series (pandas.Series): Original time series data.
        forecasted_values (pandas.Series): Forecasted values.
        forecast_steps (int): Number of forecast steps.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_series.index[:-forecast_steps], time_series[:-forecast_steps], label='Original Data')
    ax.plot(forecasted_values.index, forecasted_values, color='red', label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Forecast')
    ax.legend()

    # Convert the plot to an image buffer
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    # Convert the image buffer to a base64 string
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

    return encoded_image

def train_and_forecast_auto_arima(time_series, seasonal=False, m=None, forecast_steps=None, series_column=None, optimize_models=False):
    """
    Trains and forecasts using auto_arima.

    Args:
        time_series (pandas.DataFrame): Time series data.
        seasonal (bool): Whether to include seasonality.
        m (int): Seasonality period.
        forecast_steps (int): Number of forecast steps.
        series_column (str): Name of the series column.

    Returns:
        pandas.Series: Forecasted values.
    """
    if forecast_steps is None:
        forecast_steps = len(time_series)

    # Split the data into training and testing sets
    train_data = time_series[:-forecast_steps]
    test_data = time_series[-forecast_steps:]

    try:
        if optimize_models:
            # Find the best ARIMA model using auto_arima
            print("Model optimization: ARIMA")
            model = auto_arima(train_data, seasonal=seasonal, m=m, suppress_warnings=True)

            # Train the ARIMA model with the best parameters
            print("Training ARIMA model...")
            fitted_model = model.fit(train_data)

            # Forecast using the trained model
            print("Forecasting with ARIMA model...")
            forecasted_values = fitted_model.predict(n_periods=forecast_steps)
        else:
            # Train and forecast using ARIMA with predetermined parameters
            print("Training ARIMA model...")
            model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(0, 1, 1, m))
            fitted_model = model.fit(disp=False)

            # Forecast using the trained model
            print("Forecasting with ARIMA model...")
            forecasted_values = fitted_model.get_forecast(steps=forecast_steps).predicted_mean

        # Create the forecast index based on the date range of the test data
        forecast_index = pd.date_range(start=test_data.index[0], periods=forecast_steps, freq='MS')

        # Extend the index to include the forecasted period
        extended_index = test_data.index.union(forecast_index)

        # Plot the original data and the forecasted values
        image_buffer = plot_forecast(time_series, pd.Series(forecasted_values, index=extended_index), forecast_steps)

        # Save forecasted values to CSV
        forecast_data = pd.Series(forecasted_values, index=extended_index, name='Forecast')
        forecast_csv_buffer = io.StringIO()
        forecast_data.to_csv(forecast_csv_buffer, index=True, header=True)
        forecast_csv_buffer.seek(0)

        return forecasted_values, image_buffer, forecast_csv_buffer, forecast_steps

    except Exception as e:
        print(f"Error in ARIMA model training and forecasting: {str(e)}")
        return None, None, None, None

def train_and_forecast_auto_sarima(time_series, seasonal=False, m=None, forecast_steps=None, series_column=None, optimize_models=False):
    """
    Trains and forecasts using auto_sarima.

    Args:
        time_series (pandas.DataFrame): Time series data.
        seasonal (bool): Whether to include seasonality.
        m (int): Seasonality period.
        forecast_steps (int): Number of forecast steps.
        series_column (str): Name of the series column.

    Returns:
        pandas.Series: Forecasted values.
    """
    # Split the data into training and testing sets
    train_data = time_series[:-forecast_steps]
    test_data = time_series[-forecast_steps:]

    try:
        if optimize_models:
            # Find the best SARIMA model using auto_arima
            print("Model optimization: SARIMA")
            model = auto_arima(train_data, seasonal=seasonal, m=m)

            # Train the SARIMA model with the best parameters
            print("Training SARIMA model...")
            order = model.order
            seasonal_order = model.seasonal_order
            fitted_model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            fitted_model = fitted_model.fit()

            # Forecast using the trained model
            print("Forecasting with SARIMA model...")
            forecasted_values = fitted_model.get_forecast(steps=forecast_steps).predicted_mean
        else:
            # Train and forecast using SARIMA with predetermined parameters
            print("Training SARIMA model...")
            model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(0, 1, 1, m))
            fitted_model = model.fit(disp=False)

            # Forecast using the trained model
            print("Forecasting with SARIMA model...")
            forecasted_values = fitted_model.get_forecast(steps=forecast_steps).predicted_mean

        # Create the forecast index based on the date range of the test data
        forecast_index = pd.date_range(start=test_data.index[0], periods=forecast_steps, freq='MS')

        # Extend the index to include the forecasted period
        extended_index = test_data.index.union(forecast_index)

        # Plot the original data and the forecasted values
        image_buffer = plot_forecast(time_series, pd.Series(forecasted_values, index=extended_index), forecast_steps)

        # Save forecasted values to CSV
        forecast_data = pd.Series(forecasted_values, index=extended_index, name='Forecast')
        forecast_csv_buffer = io.StringIO()
        forecast_data.to_csv(forecast_csv_buffer, index=True, header=True)
        forecast_csv_buffer.seek(0)

        return forecasted_values, image_buffer, forecast_csv_buffer, forecast_steps

    except Exception as e:
        print(f"Error in SARIMA model training and forecasting: {str(e)}")
        return None, None, None, None

def train_and_forecast_exponential_smoothing(time_series, seasonal=False, m=None, forecast_steps=None, series_column=None, optimize_models=False):
    """
    Trains and forecasts using Exponential Smoothing.

    Args:
        time_series (pandas.DataFrame): Time series data.
        seasonal (bool): Whether to include seasonality.
        m (int): Seasonality period.
        forecast_steps (int): Number of forecast steps.
        series_column (str): Name of the series column.

    Returns:
        pandas.Series: Forecasted values.
    """
    # Split the data into training and testing sets
    train_data = time_series[:-forecast_steps]
    test_data = time_series[-forecast_steps:]

    try:
        if optimize_models:
            # Find the best SARIMA model using auto_arima
            print("Model optimization: Exponential Smoothing")
            model = auto_arima(train_data, seasonal=seasonal, m=m, suppress_warnings=True)

            # Extract the model parameters
            order = model.order
            seasonal_order = model.seasonal_order

            # Train the exponential smoothing model with the optimized parameters
            print("Training Exponential Smoothing model...")
            model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=m)
            fitted_model = model.fit()
        else:
            # Train and forecast using Exponential Smoothing with predetermined parameters
            print("Training Exponential Smoothing model...")
            fitted_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=m).fit()

        # Forecast using the trained model
        print("Forecasting with Exponential Smoothing model...")
        forecasted_values = fitted_model.forecast(forecast_steps)

        # Create the forecast index based on the date range of the test data
        forecast_index = pd.date_range(start=test_data.index[0], periods=forecast_steps, freq='MS')

        # Extend the index to include the forecasted period
        extended_index = test_data.index.union(forecast_index)

        # Plot the original data and the forecasted values
        image_buffer = plot_forecast(time_series, pd.Series(forecasted_values, index=extended_index), forecast_steps)

        # Save forecasted values to CSV
        forecast_data = pd.Series(forecasted_values, index=extended_index, name='Forecast')
        forecast_csv_buffer = io.StringIO()
        forecast_data.to_csv(forecast_csv_buffer, index=True, header=True)
        forecast_csv_buffer.seek(0)

        return forecasted_values, image_buffer, forecast_csv_buffer, forecast_steps

    except Exception as e:
        print(f"Error in Exponential Smoothing model training and forecasting: {str(e)}")
        return None, None, None, None

def train_and_forecast_random_forest(time_series, forecast_steps=None, series_column=None, optimize_models=False):
    """
    Trains and forecasts using Random Forest.

    Args:
        time_series (pandas.DataFrame): Time series data.
        forecast_steps (int): Number of forecast steps.
        series_column (str): Name of the series column.

    Returns:
        pandas.Series: Forecasted values.
    """
    window_size = forecast_steps

    # Split the data into training and testing sets
    train_data = time_series[:-forecast_steps]
    test_data = time_series[-forecast_steps:]

    try:
        # Prepare the lagged features for training
        X_train, y_train = [], []
        for i in range(window_size, len(train_data)):
            X_train.append(train_data[series_column].iloc[i-window_size:i])
            y_train.append(train_data[series_column].iloc[i])
        X_train, y_train = np.array(X_train), np.array(y_train)

        if optimize_models:
            # Define the parameter distribution for RandomizedSearchCV
            param_dist = {
                'n_estimators': randint(100, 500),  # Number of trees in the forest
                'max_depth': [None] + [np.random.randint(5, 20) for _ in range(5)],  # Maximum depth of each tree
                'min_samples_split': randint(2, 20),  # Minimum number of samples required to split an internal node
                'min_samples_leaf': randint(1, 10)  # Minimum number of samples required to be at a leaf node
            }

            # Create the Random Forest regressor
            rf_model = RandomForestRegressor(random_state=0)

            # Perform RandomizedSearchCV to find the best hyperparameters
            random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=10, cv=3, random_state=0)
            random_search.fit(X_train, y_train)

            # Get the best model from RandomizedSearchCV
            best_model = random_search.best_estimator_

            # Prepare the lagged features for forecasting
            X_test = np.array(test_data[series_column].iloc[-window_size:]).reshape(1, -1)
        else:
            # Train and forecast using Random Forest with predetermined parameters
            best_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=0)
            best_model.fit(X_train, y_train)

            # Prepare the lagged features for forecasting
            X_test = np.array(test_data[series_column].iloc[-window_size:]).reshape(1, -1)

        # Forecast using the trained model
        print("Forecasting with Random Forest model...")
        y_pred = best_model.predict(X_test)

        # Create the forecast index based on the date range of the test data
        forecast_index = pd.date_range(start=test_data.index[0], periods=1, freq='MS')

        # Extend the index to include the forecasted period
        extended_index = test_data.index.union(forecast_index)

        # Reshape the forecasted values to match the shape of test data
        forecast_values = np.concatenate([test_data[series_column].values, y_pred])

        # Check if there are issues with different lengths
        if len(forecast_values) > len(test_data):
            forecast_values = forecast_values[:-1]  # Exclude the last element

        # Plot the original data and the forecasted values
        image_buffer = plot_forecast(time_series, pd.Series(forecast_values, index=extended_index), forecast_steps)

        # Save forecasted values to CSV
        forecast_data = pd.Series(forecast_values, index=extended_index, name='Forecast')
        forecast_csv_buffer = io.StringIO()
        forecast_data.to_csv(forecast_csv_buffer, index=True, header=True)
        forecast_csv_buffer.seek(0)

        return forecast_values, image_buffer, forecast_csv_buffer, forecast_steps

    except Exception as e:
        print(f"Error in Random Forest model training and forecasting: {str(e)}")
        return None, None, None, None

def program(input_csv, time_column, series_column, seasonality, m, forecast_steps, optimize_models=False):
    """
    Main program to extract time series and perform multiple forecasts.

    Args:
        input_csv (str): Path to the input CSV file.
        time_column (str): Name of the time column in the CSV file.
        series_column (str): Name of the series column in the CSV file.
        seasonality (bool): Whether to include seasonality.
        m (int): Seasonality period.
        forecast_steps (int): Number of forecast steps.
        optimize_models (bool): Whether to optimize the models or use predetermined parameters.
    """
    if m <= 1:
        m = 2

    time_series = extract_time_series(input_csv, time_column=time_column, series_column=series_column)

    if time_series is not None:
        arima_forecast, arima_image_buffer, arima_csv_buffer, arima_steps = train_and_forecast_auto_arima(
            time_series, seasonal=seasonality, m=m, forecast_steps=forecast_steps, series_column=series_column,
            optimize_models=optimize_models
        )
        sarima_forecast, sarima_image_buffer, sarima_csv_buffer, sarima_steps = train_and_forecast_auto_sarima(
            time_series, seasonal=seasonality, m=m, forecast_steps=forecast_steps, series_column=series_column,
            optimize_models=optimize_models
        )
        es_forecast, es_image_buffer, es_csv_buffer, es_steps = train_and_forecast_exponential_smoothing(
            time_series, seasonal=seasonality, m=m, forecast_steps=forecast_steps, series_column=series_column,
            optimize_models=optimize_models
        )
        rf_forecast, rf_image_buffer, rf_csv_buffer, rf_steps = train_and_forecast_random_forest(
            time_series, forecast_steps=forecast_steps, series_column=series_column,
            optimize_models=optimize_models
        )

        forecast_data = {
            'arima_forecast': arima_forecast,
            'arima_steps': arima_steps,
            'sarima_forecast': sarima_forecast,
            'sarima_steps': sarima_steps,
            'es_forecast': es_forecast,
            'es_steps': es_steps,
            'rf_forecast': rf_forecast,
            'rf_steps': rf_steps,
            'arima_plot': arima_image_buffer,
            'arima_csv': arima_csv_buffer,
            'sarima_plot': sarima_image_buffer,
            'sarima_csv': sarima_csv_buffer,
            'es_plot': es_image_buffer,
            'es_csv': es_csv_buffer,
            'rf_plot': rf_image_buffer,
            'rf_csv': rf_csv_buffer
        }

        return forecast_data
    else:
        return None


if __name__ == '__main__':
    app.run(debug=True)
