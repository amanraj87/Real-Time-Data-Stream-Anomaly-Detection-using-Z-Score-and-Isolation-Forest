import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import csv
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.ensemble import IsolationForest
import requests
import psutil

# Replace with your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'YOUR API KEY'
# Replace with your desired stock symbol(GOOGL/AAPL/IBM etc.)
STOCK_SYMBOL = 'AAPL'

# Function to fetch real-time stock data from Alpha Vantage


def fetch_real_time_data():
    """
    Fetches the latest stock price for the specified stock symbol using the Alpha Vantage API.
    Returns the latest stock price or None if the fetch fails.
    """
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={
            STOCK_SYMBOL}&interval=1min&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        # Debugging: Print the raw response data
        print(data)

        if 'Time Series (1min)' in data:
            latest_time = next(iter(data['Time Series (1min)']))
            latest_price = float(
                data['Time Series (1min)'][latest_time]['1. open'])
            return latest_price
        else:
            print("Invalid response structure. Please check your API key and symbol.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage: {e}")
        return None

# Function to fetch CPU and memory usage


def fetch_system_metrics():
    """
    Fetches the current CPU and memory usage of the system.
    Returns a tuple containing CPU usage percentage and memory usage percentage.
    """
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    return cpu_usage, memory_info.percent

# Adaptive Learning with Isolation Forest


def detect_anomaly_isolation_forest(data, model=None, log_anomalies=False, log_file="adaptive_anomalies_log.csv"):
    """
    Detects anomalies in the provided data using an Isolation Forest model.
    If an anomaly is detected, it logs the anomaly details if logging is enabled.
    Returns a list of indices of detected anomalies.
    """
    anomalies = []
    if model is None:
        return anomalies

    current_value = np.array(data[-1]).reshape(1, -1)
    is_anomaly = model.predict(current_value)

    if is_anomaly == -1:  # -1 indicates an anomaly
        anomaly_index = len(data) - 1
        anomalies.append(anomaly_index)
        if log_anomalies:
            log_anomaly(
                anomaly_index, current_value[0][0], 'Adaptive Learning (Isolation Forest)', log_file)

    return anomalies

# Z-Score based anomaly detection


def detect_anomaly(data, window_size=50, threshold=3, log_anomalies=False, log_file="anomalies_log.csv"):
    """
    Detects anomalies in the data using a Z-Score based method.
    If an anomaly is detected, it logs the anomaly details if logging is enabled.
    Returns a list of tuples containing anomaly indices and details.
    """
    anomalies = []
    if len(data) < window_size:
        return anomalies

    window_data = data[-window_size:]
    mean = np.mean(window_data)
    std_dev = np.std(window_data)

    if std_dev == 0:
        return anomalies

    current_value = data[-1]
    z_score = (current_value - mean) / std_dev

    if abs(z_score) > threshold:
        anomaly_index = len(data) - 1
        severity = classify_anomaly(z_score)
        anomaly_type = "spike" if z_score > 0 else "dip"
        anomalies.append((anomaly_index, current_value,
                         z_score, anomaly_type, severity))

        if log_anomalies:
            log_anomaly(anomaly_index, current_value, z_score,
                        anomaly_type, severity, log_file)

    return anomalies

# Function to classify anomalies based on the z-score


def classify_anomaly(z_score):
    """
    Classifies the severity of an anomaly based on its Z-Score.
    Returns a string indicating the severity level ('extreme', 'moderate', 'mild').
    """
    if abs(z_score) > 5:
        return "extreme"
    elif abs(z_score) > 3:
        return "moderate"
    else:
        return "mild"

# Log the anomaly details into a CSV file


def log_anomaly(index, value, z_score, anomaly_type, severity, log_file="anomalies_log.csv"):
    """
    Logs the details of detected anomalies into a specified CSV file.
    Records the timestamp, index, value, Z-Score, anomaly type, and severity.
    """
    timestamp = datetime.datetime.now().isoformat()
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [timestamp, index, value, z_score, anomaly_type, severity])
    print(f"Anomaly logged: Index {index}, Value {
          value}, Z-Score {z_score}, Type {anomaly_type}, Severity {severity}")

# Email notification system for detected anomalies


def send_email_alert(subject, body, to_email="abc@email.com"):
    """
    Sends an email alert for detected anomalies.
    Takes the subject and body of the email, and optionally the recipient email.
    """
    from_email = "def@email.com"
    password = "password"  # Use environment variables in production

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Function to simulate real-time data streaming, anomaly detection, and adaptive learning
def stream_data_and_detect_anomalies(window_size=50, threshold=3, log_anomalies=False, plot_real_time=True, method="adaptive"):
    """
    Continuously streams stock price data and system metrics, detects anomalies,
    and visualizes the data in real-time.
    Uses either Z-Score or Isolation Forest for anomaly detection based on the specified method.
    """
    # A buffer for holding the latest data points
    buffer = deque(maxlen=window_size)
    detected_anomalies = []  # Stores the indices of detected anomalies
    live_data = []  # Stores data for real-time plotting

    plt.ion()  # Turn on interactive mode for live plotting
    fig, ax = plt.subplots()

    # Initialize the adaptive learning model (Isolation Forest)
    isolation_forest_model = IsolationForest(contamination=0.02)
    model_fitted = False  # Flag to check if the model has been fitted

    while True:
        # Fetch stock price
        stock_price = fetch_real_time_data()
        if stock_price is not None:
            buffer.append(stock_price)
            live_data.append(stock_price)

            # Fit the model if we have enough data
            if len(live_data) >= window_size:
                isolation_forest_model.fit(
                    np.array(live_data[-window_size:]).reshape(-1, 1))
                model_fitted = True  # Set the flag to True once fitted

            # Detect anomalies using the selected method
            anomalies = []
            if method == "zscore":
                anomalies = detect_anomaly(
                    live_data, window_size, threshold, log_anomalies)
            elif method == "adaptive" and model_fitted:
                anomalies = detect_anomaly_isolation_forest(
                    live_data, model=isolation_forest_model, log_anomalies=log_anomalies)

            # Send email alert if anomalies are detected
            if anomalies:
                for anomaly in anomalies:
                    send_email_alert(
                        subject="Anomaly Detected!",
                        body=f"Anomaly detected at index {
                            anomaly} with value {live_data[anomaly]}."
                    )

            # Fetch system metrics
            cpu_usage, memory_usage = fetch_system_metrics()
            buffer.append(cpu_usage)  # Append CPU usage
            buffer.append(memory_usage)  # Append memory usage
            live_data.append(cpu_usage)
            live_data.append(memory_usage)

            # Real-time plotting
            ax.clear()
            ax.plot(live_data, label='Data Stream (Stock Prices and System Metrics)')
            ax.scatter(detected_anomalies, [
                       live_data[i] for i in detected_anomalies], color='red', label='Anomalies')
            ax.set_title(
                f'Real-time Data Stream with {method.upper()} Anomalies')
            ax.set_xlabel('Time')
            ax.set_ylabel('Values')
            ax.legend()
            plt.pause(0.01)  # Pause for real-time effect

        time.sleep(60)  # Adjust the frequency of data fetching as needed

    plt.ioff()  # Turn off interactive mode
    plt.show()


# Main function
if __name__ == '__main__':
    # Step 1: Take user input for window size and threshold only
    window_size = 50
    threshold = 3

    # Use default values for log_file and method
    log_file = "anomalies_log.csv"  # Default log file
    method = "adaptive"  # Default method

    # Step 2: Stream data and detect anomalies in real-time
    stream_data_and_detect_anomalies(
        window_size, threshold, True, plot_real_time=True, method=method)
