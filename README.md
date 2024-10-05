# Real-Time-Data-Stream-Anomaly-Detection-using-Z-Score-and-Isolation-Forest
Project Title: Efficient Data Stream Anomaly Detection

Overview:
This project focuses on developing a real-time anomaly detection system for streaming data, combining two powerful methods: Z-Score based statistical anomaly detection and machine learning-based Isolation Forest. The system monitors data streams such as stock prices and system metrics, detecting anomalies that indicate sudden spikes, dips, or unusual behavior.

Objectives:
Real-time Anomaly Detection: Continuously monitor incoming data and flag abnormal behavior in real-time using sliding window techniques and machine learning.
Adaptive Learning: Implement adaptive anomaly detection using Isolation Forest, allowing the model to learn from recent data and adapt to changing trends in the stream.
Multi-Metric Monitoring: Simultaneously track stock prices (e.g., IBM) and system performance metrics like CPU and memory usage, detecting anomalies in both streams.
Email Alerts: Automatically send email notifications when anomalies are detected, providing instant alerts for critical deviations.
Data Logging: Log all detected anomalies in a CSV file for later analysis, including details about the anomaly, severity, and classification.


Key Features:

Algorithm Selection for Anomaly Detection:
Z-Score Based Detection: Detects statistical anomalies by calculating the Z-Score of the latest data points and flagging outliers based on a predefined threshold.
Isolation Forest Based Detection: Utilizes a machine learning model (Isolation Forest) to identify outliers based on rarity, continuously retraining on the latest data for adaptive anomaly detection.
The user can easily switch between these two detection methods for flexible and context-appropriate anomaly detection.

Data Stream Simulation:
Simulates real-time data streams by fetching real-time stock price data (e.g., IBM) using the Alpha Vantage API.
Also monitors system performance metrics such as CPU and memory usage using the psutil library, detecting performance anomalies in system behavior.

Real-Time Anomaly Detection:
Z-Score Method: Detects anomalies by monitoring a sliding window of data, calculating the Z-Score of the current value, and flagging it as an anomaly if the Z-Score exceeds a set threshold.
Isolation Forest Method: Learns the distribution of data using a sliding window and detects anomalies when the model identifies outliers (-1).
Both methods are applied in real-time, adapting to changes in the data stream.

Optimization for Speed and Efficiency:
Implements optimized data handling with sliding windows, reducing memory usage and improving computation speed.Continuously retrains the Isolation Forest model on recent data to ensure accurate detection with minimal lag.

Real-Time Visualization:
Uses Matplotlib to visualize the data stream in real-time, highlighting detected anomalies with different markers on the plot.
The graph updates dynamically, allowing users to see trends and anomalies as they happen.

Email Notification System:
Sends email alerts when anomalies are detected, providing instant notifications for critical deviations.Can be configured to log anomaly details in a CSV file for post-analysis.

Data Logging:
Logs detailed anomaly information, including the time of occurrence, value, Z-Score, anomaly type (spike or dip), and severity level (mild, moderate, or extreme).

Technologies Used:
Python 3.x: Core programming language used for real-time data streaming, anomaly detection, and system automation.
Alpha Vantage API: Provides real-time stock price data for financial anomaly detection.
Scikit-Learn (Isolation Forest): Implements adaptive anomaly detection using machine learning.
Psutil: Monitors system performance, tracking CPU and memory usage.
Matplotlib: Visualizes the data stream and detected anomalies with dynamic, real-time plotting.
SmtpLib: Sends email alerts when anomalies are detected, facilitating real-time notifications.
Numpy: Performs statistical calculations for Z-Score-based anomaly detection.

Usage:
Users can specify the window size and anomaly threshold, allowing them to control the sensitivity of the anomaly detection.
The system defaults to using the Isolation Forest model for adaptive learning but can switch to the Z-Score method for statistical analysis.
Anomalies are logged, and email alerts are sent automatically.

Future Enhancements:
Expanded Data Stream Simulation: Include additional data sources such as weather, IoT sensor data, or social media sentiment analysis.
Advanced Visualization: Create more interactive, multi-metric dashboards for comprehensive real-time monitoring.
Improved Alert Mechanisms: Add SMS or other third-party notification integrations like Slack for more versatile alerting systems.
Algorithm Expansion: Incorporate additional anomaly detection algorithms (e.g., One-Class SVM or DBSCAN) for different types of data streams and use cases.

This project offers a comprehensive, flexible system for real-time anomaly detection, combining statistical and machine learning methods. It can be applied to a wide range of domains, including financial monitoring, system performance tracking, and IoT applications.
