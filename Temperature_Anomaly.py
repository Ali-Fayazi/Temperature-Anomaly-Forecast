# Temperature Anomaly Forecast (EnergySense Demo)

import pandas as pd
from prophet import Prophet 
from matplotlib import pyplot as plt

# Load the raw data file (assumes file is in the current directory for portability)
file_name = "flat-ui__data-Wed Nov 12 2025.csv"
df_climate = pd.read_csv(file_name)

## 1. Data Cleaning and Preparation

# 1.1: Select necessary columns (Date, Mean) and filter invalid rows
df_climate = df_climate[['Date' , 'Mean']]                       
df_climate = df_climate[df_climate['Date'].str.contains('-')]      # Filter rows not in Year-Month format

# 1.2: Convert 'Date' column to Datetime format
df_climate['Date'] = pd.to_datetime(df_climate['Date'])

# 1.3: Rename columns according to Prophet model requirements ('ds' for date, 'y' for value)
df_climate = df_climate.rename(columns= {'Date' : 'ds' , 'Mean' : 'y'})

# 1.4: Display the cleaned data summary
print("\n--- Final output of step1 ---")
print("\nData Head: ")
print(df_climate.head())
print("\nData Info: ")
print(df_climate.info())

## 2. Modeling, EDA, and Evaluation

# 2.1: Initialize the Prophet Model and manage Outliers
model = Prophet(
    yearly_seasonality=True,
    daily_seasonality=False,
    weekly_seasonality=False,
    changepoint_prior_scale=0.01    # Lower scale makes the model more robust against outliers/noise
)

# 2.2: Train the model on the entire historical dataset
model.fit(df_climate)

# 2.3: Create a future dataframe for a 12-year forecast (144 months)
future = model.make_future_dataframe(periods=144 , freq='M')

# 2.4: Generate the forecast
forecast = model.predict(future)

# 2.5: Exploratory Data Analysis (EDA): Visualize model components (Trend, Seasonality)
fig_comp = model.plot_components(forecast)
fig_comp.savefig('component_analysis.png')  
plt.close(fig_comp)   

# 2.6: Model Evaluation - Mean Absolute Error (MAE)
    # Calculate MAE on historical data for performance evaluation
df_performance = pd.merge(df_climate, forecast[['ds' , 'yhat']] , on='ds' , how='inner')
df_performance['absolute_error'] = abs(df_performance['y'] - df_performance['yhat'])
mea = df_performance['absolute_error'].mean()
print("\n--- Model Performance ---")
print(f"Mean Absolute Error (MAE) on historical data: {mea:.4f}")

# 2.7: Final Visualization (Full Forecast Plot)
fig = model.plot(forecast)
plt.title("Global Temperature Anomaly Forecast (12 Years)")
plt.xlabel("Data")
plt.ylabel("Temperature Anomaly (°C)")
fig.savefig('final_forecast.png') 
plt.close(fig)  

# 2.8: Display the last 5 rows of the generated forecast
print("\n---Final output of step2 (Forecast)---")
print("\nLast 5 rows of prediction (yhat is the forecast value):")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

## 3. Data Storage for Streamlit Deployment

# 3.1: Define the output file name
forecast_file_name = "climate_forecast_data.csv"

# 3.2: Save the forecast data (ds, yhat, confidence intervals) to the current directory for Streamlit access
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_file_name , index=False)

# 3.3: Confirmation of file saving
print("\n---Final output of step3---")
print(f"The final forecast file has been saved. : {forecast_file_name}")



