import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from geopy.geocoders import Nominatim


# API Keys
OPENWEATHER_API_KEY = "ae3a09bacee8313ac35ba004aa1cde60"
WAQI_API_KEY = "35abf66a94555b3b17e3c93700d9f492d5a102c9"


# Fetch Weather Data
def fetch_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/forecast"
    params = {
        'lat': city['lat'],
        'lon': city['lon'],
        'appid': api_key,
        'units': 'metric'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch weather data: {response.status_code}")


# Fetch WAQI Data
def fetch_waqi_data(api_key, city):
    url = f"https://api.waqi.info/feed/geo:{city['lat']};{city['lon']}/"
    params = {'token': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch air quality data: {response.status_code}")


# Safe conversion to numeric
def safe_numeric(value):
    try:
        return pd.to_numeric(value)
    except:
        return None


# Process forecast data
def process_forecast(forecast_data):
    processed_data = []
    for item in forecast_data:
        date = pd.to_datetime(item['dt'], unit='s')
        temp = safe_numeric(item['main']['temp'])
        humidity = safe_numeric(item['main']['humidity'])
        wind_speed = safe_numeric(item['wind']['speed'])
       
        if all(v is not None for v in [temp, humidity, wind_speed]):
            processed_data.append({
                'date': date,
                'temperature': temp,
                'humidity': humidity,
                'wind_speed': wind_speed
            })
   
    return pd.DataFrame(processed_data).set_index('date')


# Preprocess Weather Data
def preprocess_weather_data(weather_data):
    current_weather = weather_data['list'][0]
    current_data = {
        'temperature': safe_numeric(current_weather['main']['temp']),
        'humidity': safe_numeric(current_weather['main']['humidity']),
        'wind_speed': safe_numeric(current_weather['wind']['speed']),
        'weather_condition': current_weather['weather'][0]['main'],
    }
   
    forecast_df = process_forecast(weather_data['list'])
   
    return current_data, forecast_df


# Preprocess Air Quality Data
def preprocess_air_quality_data(air_quality_data):
    aqi_data = air_quality_data['data']
    current_aqi = aqi_data['aqi']
    records = [{'datetime': pd.Timestamp.now(), 'aqi': current_aqi}]
    if 'forecast' in aqi_data and 'daily' in aqi_data['forecast']:
        for pollutant in ['pm25', 'pm10', 'o3']:
            if pollutant in aqi_data['forecast']['daily']:
                for entry in aqi_data['forecast']['daily'][pollutant]:
                    record = {
                        'datetime': pd.to_datetime(entry['day']),
                        pollutant: safe_numeric(entry['avg'])
                    }
                    records.append(record)
    df = pd.DataFrame(records)
    df.set_index('datetime', inplace=True)
    return df


# Get City Coordinates
def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(city_name)
    return {'lat': location.latitude, 'lon': location.longitude}


# Streamlit App
st.title("Weather and Air Quality Prediction App")


city_name = st.text_input("Enter City Name", "New York")


if st.button("Get Current Data and Predictions"):
    try:
        city = get_city_coordinates(city_name)
        weather_data = fetch_weather_data(OPENWEATHER_API_KEY, city)
        air_quality_data = fetch_waqi_data(WAQI_API_KEY, city)
       
        current_weather, weather_forecast = preprocess_weather_data(weather_data)
        air_quality_df = preprocess_air_quality_data(air_quality_data)
       
        # Display current weather data
        st.subheader("Current Weather Data")
        st.write(f"Temperature: {current_weather['temperature']:.2f}°C")
        st.write(f"Humidity: {current_weather['humidity']}%")
        st.write(f"Wind Speed: {current_weather['wind_speed']} m/s")
        st.write(f"Weather Condition: {current_weather['weather_condition']}")


        # Display current AQI data
        st.subheader("Current Air Quality Data")
        current_aqi = air_quality_df['aqi'].iloc[0]
        st.write(f"Air Quality Index (AQI): {current_aqi}")


        # AQI inference
        if current_aqi <= 50:
            st.write("Air quality is Good")
        elif current_aqi <= 100:
            st.write("Air quality is Moderate")
        elif current_aqi <= 150:
            st.write("Air quality is Unhealthy for Sensitive Groups")
        elif current_aqi <= 200:
            st.write("Air quality is Unhealthy")
        elif current_aqi <= 300:
            st.write("Air quality is Very Unhealthy")
        else:
            st.write("Air quality is Hazardous")


        # 5-day prediction
        st.subheader("5-Day Weather Forecast")
        five_day_forecast = weather_forecast.resample('D').mean()
        five_day_forecast = five_day_forecast.iloc[:5]  # Get only the next 5 days


        fig = px.line(five_day_forecast, y='temperature', title='5-Day Temperature Forecast')
        st.plotly_chart(fig)


        fig = px.line(five_day_forecast, y='humidity', title='5-Day Humidity Forecast')
        st.plotly_chart(fig)


        fig = px.line(five_day_forecast, y='wind_speed', title='5-Day Wind Speed Forecast')
        st.plotly_chart(fig)


        # Air Quality Forecast
        st.subheader("Air Quality Forecast")
        aqi_forecast = air_quality_df.resample('D').mean()
        aqi_forecast = aqi_forecast.iloc[:5]  # Get only the next 5 days


        for column in ['pm25', 'pm10', 'o3']:
            if column in aqi_forecast.columns:
                fig = px.line(aqi_forecast, y=column, title=f'5-Day {column.upper()} Forecast')
                st.plotly_chart(fig)


        # Display Map
        st.subheader("Map")
        st.map(pd.DataFrame([city], columns=['lat', 'lon']))


    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your API keys and try again.")
        import traceback
        st.error(traceback.format_exc())



