import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model and scaler
def load_models():
    with open('regression.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

# Create the Streamlit app
def main():
    st.title('Taxi Fare Prediction System')
    st.write('Enter the trip details to predict the fare')

    # Create input fields for features
    col1, col2 = st.columns(2)
    
    with col1:
        trip_distance = st.number_input('Trip Distance (km)', min_value=0.0, max_value=100.0, value=5.0)
        time_of_day = st.selectbox('Time of Day', options=[0, 1, 2, 3])  # 0: Morning, 1: Afternoon, 2: Evening, 3: Night
        day_of_week = st.selectbox('Day of Week', options=list(range(7)))  # 0: Monday to 6: Sunday
        passenger_count = st.number_input('Passenger Count', min_value=1, max_value=6, value=1)
        traffic_conditions = st.selectbox('Traffic Conditions', options=[0, 1, 2])  # 0: Light, 1: Medium, 2: Heavy
    
    with col2:
        weather = st.selectbox('Weather', options=[0, 1, 2, 3])  # 0: Clear, 1: Cloudy, 2: Rain, 3: Snow
        base_fare = st.number_input('Base Fare', min_value=0.0, max_value=100.0, value=2.5)
        per_km_rate = st.number_input('Per Km Rate', min_value=0.0, max_value=10.0, value=1.5)
        per_minute_rate = st.number_input('Per Minute Rate', min_value=0.0, max_value=5.0, value=0.35)
        trip_duration = st.number_input('Trip Duration (Minutes)', min_value=1, max_value=180, value=15)

    # Add helpful tooltips
    st.sidebar.markdown("""
    ### Feature Information
    - **Time of Day**: 0=Morning, 1=Afternoon, 2=Evening, 3=Night
    - **Day of Week**: 0=Monday through 6=Sunday
    - **Traffic Conditions**: 0=Light, 1=Medium, 2=Heavy
    - **Weather**: 0=Clear, 1=Cloudy, 2=Rain, 3=Snow
    """)

    # Create a button to make prediction
    if st.button('Predict Fare'):
        try:
            # Load the models
            model, scaler = load_models()

            # Create a dataframe with the input values
            input_data = pd.DataFrame([[
                trip_distance, time_of_day, day_of_week, passenger_count,
                traffic_conditions, weather, base_fare, per_km_rate,
                per_minute_rate, trip_duration
            ]], columns=['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 
                        'Passenger_Count', 'Traffic_Conditions', 'Weather',
                        'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate',
                        'Trip_Duration_Minutes'])

            # Scale the input data
            scaled_data = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(scaled_data)

            # Display the prediction
            st.success(f'Predicted Fare: ${prediction[0]:.2f}')

            # Display feature importance if available
            if hasattr(model, 'coef_'):
                st.subheader('Feature Importance')
                importance = pd.DataFrame({
                    'Feature': input_data.columns,
                    'Importance': abs(model.coef_)
                }).sort_values('Importance', ascending=False)
                st.bar_chart(importance.set_index('Feature'))

        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
            st.error('Please make sure your model and scaler files are in the same directory as this script.')

if __name__ == '__main__':
    main()