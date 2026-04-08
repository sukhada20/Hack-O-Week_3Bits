import pandas as pd  # pandas is used to handle tables like Excel or CSV files

# Load the energy data
energy = pd.read_csv("data/energy.csv")

# Load the calendar data
calendar = pd.read_csv("data/calendar.csv")

# Check the first 5 rows of each dataset
print("Energy Data:")
print(energy.head())

print("\nCalendar Data:")
print(calendar.head())


# Convert UNIX timestamp to datetime
energy["timestamp"] = pd.to_datetime(energy["timestamp"], unit="s")  # 's' means seconds

# Extract only the date (without time)
energy["Date"] = energy["timestamp"].dt.date

# Make sure calendar dates are in the same format
calendar["Date"] = pd.to_datetime(calendar["Date"]).dt.date

# Check the first 5 rows after conversion
print("\nEnergy Data with Date:")
print(energy.head())

print("\nCalendar Data with Date:")
print(calendar.head())

daily_energy = energy.groupby("Date")["power"].sum().reset_index()
daily_energy.rename(columns={"power": "energy_kWh"}, inplace=True)  # rename for consistency

# Check the first 5 rows
print("\nDaily Energy Usage:")
print(daily_energy.head())


# Merge daily energy with calendar
daily_data = pd.merge(daily_energy, calendar, on="Date", how="left")

# Check the first 5 rows after merging
print("\nDaily Energy with Calendar Info:")
print(daily_data.head())


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set Date as index (time series requires Date as index)
daily_data.set_index("Date", inplace=True)

# Apply exponential smoothing
model = ExponentialSmoothing(
    daily_data["energy_kWh"], 
    trend="add",       # additive trend
    seasonal="add",    # additive seasonality
    seasonal_periods=7 # weekly pattern
).fit()

# Add the fitted forecast to our table
daily_data["forecast"] = model.fittedvalues

# Check the first 5 rows
print("\nDaily Energy with Forecast:")
print(daily_data.head())



import streamlit as st
import plotly.graph_objects as go

st.title("📚 Library Energy During Exams")

# Show a table of the last 5 days
st.subheader("Last some Days Energy Usage")
st.dataframe(daily_data.tail())

# Show latest day's actual vs forecast
# Improved Gauge for Latest Day
last_day = daily_data.index.max()
last_energy = daily_data.loc[last_day, "energy_kWh"]
forecast_energy = daily_data.loc[last_day, "forecast"]
activity_type = daily_data.loc[last_day, "activity"]  # H, L, or other

fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=last_energy,
    title={'text': f"Energy Usage on {last_day} ({'Exam Day' if activity_type=='H' else 'Normal Day'})",
           'font': {'size': 18}},
    delta={'reference': forecast_energy, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge={
        'axis': {'range': [0, daily_data["energy_kWh"].max() * 1.2], 'tickwidth': 2, 'tickcolor': "darkblue"},
        'bar': {'color': "blue"},
        'bgcolor': "lightgray",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, forecast_energy*0.8], 'color': "green", 'name': "Below Forecast"},
            {'range': [forecast_energy*0.8, forecast_energy*1.2], 'color': "yellow", 'name': "Around Forecast"},
            {'range': [forecast_energy*1.2, daily_data["energy_kWh"].max()*1.2], 'color': "red", 'name': "Above Forecast"},
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': forecast_energy
        }
    }
))

st.plotly_chart(fig, use_container_width=True)

# Optional: Line chart of actual vs forecast
st.subheader("Energy Trend (Actual vs Forecast)")
st.line_chart(daily_data[["energy_kWh", "forecast"]])
