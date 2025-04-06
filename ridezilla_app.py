import os
import json
import time
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import matplotlib.pyplot as plt

# ✅ Streamlit Config
st.set_page_config(page_title="Ridezilla Dashboard", layout="wide")
st.title("Ridezilla: Strava Performance Tracker")

# ✅ Constants
CLIENT_ID = 154543
CLIENT_SECRET = 'e95ea1374cf55a20af635cf39c09fbec0cd15229'
TOKEN_FILE = 'strava_tokens.json'

# ✅ Refresh Access Token
def refresh_access_token():
    with open(TOKEN_FILE) as f:
        tokens = json.load(f)

    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'refresh_token',
        'refresh_token': tokens['refresh_token']
    }

    response = requests.post('https://www.strava.com/oauth/token', data=payload)
    if response.status_code != 200:
        st.error("Failed to refresh token.")
        st.stop()

    new_tokens = response.json()
    with open(TOKEN_FILE, 'w') as f:
        json.dump(new_tokens, f)

    return new_tokens['access_token']

# ✅ Get Access Token
def get_access_token():
    if not os.path.exists(TOKEN_FILE):
        st.error("Token file not found. Please run exchange_token.py first.")
        st.stop()

    with open(TOKEN_FILE) as f:
        tokens = json.load(f)

    # ⏳ Check if token is expired
    if tokens.get("expires_at", 0) < time.time():
        return refresh_access_token()

    return tokens["access_token"]

# ✅ Show Token Expiry in Sidebar (Optional)
try:
    with open(TOKEN_FILE) as f:
        tokens = json.load(f)
    expiry = datetime.fromtimestamp(tokens["expires_at"])
    st.sidebar.info(f" Token expires at: {expiry.strftime('%Y-%m-%d %H:%M:%S')}")
except:
    st.sidebar.warning("Token info unavailable")

# ✅ Fetch Activities with Cache and Refresh TTL
@st.cache_data(ttl=300)
def fetch_activities():
    access_token = get_access_token()
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'per_page': 200, 'page': 1}
    response = requests.get("https://www.strava.com/api/v3/athlete/activities", headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch activities. Status code: {response.status_code}")
        return []

# ✅ Fetch and Check
activities = fetch_activities()
if not activities:
    st.warning("No activities found.")
    st.stop()

# ✅ Process Data
data = []
for activity in activities:
    data.append({
        'name': activity['name'],
        'distance_km': round(activity['distance'] / 1000, 2),
        'moving_time_min': round(activity['moving_time'] / 60, 2),
        'elapsed_time_min': round(activity['elapsed_time'] / 60, 2),
        'total_elevation_gain': activity['total_elevation_gain'],
        'type': activity['type'],
        'start_date': activity['start_date_local'],
        'average_speed_kmh': round(activity['average_speed'] * 3.6, 2),
        'max_speed_kmh': round(activity['max_speed'] * 3.6, 2)
    })

df = pd.DataFrame(data)

# ✅ Display Table
st.subheader("Activity Summary")
st.dataframe(df)

df.head()
df.to_csv('ridezilla_data.csv', index=False)
df.info()
df.isnull().sum()
# Drop HR columns
df = df.drop(columns=[col for col in ['average_heartrate', 'max_heartrate'] if col in df.columns])

# Convert start_date to datetime
if 'start_date' in df.columns:
    df['start_date'] = pd.to_datetime(df['start_date'])
else:
    st.warning("'start_date' not found in data. Please check your API response.")


df.info()
df = df[df['type'].str.lower() == 'ride'].reset_index(drop=True)
print(f" Filtered dataset to only include cycling rides – total: {len(df)} rides.")
# Week & Month
df['week'] = df['start_date'].dt.isocalendar().week
df['month'] = df['start_date'].dt.month

# Pace = time / distance
df['pace_min_per_km'] = df['moving_time_min'] / df['distance_km']

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Prepare Data
df['start_date'] = pd.to_datetime(df['start_date'])
df['month'] = df['start_date'].dt.strftime('%B %Y')  # e.g., 'March 2025'
df['pace_min_per_km'] = df['moving_time_min'] / df['distance_km']

import seaborn as sns

# Convert start_date to datetime
df['start_date'] = pd.to_datetime(df['start_date'])

# Add month column
df['month'] = df['start_date'].dt.strftime('%B')  # 'January', 'February', etc.
df['pace_min_per_km'] = round(df['moving_time_min'] / df['distance_km'], 2)

# Monthly Avg Speed Barplot
st.subheader("Average Speed by Month")
monthly_speed = df.groupby('month')['average_speed_kmh'].mean().reset_index()
fig1 = plt.figure(figsize=(10, 5))
sns.barplot(x='month', y='average_speed_kmh', data=monthly_speed, palette='coolwarm')
plt.title("Average Speed by Month")
plt.xticks(rotation=45)
plt.ylabel("Speed (km/h)")
plt.tight_layout()
st.pyplot(fig1)

# Weekly Consistency Score
st.subheader("Weekly Ride Consistency")

df['week'] = df['start_date'].dt.isocalendar().week
weekly_counts = df['week'].value_counts().sort_index()
total_weeks = weekly_counts.count()
consistent_weeks = (weekly_counts >= 2).sum()
consistency_score = round((consistent_weeks / total_weeks) * 100, 1)

fig2 = plt.figure(figsize=(10, 4))
plt.bar(weekly_counts.index, weekly_counts.values, color='skyblue')
plt.axhline(2, color='green', linestyle='--', label='Goal: 2+ rides/week')
plt.title(f'Weekly Ride Counts – Consistency Score: {consistency_score}/100')
plt.xlabel('Week Number')
plt.ylabel('Number of Rides')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig2)

# Distance vs Pace (Recent 2 Months)
st.subheader("Distance vs Pace (Last 2 Months)")
last_two_months = df['month'].unique()[-2:]
df_recent = df[df['month'].isin(last_two_months)]

fig3 = plt.figure(figsize=(8, 5))
sns.scatterplot(x='distance_km', y='pace_min_per_km', hue='month', data=df_recent, palette='viridis')
plt.title("Distance vs Pace (Recent 2 Months)")
plt.xlabel("Distance (km)")
plt.ylabel("Pace (min/km)")
plt.tight_layout()
st.pyplot(fig3)

# Elevation vs Speed (Recent 2 Months)
st.subheader("Elevation vs Speed (Last 2 Months)")
fig4 = plt.figure(figsize=(8, 5))
sns.scatterplot(x='total_elevation_gain', y='average_speed_kmh', hue='month', data=df_recent, palette='magma')
plt.title("Elevation Gain vs Speed (Recent 2 Months)")
plt.xlabel("Total Elevation Gain (m)")
plt.ylabel("Average Speed (km/h)")
plt.tight_layout()
st.pyplot(fig4)



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample Data (you will load your actual data here)
# df = pd.read_csv("your_data.csv")

# --- Monthly Summary & Suggestions ---
st.title("Monthly Recap and Suggestions")

# Ensure 'month' column exists and is in proper format
df['start_date'] = pd.to_datetime(df['start_date'])  # Ensure start_date is datetime
df['month'] = df['start_date'].dt.strftime('%B %Y')

# Calculate monthly stats
monthly_stats = df.groupby('month')[['distance_km', 'average_speed_kmh', 'moving_time_min']].mean().round(2)

# Check that at least 2 months of data are present
if len(monthly_stats) >= 2:
    this_month, last_month = monthly_stats.index[-1], monthly_stats.index[-2]

    summary = f"""
    **Monthly Recap — {this_month}**
    
    Compared to {last_month}:
    - Distance: {monthly_stats.loc[this_month, 'distance_km']} km vs {monthly_stats.loc[last_month, 'distance_km']} km
    - Avg Speed: {monthly_stats.loc[this_month, 'average_speed_kmh']} km/h vs {monthly_stats.loc[last_month, 'average_speed_kmh']} km/h
    - Avg Duration: {monthly_stats.loc[this_month, 'moving_time_min']} min vs {monthly_stats.loc[last_month, 'moving_time_min']} min

    **Suggestions**:
    """
    
    # Suggestions Logic
    if monthly_stats.loc[this_month, 'average_speed_kmh'] < monthly_stats.loc[last_month, 'average_speed_kmh']:
        summary += "- Try interval training to boost pacing.\n"
    else:
        summary += "- Keep up the strong pacing! Consider longer steady rides.\n"

    if monthly_stats.loc[this_month, 'distance_km'] < monthly_stats.loc[last_month, 'distance_km']:
        summary += "- Increase weekly ride lengths to build endurance.\n"
    else:
        summary += "- Excellent consistency in distance. Maintain this streak!\n"

    if monthly_stats.loc[this_month, 'moving_time_min'] < monthly_stats.loc[last_month, 'moving_time_min']:
        summary += "- Ride a bit longer — shorter durations might be limiting progress.\n"
    else:
        summary += "- Great ride duration! Keep pushing boundaries.\n"

    st.markdown(summary)
else:
    st.write("Not enough monthly data to compare this month with the previous month.")

# --- Underperformance Detection ---
st.title("Underperformance Detection")

# Overall performance metrics
overall_avg_speed = df['average_speed_kmh'].mean()
df['pace_min_per_km'] = df['moving_time_min'] / df['distance_km']
overall_avg_pace = df['pace_min_per_km'].mean()

# Flag underperforming rides
df['underperformed'] = (df['average_speed_kmh'] < overall_avg_speed) | (df['pace_min_per_km'] > overall_avg_pace)
underperform_count = df['underperformed'].sum()

st.write(f"You underperformed in {underperform_count} out of {len(df)} rides.")
st.write("These rides either had below-average speed or above-average pace.")

# View underperforming rides
underperform_rides = df[df['underperformed']][['name', 'start_date', 'average_speed_kmh', 'pace_min_per_km']]

st.dataframe(underperform_rides)

# --- Plot Speed Over Time ---
st.title("Speed Trend – Underperformance Detection")

plt.figure(figsize=(12, 5))
plt.plot(df['start_date'], df['average_speed_kmh'], label='Avg Speed', marker='o', color='blue')
plt.axhline(overall_avg_speed, color='red', linestyle='--', label='Overall Avg Speed')
plt.xticks(rotation=45)
plt.title('Speed Trend')
plt.xlabel('Date')
plt.ylabel('Speed (km/h)')
plt.legend()
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Ensure this import is present
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load your data (this will depend on how you're loading the dataset)
# For now, assuming `df` is already loaded and pre-processed.
# df = pd.read_csv('your_data.csv')  # Uncomment and modify based on your dataset

# Title of the Streamlit app
st.title('Ride Performance Analysis and Habit Tracker')

# Ride Performance Predictor
st.header("Ride Performance Predictor")

# Drop non-cycling rides
df = df[df['type'] == 'Ride']

# Define input features (X) and target variable (y)
X = df[['distance_km', 'total_elevation_gain', 'moving_time_min']]
y = df['average_speed_kmh']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model evaluation
st.subheader("Model Evaluation:")
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Predict future ride speed (example: 25 km ride with 300 m elevation gain)
future_ride = pd.DataFrame({
    'distance_km': [25],
    'total_elevation_gain': [300],
    'moving_time_min': [80]
})

predicted_speed = model.predict(future_ride)[0]
st.write(f"Expected average speed: {predicted_speed:.2f} km/h for the next ride.")

# Ride Habit Heatmap
st.header("Ride Habit Heatmap")

# Ensure start_date is datetime
df['start_date'] = pd.to_datetime(df['start_date'])

# Extract weekday and week number
df['weekday'] = df['start_date'].dt.day_name()
df['week'] = df['start_date'].dt.isocalendar().week
df['month'] = df['start_date'].dt.strftime('%B %Y')  # Optional: for grouping

# Filter just 'Ride' activities
df_rides = df[df['type'] == 'Ride']

# Create the pivot table
heatmap_data = df_rides.pivot_table(index='week', columns='weekday', values='name', aggfunc='count').fillna(0)

# Reindex the columns safely
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(columns=ordered_days, fill_value=0)

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap='YlGnBu', linewidths=0.5, linecolor='gray')
plt.title("Ride Habit Heatmap – Weekly Discipline Tracker")
plt.xlabel("Day of the Week")
plt.ylabel("Week Number")
plt.tight_layout()
st.pyplot(plt)

# Ride Performance Classifier
st.header("Ride Performance Classifier")

# Select features for clustering
features = df[['distance_km', 'moving_time_min', 'average_speed_kmh', 'total_elevation_gain']]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['performance_cluster'] = kmeans.fit_predict(scaled_features)

# Map cluster labels to performance levels based on average speed
cluster_avg_speed = df.groupby('performance_cluster')['average_speed_kmh'].mean()
sorted_clusters = cluster_avg_speed.sort_values(ascending=False).index.tolist()

performance_map = {sorted_clusters[0]: 'Good', sorted_clusters[1]: 'Average', sorted_clusters[2]: 'Poor'}
df['performance_label'] = df['performance_cluster'].map(performance_map)

# Visualize cluster distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='performance_label', data=df, order=['Good', 'Average', 'Poor'])
plt.title('Ride Performance Classification')
plt.xlabel("Performance")
plt.ylabel("Number of Rides")
plt.tight_layout()
st.pyplot(plt)
