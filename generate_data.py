"""
Generates a synthetic quick-commerce last-mile delivery dataset.

The data models realistic delivery-time drivers (distance, time-of-day,
traffic, weather, rider experience) with noise, so it behaves like a
real-world logistics dataset for training/evaluating regression models.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 52000

# Bengaluru quick-commerce delivery radius (~5 km hub catchment)
LAT_MIN, LAT_MAX = 12.960, 12.995
LON_MIN, LON_MAX = 77.580, 77.615

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

pickup_lat = np.random.uniform(LAT_MIN, LAT_MAX, N)
pickup_lon = np.random.uniform(LON_MIN, LON_MAX, N)
drop_lat = np.random.uniform(LAT_MIN, LAT_MAX, N)
drop_lon = np.random.uniform(LON_MIN, LON_MAX, N)
distance_km = haversine_km(pickup_lat, pickup_lon, drop_lat, drop_lon)

order_hour = np.random.randint(0, 24, N)
day_of_week = np.random.randint(0, 7, N)  # 0=Mon
is_weekend = (day_of_week >= 5).astype(int)

# peak hours: lunch (12-14) and dinner (19-22)
is_peak = np.isin(order_hour, [12, 13, 19, 20, 21]).astype(int)

weather = np.random.choice(["clear", "cloudy", "rain", "storm"], N, p=[0.55, 0.25, 0.15, 0.05])
weather_delay = {"clear": 0, "cloudy": 0.8, "rain": 3.5, "storm": 7.0}
weather_delay_arr = np.array([weather_delay[w] for w in weather])

traffic = np.random.choice(["low", "medium", "high"], N, p=[0.3, 0.45, 0.25])
traffic_delay = {"low": 0, "medium": 2.2, "high": 5.5}
traffic_delay_arr = np.array([traffic_delay[t] for t in traffic])

rider_experience_days = np.random.exponential(scale=180, size=N).clip(1, 1500)
rider_speed_factor = 1.0 - np.clip(rider_experience_days / 3000, 0, 0.25)  # experienced riders slightly faster

num_items = np.random.poisson(lam=3, size=N).clip(1, 15)

# ground-truth delivery time model (minutes)
base_time = 4.0                                   # order handoff / prep buffer
distance_time = distance_km * 4.4 * rider_speed_factor
peak_delay = is_peak * np.random.uniform(2.0, 4.5, N)
item_delay = (num_items - 1) * 0.35
noise = np.random.normal(0, 1.8, N)

delivery_time_min = (
    base_time + distance_time + weather_delay_arr + traffic_delay_arr
    + peak_delay + item_delay + noise
).clip(3, None)

df = pd.DataFrame({
    "order_id": [f"ORD{100000+i}" for i in range(N)],
    "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
    "drop_lat": drop_lat, "drop_lon": drop_lon,
    "distance_km": distance_km.round(3),
    "order_hour": order_hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "is_peak_hour": is_peak,
    "weather": weather,
    "traffic_density": traffic,
    "rider_experience_days": rider_experience_days.round(1),
    "num_items": num_items,
    "delivery_time_min": delivery_time_min.round(2),
})

df.to_csv("data/delivery_data.csv", index=False)
print(f"Wrote {len(df):,} rows to data/delivery_data.csv")
print(df.head())
