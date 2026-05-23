"""
Synthetic data generator for AI Field Copilot.

Region: Karnataka & Andhra Pradesh
Crops: chilli, maize, cotton

Produces:
  data/retailers.csv     - retailer master (id, name, district, lat, lon, crop_focus, ...)
  data/weather.csv       - daily weather per district (rainfall, temp, humidity)
  data/ndvi.csv          - weekly NDVI per district
  data/pest_reports.csv  - pest surveillance events (pink bollworm, fall armyworm, chilli thrips)
  data/sales.csv         - 12 months of monthly sales per retailer x product
  data/visits.csv        - past 90 days of rep visits with conversion flag
  data/training.csv      - feature table for ML (one row per retailer per day, 60 days)
"""

import csv
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference: districts with approximate centroids (lat, lon).
# ---------------------------------------------------------------------------
DISTRICTS = [
    # Karnataka
    ("Bellary",      "Karnataka",     15.139, 76.921),
    ("Raichur",      "Karnataka",     16.205, 77.355),
    ("Davangere",    "Karnataka",     14.466, 75.924),
    ("Haveri",       "Karnataka",     14.795, 75.404),
    ("Dharwad",      "Karnataka",     15.458, 75.008),
    ("Gadag",        "Karnataka",     15.430, 75.635),
    ("Belgaum",      "Karnataka",     15.852, 74.498),
    ("Bidar",        "Karnataka",     17.913, 77.530),
    # Andhra Pradesh
    ("Guntur",       "Andhra Pradesh", 16.306, 80.436),
    ("Krishna",      "Andhra Pradesh", 16.518, 80.620),
    ("Kurnool",      "Andhra Pradesh", 15.829, 78.037),
    ("Anantapur",    "Andhra Pradesh", 14.681, 77.600),
    ("Prakasam",     "Andhra Pradesh", 15.348, 79.560),
    ("Chittoor",     "Andhra Pradesh", 13.217, 79.101),
    ("Kadapa",       "Andhra Pradesh", 14.467, 78.824),
    ("Nellore",      "Andhra Pradesh", 14.443, 79.987),
]

CROPS = ["chilli", "maize", "cotton"]
PRODUCTS = {
    "chilli": ["Amistar Top Fungicide", "Karate Zeon Insecticide", "Cruiser Seed Treatment"],
    "maize":  ["Callisto Herbicide", "Force Insecticide", "NK Maize Hybrid Seed"],
    "cotton": ["Tihan Insecticide", "Polo Insecticide", "Cruiser Cotton Seed"],
}
PESTS = {
    "chilli": "chilli thrips",
    "maize":  "fall armyworm",
    "cotton": "pink bollworm",
}

# Simulation window: 90 days ending today.
TODAY = date(2026, 5, 22)
WINDOW_DAYS = 90
START = TODAY - timedelta(days=WINDOW_DAYS - 1)

# ---------------------------------------------------------------------------
# 1. Retailers
# ---------------------------------------------------------------------------
def jitter(value: float, amount: float = 0.25) -> float:
    return round(value + random.uniform(-amount, amount), 4)

retailer_rows = []
retailer_id = 1
for district, state, lat, lon in DISTRICTS:
    # 5-7 retailers per district -> ~95 retailers total.
    n = random.randint(5, 7)
    for _ in range(n):
        crop = random.choices(CROPS, weights=[3, 2, 4])[0]
        avg_monthly_sales = random.randint(40000, 350000)
        tier = "A" if avg_monthly_sales > 200000 else "B" if avg_monthly_sales > 100000 else "C"
        days_since_visit = random.randint(2, 80)
        retailer_rows.append({
            "retailer_id":          f"R{retailer_id:04d}",
            "retailer_name":        f"{district} Agri {retailer_id}",
            "district":             district,
            "state":                state,
            "lat":                  jitter(lat),
            "lon":                  jitter(lon),
            "primary_crop":         crop,
            "tier":                 tier,
            "avg_monthly_sales_inr": avg_monthly_sales,
            "days_since_last_visit": days_since_visit,
            "credit_score":          random.randint(50, 95),
        })
        retailer_id += 1

with (DATA_DIR / "retailers.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(retailer_rows[0].keys()))
    writer.writeheader()
    writer.writerows(retailer_rows)

print(f"retailers.csv  -> {len(retailer_rows)} rows")

# ---------------------------------------------------------------------------
# 2. Weather (daily per district)
# ---------------------------------------------------------------------------
weather_rows = []
for district, _, _, _ in DISTRICTS:
    # Each district gets a baseline + a pre-monsoon trend.
    base_temp = random.uniform(28, 34)
    base_humidity = random.uniform(45, 70)
    for i in range(WINDOW_DAYS):
        day = START + timedelta(days=i)
        # Slight upward humidity trend as monsoon approaches.
        humidity = max(30, min(95, base_humidity + i * 0.2 + random.gauss(0, 5)))
        temp = max(22, min(44, base_temp + math.sin(i / 7) * 2 + random.gauss(0, 1.5)))
        # Rainfall: zero most days, occasional spike.
        rainfall = 0.0
        if random.random() < 0.12 + i / 1000:
            rainfall = round(random.expovariate(1 / 8), 1)
        weather_rows.append({
            "date": day.isoformat(),
            "district": district,
            "rainfall_mm": rainfall,
            "temp_c": round(temp, 1),
            "humidity_pct": round(humidity, 1),
        })

with (DATA_DIR / "weather.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(weather_rows[0].keys()))
    writer.writeheader()
    writer.writerows(weather_rows)

print(f"weather.csv    -> {len(weather_rows)} rows")

# ---------------------------------------------------------------------------
# 3. NDVI (weekly per district)
# ---------------------------------------------------------------------------
ndvi_rows = []
for district, _, _, _ in DISTRICTS:
    base = random.uniform(0.35, 0.55)
    for week in range(WINDOW_DAYS // 7):
        day = START + timedelta(days=week * 7)
        # NDVI declines as crops mature pre-harvest, then drops.
        ndvi = max(0.1, min(0.9, base - week * 0.012 + random.gauss(0, 0.04)))
        ndvi_rows.append({
            "week_start": day.isoformat(),
            "district":   district,
            "ndvi":       round(ndvi, 3),
        })

with (DATA_DIR / "ndvi.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(ndvi_rows[0].keys()))
    writer.writeheader()
    writer.writerows(ndvi_rows)

print(f"ndvi.csv       -> {len(ndvi_rows)} rows")

# ---------------------------------------------------------------------------
# 4. Pest reports
# ---------------------------------------------------------------------------
pest_rows = []
# Each district has 0-4 pest events in the window, biased toward recent days.
for district, _, _, _ in DISTRICTS:
    n_events = random.randint(0, 4)
    for _ in range(n_events):
        day_offset = int(random.triangular(0, WINDOW_DAYS - 1, WINDOW_DAYS - 7))
        day = START + timedelta(days=day_offset)
        crop = random.choice(CROPS)
        severity = random.choice(["low", "medium", "medium", "high"])
        pest_rows.append({
            "report_date": day.isoformat(),
            "district":    district,
            "crop":        crop,
            "pest":        PESTS[crop],
            "severity":    severity,
            "fields_affected": random.randint(2, 40),
        })

with (DATA_DIR / "pest_reports.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(pest_rows[0].keys()))
    writer.writeheader()
    writer.writerows(pest_rows)

print(f"pest_reports.csv -> {len(pest_rows)} rows")

# ---------------------------------------------------------------------------
# 5. Monthly sales per retailer x product (last 12 months)
# ---------------------------------------------------------------------------
sales_rows = []
for r in retailer_rows:
    for prod in PRODUCTS[r["primary_crop"]]:
        for m in range(12):
            month_start = TODAY.replace(day=1) - timedelta(days=30 * m)
            month_key = month_start.strftime("%Y-%m")
            # Sales scaled by tier with seasonality.
            base = r["avg_monthly_sales_inr"] / len(PRODUCTS[r["primary_crop"]])
            season = 1 + 0.4 * math.sin((month_start.month + 3) / 12 * 2 * math.pi)
            noise = random.gauss(1, 0.15)
            units = max(0, int(base * season * noise / random.randint(800, 1500)))
            sales_rows.append({
                "month":       month_key,
                "retailer_id": r["retailer_id"],
                "product":     prod,
                "units_sold":  units,
            })

with (DATA_DIR / "sales.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(sales_rows[0].keys()))
    writer.writeheader()
    writer.writerows(sales_rows)

print(f"sales.csv      -> {len(sales_rows)} rows")

# ---------------------------------------------------------------------------
# 6. Visits log (past 90 days)
# ---------------------------------------------------------------------------
visit_rows = []
for r in retailer_rows:
    # 1-6 visits in the window depending on tier.
    n_visits = {"A": random.randint(4, 6), "B": random.randint(2, 4), "C": random.randint(1, 3)}[r["tier"]]
    used = set()
    for _ in range(n_visits):
        offset = random.randint(1, WINDOW_DAYS - 1)
        while offset in used:
            offset = random.randint(1, WINDOW_DAYS - 1)
        used.add(offset)
        day = START + timedelta(days=offset)
        converted = random.random() < {"A": 0.55, "B": 0.35, "C": 0.18}[r["tier"]]
        visit_rows.append({
            "visit_date":  day.isoformat(),
            "retailer_id": r["retailer_id"],
            "rep_id":      f"REP{random.randint(1, 8):02d}",
            "duration_min": random.randint(15, 90),
            "converted":   int(converted),
            "order_value_inr": random.randint(5000, 80000) if converted else 0,
        })

with (DATA_DIR / "visits.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(visit_rows[0].keys()))
    writer.writeheader()
    writer.writerows(visit_rows)

print(f"visits.csv     -> {len(visit_rows)} rows")

# ---------------------------------------------------------------------------
# 7. Training table (one row per retailer per day for last 60 days)
# Label = whether the retailer placed an order within the next 7 days.
# ---------------------------------------------------------------------------
# Build quick lookup for weather and pest signals.
weather_by_day_district = {}
for w in weather_rows:
    weather_by_day_district[(w["date"], w["district"])] = w

# Order events from visits.
orders_by_retailer = {}
for v in visit_rows:
    if v["converted"]:
        orders_by_retailer.setdefault(v["retailer_id"], []).append(date.fromisoformat(v["visit_date"]))

pest_by_district = {}
for p in pest_rows:
    pest_by_district.setdefault(p["district"], []).append(p)

training_rows = []
TRAIN_WINDOW = 60
for r in retailer_rows:
    for i in range(TRAIN_WINDOW):
        day = TODAY - timedelta(days=TRAIN_WINDOW - 1 - i)
        w = weather_by_day_district.get((day.isoformat(), r["district"]))
        if not w:
            continue

        # Recent 7-day rainfall and humidity averages.
        recent_rain = 0.0
        recent_humidity = 0.0
        days_counted = 0
        for d in range(7):
            ww = weather_by_day_district.get(((day - timedelta(days=d)).isoformat(), r["district"]))
            if ww:
                recent_rain += float(ww["rainfall_mm"])
                recent_humidity += float(ww["humidity_pct"])
                days_counted += 1
        recent_humidity /= max(days_counted, 1)

        # Pest pressure: count high-severity events in district in last 14 days.
        pest_pressure = sum(
            1 for p in pest_by_district.get(r["district"], [])
            if 0 <= (day - date.fromisoformat(p["report_date"])).days <= 14
            and p["severity"] in ("medium", "high")
        )

        # Days since last visit relative to this day.
        prior_visits = [date.fromisoformat(v["visit_date"]) for v in visit_rows
                        if v["retailer_id"] == r["retailer_id"]
                        and date.fromisoformat(v["visit_date"]) <= day]
        if prior_visits:
            days_since_visit = (day - max(prior_visits)).days
        else:
            days_since_visit = r["days_since_last_visit"]

        # Label: did an order happen in next 7 days?
        future_orders = [
            d for d in orders_by_retailer.get(r["retailer_id"], [])
            if 0 < (d - day).days <= 7
        ]
        label = 1 if future_orders else 0

        training_rows.append({
            "date":                  day.isoformat(),
            "retailer_id":           r["retailer_id"],
            "district":              r["district"],
            "tier":                  r["tier"],
            "primary_crop":          r["primary_crop"],
            "avg_monthly_sales_inr": r["avg_monthly_sales_inr"],
            "credit_score":          r["credit_score"],
            "days_since_visit":      days_since_visit,
            "recent_rain_7d_mm":     round(recent_rain, 1),
            "recent_humidity_pct":   round(recent_humidity, 1),
            "pest_pressure_14d":     pest_pressure,
            "label_order_next_7d":   label,
        })

with (DATA_DIR / "training.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(training_rows[0].keys()))
    writer.writeheader()
    writer.writerows(training_rows)

print(f"training.csv   -> {len(training_rows)} rows")
print(f"\nAll datasets written to: {DATA_DIR}")
