"""
AI Field Copilot - Streamlit prototype.

Run:
    streamlit run app/copilot.py

Region : Karnataka & Andhra Pradesh
Crops  : chilli, maize, cotton
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "model" / "visit_model.joblib"
TODAY = date(2026, 5, 22)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Field Copilot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0a7e3d"  # Syngenta-ish green
ACCENT = "#f6a623"

st.markdown(
    f"""
    <style>
      .stApp {{ background: #f7faf7; }}
      .priority-card {{
          padding: 14px 18px; border-radius: 10px; background: white;
          border-left: 6px solid {PRIMARY}; margin-bottom: 10px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      }}
      .priority-card h4 {{ margin: 0 0 4px 0; color: {PRIMARY}; }}
      .reason-chip {{
          display: inline-block; background: #e8f4ec; color: #0a5e2d;
          padding: 3px 9px; border-radius: 12px; font-size: 12px;
          margin-right: 6px; margin-top: 4px;
      }}
      .chip-warn {{ background: #fff3d6; color: #8a5a00; }}
      .chip-danger {{ background: #fde0e0; color: #a02020; }}
      .metric-box {{
          background: white; padding: 14px; border-radius: 8px;
          text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
      }}
      .metric-value {{ font-size: 28px; font-weight: 700; color: {PRIMARY}; }}
      .metric-label {{ font-size: 12px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data + model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    retailers = pd.read_csv(DATA / "retailers.csv")
    weather   = pd.read_csv(DATA / "weather.csv", parse_dates=["date"])
    ndvi      = pd.read_csv(DATA / "ndvi.csv", parse_dates=["week_start"])
    pests     = pd.read_csv(DATA / "pest_reports.csv", parse_dates=["report_date"])
    sales     = pd.read_csv(DATA / "sales.csv")
    visits    = pd.read_csv(DATA / "visits.csv", parse_dates=["visit_date"])
    return retailers, weather, ndvi, pests, sales, visits


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


retailers, weather, ndvi, pests, sales, visits = load_data()
bundle = load_model()
model = bundle["model"]
FEATURES = bundle["features"]
MODEL_AUC = bundle["auc"]


# ---------------------------------------------------------------------------
# Feature builder (today's snapshot)
# ---------------------------------------------------------------------------
def build_today_features(today: date) -> pd.DataFrame:
    df = retailers.copy()

    # Recent 7-day rain & humidity per district.
    w_recent = weather[weather["date"] >= pd.Timestamp(today) - pd.Timedelta(days=7)]
    w_agg = w_recent.groupby("district").agg(
        recent_rain_7d_mm=("rainfall_mm", "sum"),
        recent_humidity_pct=("humidity_pct", "mean"),
    ).reset_index()
    df = df.merge(w_agg, on="district", how="left")

    # Pest pressure (medium/high events in last 14 days).
    p_recent = pests[
        (pests["report_date"] >= pd.Timestamp(today) - pd.Timedelta(days=14)) &
        (pests["severity"].isin(["medium", "high"]))
    ]
    p_agg = p_recent.groupby("district").size().reset_index(name="pest_pressure_14d")
    df = df.merge(p_agg, on="district", how="left")
    df["pest_pressure_14d"] = df["pest_pressure_14d"].fillna(0).astype(int)

    # Days since visit using current snapshot column.
    df["days_since_visit"] = df["days_since_last_visit"]

    # Latest NDVI per district.
    ndvi_latest = ndvi.groupby("district").last().reset_index()
    df = df.merge(ndvi_latest[["district", "ndvi"]], on="district", how="left")
    df["ndvi"] = df["ndvi"].fillna(0.3)

    # Encodings.
    df["tier_score"] = df["tier"].map({"A": 3, "B": 2, "C": 1})
    df["crop_chilli"] = (df["primary_crop"] == "chilli").astype(int)
    df["crop_maize"]  = (df["primary_crop"] == "maize").astype(int)
    df["crop_cotton"] = (df["primary_crop"] == "cotton").astype(int)

    return df


def score_retailers(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["priority_score"] = (proba * 100).round(1)
    return df


def reason_chips(row: pd.Series) -> str:
    chips = []
    if row["days_since_visit"] >= 30:
        chips.append(f'<span class="reason-chip chip-warn">⏱ {int(row["days_since_visit"])}d since visit</span>')
    if row["pest_pressure_14d"] >= 2:
        chips.append(f'<span class="reason-chip chip-danger">🐛 high pest activity</span>')
    elif row["pest_pressure_14d"] >= 1:
        chips.append(f'<span class="reason-chip chip-warn">🐛 pest activity</span>')
    if row["recent_rain_7d_mm"] >= 25:
        chips.append(f'<span class="reason-chip">🌧 {row["recent_rain_7d_mm"]:.0f}mm rain (7d)</span>')
    if row["tier"] == "A":
        chips.append('<span class="reason-chip">⭐ Tier A account</span>')
    if row["avg_monthly_sales_inr"] >= 200000:
        chips.append(f'<span class="reason-chip">💰 ₹{row["avg_monthly_sales_inr"]/1000:.0f}k/mo</span>')
    if row.get("ndvi", 1) < 0.3:
        chips.append(f'<span class="reason-chip chip-danger">🌱 crop stress (NDVI {row["ndvi"]:.2f})</span>')
    if not chips:
        chips.append('<span class="reason-chip">Routine check-in</span>')
    return " ".join(chips)


# ---------------------------------------------------------------------------
# Sidebar (rep settings)
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"### 🌾 AI Field Copilot")
st.sidebar.caption("Karnataka & Andhra Pradesh")

rep_id = st.sidebar.selectbox("Rep ID", [f"REP{i:02d}" for i in range(1, 9)], index=0)

states = ["All"] + sorted(retailers["state"].unique().tolist())
state_pick = st.sidebar.selectbox("State", states)

if state_pick == "All":
    districts = ["All"] + sorted(retailers["district"].unique().tolist())
else:
    districts = ["All"] + sorted(retailers[retailers["state"] == state_pick]["district"].unique().tolist())
district_pick = st.sidebar.selectbox("District", districts)

crop_pick = st.sidebar.selectbox("Crop focus", ["All", "chilli", "maize", "cotton"])
top_n = st.sidebar.slider("Visits to surface", 5, 25, 10)

st.sidebar.markdown("---")
language = st.sidebar.selectbox("Language", ["English", "हिन्दी (Hindi)", "ಕನ್ನಡ (Kannada)", "తెలుగు (Telugu)"])
st.sidebar.caption(f"_Demo: language toggle wired but UI shown in English._")

st.sidebar.markdown("---")
st.sidebar.metric("Model ROC-AUC", f"{MODEL_AUC:.3f}")
st.sidebar.caption(f"As of {TODAY.isoformat()}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Today's Plan", "🗺 Risk Heatmap", "🏪 Retailer 360", "📊 Insights"])

# Scoring snapshot
scored = score_retailers(build_today_features(TODAY))

# Filter by sidebar.
filt = scored.copy()
if state_pick != "All":
    filt = filt[filt["state"] == state_pick]
if district_pick != "All":
    filt = filt[filt["district"] == district_pick]
if crop_pick != "All":
    filt = filt[filt["primary_crop"] == crop_pick]


# ---------------------------------------------------------------------------
# Tab 1: Today's Plan
# ---------------------------------------------------------------------------
with tab1:
    st.markdown(f"## Good morning, **{rep_id}** 👋")
    st.caption(f"Here's your prioritized plan for {TODAY.strftime('%A, %d %b %Y')}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><div class="metric-value">{len(filt)}</div><div class="metric-label">Retailers in scope</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="metric-value">{top_n}</div><div class="metric-label">Suggested visits</div></div>', unsafe_allow_html=True)
    high_pest = int((filt["pest_pressure_14d"] >= 2).sum())
    c3.markdown(f'<div class="metric-box"><div class="metric-value">{high_pest}</div><div class="metric-label">High-pest districts</div></div>', unsafe_allow_html=True)
    overdue = int((filt["days_since_visit"] >= 30).sum())
    c4.markdown(f'<div class="metric-box"><div class="metric-value">{overdue}</div><div class="metric-label">Overdue (30d+)</div></div>', unsafe_allow_html=True)

    st.markdown("### Top suggested visits")

    top = filt.sort_values("priority_score", ascending=False).head(top_n).reset_index(drop=True)

    for _, row in top.iterrows():
        st.markdown(
            f"""
            <div class="priority-card">
              <h4>{row['retailer_name']} <span style="font-size:13px;color:#888;font-weight:400;">· {row['district']}, {row['state']}</span>
                <span style="float:right;background:{PRIMARY};color:white;padding:3px 10px;border-radius:12px;font-size:13px;">Score {row['priority_score']:.0f}</span>
              </h4>
              <div style="font-size:13px;color:#555;">
                {row['primary_crop'].title()} · Tier {row['tier']} · ₹{row['avg_monthly_sales_inr']:,}/mo
              </div>
              <div>{reason_chips(row)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 2: Risk Heatmap
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("## Regional risk heatmap")
    st.caption("Combined pest + weather signal by district. Hover for details.")

    # Aggregate risk per district.
    risk = scored.groupby(["district", "state", "lat", "lon"]).agg(
        retailers=("retailer_id", "count"),
        avg_priority=("priority_score", "mean"),
        pest_events=("pest_pressure_14d", "max"),
        rain_7d=("recent_rain_7d_mm", "mean"),
    ).reset_index()
    risk["lat"] = risk["lat"].astype(float)
    risk["lon"] = risk["lon"].astype(float)

    fig = px.scatter_mapbox(
        risk,
        lat="lat", lon="lon",
        size="avg_priority",
        color="pest_events",
        hover_name="district",
        hover_data={"state": True, "retailers": True, "avg_priority": ":.1f", "rain_7d": ":.0f", "lat": False, "lon": False},
        color_continuous_scale=[(0, "#2c8a3f"), (0.5, "#f6a623"), (1, "#c43c3c")],
        size_max=40,
        zoom=5.5,
        height=560,
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 15.2, "lon": 77.5},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### District signals")
    st.dataframe(
        risk.sort_values("avg_priority", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Tab 3: Retailer 360
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("## Retailer 360")
    chosen = st.selectbox(
        "Pick a retailer",
        scored.sort_values("priority_score", ascending=False)["retailer_name"].tolist(),
    )
    r = scored[scored["retailer_name"] == chosen].iloc[0]

    c1, c2, c3 = st.columns([2, 2, 1])
    c1.markdown(f"### {r['retailer_name']}")
    c1.markdown(f"**District:** {r['district']}, {r['state']}")
    c1.markdown(f"**Crop focus:** {r['primary_crop'].title()} · **Tier {r['tier']}**")
    c1.markdown(f"**Avg monthly sales:** ₹{r['avg_monthly_sales_inr']:,}")
    c1.markdown(f"**Credit score:** {r['credit_score']}/100")
    c2.markdown("### Priority signal")
    c2.metric("Visit score", f"{r['priority_score']:.0f}/100")
    c2.markdown(reason_chips(r), unsafe_allow_html=True)
    c3.markdown("### Action")
    c3.button("📞 Call", use_container_width=True)
    c3.button("📍 Navigate", use_container_width=True)
    c3.button("✅ Log visit", use_container_width=True)

    st.markdown("---")

    # Sales trend
    st.markdown("### Sales trend (12 months)")
    rsales = sales[sales["retailer_id"] == r["retailer_id"]].copy()
    rsales_agg = rsales.groupby("month")["units_sold"].sum().reset_index()
    rsales_agg["month_dt"] = pd.to_datetime(rsales_agg["month"])
    rsales_agg = rsales_agg.sort_values("month_dt")
    fig2 = px.line(rsales_agg, x="month_dt", y="units_sold", markers=True)
    fig2.update_traces(line_color=PRIMARY)
    fig2.update_layout(height=280, margin=dict(t=20, b=20, l=10, r=10), xaxis_title="", yaxis_title="Units sold")
    st.plotly_chart(fig2, use_container_width=True)

    # Recommended products
    st.markdown("### Recommended products")
    rprods = rsales.groupby("product")["units_sold"].sum().reset_index().sort_values("units_sold", ascending=False)
    pc1, pc2, pc3 = st.columns(3)
    for col, (_, p) in zip([pc1, pc2, pc3], rprods.iterrows()):
        with col:
            st.markdown(f"**{p['product']}**")
            st.caption(f"Past 12 months: {int(p['units_sold'])} units")
            st.progress(min(1.0, p["units_sold"] / max(rprods["units_sold"].max(), 1)))


# ---------------------------------------------------------------------------
# Tab 4: Insights
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("## Why these recommendations? (Explainability)")
    st.caption("Global feature importance from the gradient-boosting model.")

    fi = pd.DataFrame([
        {"feature": f, "importance": float(model.feature_importances_[i])}
        for i, f in enumerate(FEATURES)
    ]).sort_values("importance")

    fig3 = px.bar(fi, x="importance", y="feature", orientation="h", color="importance",
                  color_continuous_scale=["#cde3d4", PRIMARY])
    fig3.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Pipeline performance")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Holdout ROC-AUC", f"{MODEL_AUC:.3f}")
    cc2.metric("Retailers tracked", f"{len(retailers)}")
    cc3.metric("Districts covered", f"{retailers['district'].nunique()}")

    st.markdown("### Pest activity (last 14 days)")
    p_recent = pests[pests["report_date"] >= pd.Timestamp(TODAY) - pd.Timedelta(days=14)]
    if len(p_recent):
        p_agg = p_recent.groupby(["district", "pest", "severity"]).size().reset_index(name="events")
        st.dataframe(p_agg, use_container_width=True, hide_index=True)
    else:
        st.info("No pest events reported in the last 14 days.")

st.markdown("---")
st.caption("AI Field Copilot · synthetic-data prototype · hackathon build")
