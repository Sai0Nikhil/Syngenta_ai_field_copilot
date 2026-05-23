# AI Field Copilot

Smart AI decision-support for agricultural field reps. Hackathon prototype focused on **visit prioritization** for Karnataka & Andhra Pradesh (chilli, maize, cotton).

## What it does

Each morning the Copilot scores every retailer on the rep's beat and surfaces a ranked daily plan. Each recommendation comes with **reason chips** (overdue visit, high pest activity, recent rainfall, Tier-A account, etc.) so the rep understands *why*.

## Quick start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Generate synthetic data (one time)
python scripts/generate_data.py

# 3. Train the model (one time)
python scripts/train_model.py

# 4. Launch the app
streamlit run app/copilot.py
```

The app opens at http://localhost:8501.

## Folder layout

```
ai_field_copilot/
├── app/
│   └── copilot.py            # Streamlit prototype (entry point)
├── scripts/
│   ├── generate_data.py      # Synthetic dataset generator
│   └── train_model.py        # Trains the gradient-boosting model
├── data/                     # Generated CSVs
├── model/                    # Trained model + feature importance
├── docs/                     # Architecture & design documents
├── deck/                     # Pitch deck
├── demo/                     # Demo video script + UI mockup
└── README.md
```

## The four screens

1. **Today's Plan** — top-N prioritized retailers with reason chips and one-tap actions.
2. **Risk Heatmap** — district-level risk overlay (pest + weather) on a map of Karnataka/AP.
3. **Retailer 360** — sales trend, recommended products, account snapshot, action buttons.
4. **Insights** — model explainability (feature importance) and pipeline health metrics.

## Model

- **Task**: predict whether a retailer will place an order in the next 7 days.
- **Algorithm**: gradient-boosting classifier (sklearn).
- **Features (10)**: avg monthly sales, days since visit, recent 7-day rainfall, recent humidity, pest pressure (14d), credit score, tier, crop one-hots.
- **Holdout ROC-AUC**: 0.815 on 5,760 synthetic rows.

## Data sources (production)

The synthetic generator mirrors the schema we'd consume in production:
- **Weather** → IMD / OpenWeather APIs
- **NDVI** → Sentinel-2 via Sentinel Hub or Bhuvan
- **Pest surveillance** → ICAR-NCIPM / state agriculture department feeds
- **Retailer & sales** → Syngenta CRM and distributor sell-out data

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit + Plotly |
| ML | scikit-learn GradientBoostingClassifier |
| Data prep | pandas |
| Persistence | joblib |
| Maps | Plotly Mapbox (carto-positron tiles) |


## Team

Built for the Syngenta Hackathon by a team of 3.
