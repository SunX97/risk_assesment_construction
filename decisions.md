# Architecture Decision Log: Construction Risk Assessment Pipeline

This document records the meaningful architectural, structural, and algorithmic decisions made during the development of the ML Risk Engine and Dashboard, including the context, chosen patterns, and accepted trade-offs.

---

## 1. Migration from Plotly to Altair for Data Visualization
**Context:** During initial prototyping in Google Colab, Streamlit's dynamic Javascript imports for Plotly charts and CSS stylesheets were constantly blocked by the `localtunnel` proxy due to Cross-Origin Resource Sharing (CORS) security protocols.
**Decision:** We abandoned Plotly and migrated the interactive bubble chart to **Altair**. 
**Why this pattern:** Altair is a declarative statistical visualization library based on Vega-Lite. It renders natively and safely across proxies without requiring heavy, asynchronous JavaScript chunk fetching.
**Accepted Trade-off:** We traded Plotly's out-of-the-box 3D capabilities and simpler syntax for Altair's network stability and rock-solid rendering in cloud/proxy environments.

## 2. Environment Pinning & Version Downgrades (NumPy/Pandas/Streamlit)
**Context:** Python 3.12 environments aggressively upgraded to NumPy 2.0+, which changed C-header byte sizes (from 88 to 96 bytes), causing severe binary incompatibility crashes with Pandas. Additionally, Python 3.12 removed the `distutils` module, breaking newer Streamlit browser launchers.
**Decision:** We pinned the environment to older, highly stable library versions (`pandas==2.1.4`, `numpy==1.26.4`, `streamlit==1.28.0`) and manually injected `setuptools`.
**Why this pattern:** Pinning dependencies ensures exact reproducibility across local machines, GitHub environments, and presentation laptops. 
**Accepted Trade-off:** We sacrificed access to the bleeding-edge features of Pandas 2.2+ and Streamlit 1.30+ to guarantee immediate system stability and prevent "dependency hell" right before a mid-project presentation.

## 3. Decoupling ML Training from UI Inference (Joblib)
**Context:** The Streamlit dashboard originally re-ran the entire script top-to-bottom on every user interaction, which would mean re-training the Random Forest model on 70,000 records every time a user clicked a button.
**Decision:** We structurally separated the project into `phase1_ml_engine.py` (Training) and `dashboard.py` (Inference). We used `joblib` to serialize (pickle) the trained model, TF-IDF vectorizer, and metrics to the `data/processed/` folder.
**Why this pattern:** This follows the standard ML Ops pattern of separating the "training pipeline" from the "deployment artifact." Streamlit now only reads the lightweight `.pkl` files.
**Accepted Trade-off:** The developer must remember to manually re-run `phase1_ml_engine.py` anytime the raw data changes. The UI does not automatically update its intelligence without this manual trigger.

## 4. Random Forest: Handling Severe Class Imbalance
**Context:** The OSHA dataset suffers from extreme survivorship bias/reporting bias. Employers only report severe injuries or fatalities. Therefore, the dataset is dominated by "High" severity events, with very few "Moderate" or "Low" events.
**Decision:** We implemented `class_weight='balanced'` in the Random Forest Classifier instead of using complex synthetic data generation like SMOTE.
**Why this pattern:** The algorithm automatically heavily penalizes itself for missing rare events (Fatalities), forcing the decision trees to pay close attention to niche NLP keywords like "died" or "trench".
**Accepted Trade-off:** We accepted a lower absolute confidence score (e.g., ~52% confidence for Fatal predictions rather than 90%+) because the model is hyper-sensitive to fatal keywords. In safety engineering, it is better to have a slightly less confident model that catches rare fatalities than an over-confident model that ignores them.

## 5. UI: Continuous Color Gradients vs. Categorical Modes
**Context:** Initially, the interactive risk map colored bubbles by their "Dominant Severity Index" (the statistical mode). Because the raw OSHA data is biased toward severe injuries, almost every construction activity defaulted to the color Red ("High"), making the chart visually useless.
**Decision:** We shifted the bubble chart to use a continuous color scale (`yelloworangered`) mapped directly to the mathematical `Fatality Probability`.
**Why this pattern:** It reveals the hidden, granular gradients of risk. It allows users to visually differentiate between an activity with a 2% fatality rate (yellow) and a 15% fatality rate (dark red).
**Accepted Trade-off:** We removed the simplicity of strict categorical labels (Low/Moderate/High) from the main chart, requiring the user to interpret a mathematical gradient instead of a simple traffic-light system.