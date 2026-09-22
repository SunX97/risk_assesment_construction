# Execution Flow & Session Modification Log

This document maps the exact entry points, call stacks, and execution lifecycle of the Construction Risk Assessment application. It also tracks the specific code modifications made during the current development session.

---

## 1. Offline ML Training Flow (`phase1_ml_engine.py`)
**Entry Point:** The `if __name__ == "__main__":` block at the absolute bottom of the script.
**Trigger:** Terminal execution via `python phase1_ml_engine.py`.

**Execution Order & Call Stack:**
1. **`load_and_standardize_osha()` is called.**
   ↳ *Calls:* `resolve_file_paths()` sequentially for SIR, Fatality, and Kaggle paths to expand wildcards (`*`) into actual OS file paths.
   ↳ *Returns:* `master_df` (a unified Pandas DataFrame of ~70k records).
2. **`train_ml_risk_engine(master_df)` is called.**
   ↳ *Calls:* `df['Description'].apply(categorize_activity)` 
      ↳ *Calls:* `re.search()` against the `ACTIVITY_TAXONOMY` regex dictionary.
   ↳ *Calls:* `tfidf.fit_transform()` to vectorize text.
   ↳ *Calls:* `clf.fit()` to train the Random Forest.
   ↳ *Calls:* `classification_report()` and `clf.feature_importances_` to generate metrics.
   ↳ *Returns:* `model`, `vectorizer`, `processed_df`, `report_df`, `feat_imp_df`.
3. **`build_activity_risk_matrix(processed_df, model, vectorizer)` is called.**
   ↳ *Calculates:* Fatality probabilities based on historical frequency grouped by Activity.
   ↳ *Returns:* `matrix` (Pandas DataFrame).
4. **Disk I/O Execution (No custom functions called):**
   ↳ Calls Pandas `.to_csv()` three times to save `activity_risk_matrix.csv`, `classification_report.csv`, and `feature_importance.csv`.
   ↳ Calls `joblib.dump()` twice to save `rf_risk_model.pkl` and `tfidf_vectorizer.pkl`.

---

## 2. Online Inference & UI Flow (`dashboard.py`)
**Entry Point:** Top of the script (Streamlit executes top-to-bottom).
**Trigger:** Terminal execution via `streamlit run dashboard.py`.

**Execution Order & Call Stack:**
1. **Script Initialization:** Imports Pandas, Altair, Joblib, and defines `ACTIVITY_TAXONOMY` and `categorize_activity()`.
2. **Resource Caching (Executed once per server start):**
   ↳ **`load_data()` is called.** Reads the three pre-calculated CSVs from `data/processed/`.
   ↳ **`load_models()` is called.** Deserializes the two Joblib `.pkl` models into memory.
3. **Static UI Rendering:**
   ↳ Calls `st.tabs()` to build the layout.
   ↳ *Tab 1 & 3:* Calls `alt.Chart().mark_circle/bar().encode()` to render data visualizations directly from memory.
   ↳ *Tab 2:* Calls `st.dataframe()` with Pandas Styler gradients.
4. **Dynamic Event-Driven Execution (Tab 4 - Live Predictor):**
   ↳ Pauses at `st.button("Predict Severity")`. When clicked by the user:
      1. *Calls:* `categorize_activity(user_input)` to identify the construction task.
      2. *Calls:* `tfidf_vectorizer.transform([user_input])` to convert string to tensor.
      3. *Calls:* `rf_model.predict(input_vector)` to get the risk class (Low/Mod/High/Fatal).
      4. *Calls:* `rf_model.predict_proba(input_vector)` to get the mathematical confidence percentage.
      5. Updates UI via `st.error/warning/info/success`.

---

## 3. Current Session Code Modifications (AI Changelog)
*This section logs exactly which parts of the codebase were altered during the transition from Google Colab to VS Code.*

1. **Path Routing & Project Structure:**
   - *Changed:* Replaced flat file reads (e.g., `'January2015...csv'`) with explicit relative paths routing to `data/raw/` (inputs) and `data/processed/` (outputs).
2. **Library Migration (Plotly -> Altair):**
   - *Changed:* Replaced Plotly (`px.scatter`) in Tab 1 with `alt.Chart().mark_circle()`.
   - *Reason:* Resolved Localtunnel/VS Code proxy JS injection failures. 
   - *Changed:* Modified the color encoding from a categorical map to a continuous gradient (`yelloworangered`) based on `Fatality Probability`.
3. **Execution Order Fix (`phase1_ml_engine.py`):**
   - *Changed:* Moved the `if __name__ == "__main__":` execution block from the top of the file to the absolute bottom.
   - *Reason:* Fixed a `NameError` caused by Python attempting to call `load_and_standardize_osha` before the function was loaded into memory.
4. **Decoupling Training from Inference (`joblib`):**
   - *Changed:* Added `joblib.dump()` to `phase1_ml_engine.py` and `joblib.load()` to `dashboard.py`. Added a "Live Predictor" tab.
   - *Reason:* Prevents Streamlit from retraining a 70,000-row Random Forest every time a user clicks a button.
5. **Metric Extraction Pipeline:**
   - *Changed:* Altered `train_ml_risk_engine()` to return `report_df` and `feat_imp_df`. 
   - *Changed:* Added Tab 3 to `dashboard.py` to ingest these CSVs and render model performance charts.