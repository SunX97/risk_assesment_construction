import streamlit as st
import pandas as pd
import altair as alt
import joblib
import re

st.set_page_config(page_title="Construction Risk Assessment", page_icon="🚧", layout="wide")

st.title("🚧 Construction Safety Risk Assessment & Prevention")
st.markdown("Phase 1: ML Risk Engine Output - Activity Severity and Frequency Modeling")

ACTIVITY_TAXONOMY = {
    'Slab Casting & Concrete Pouring': r'slab|concrete|pour|formwork|curing|rebar',
    'Scaffolding & Work at Height': r'scaffold|elevat|staging|plank|ladder|height',
    'Excavation & Trenching': r'excavat|trench|digging|shoring|earthwork|cave-in',
    'Structural Steel & Ironwork': r'steel|girder|beam|erection|welding|crane',
    'Demolition Works': r'demolition|wrecking|jackhammer|collapse',
    'Electrical Installation': r'electrical|wire|circuit|voltage|electrocution|breaker',
    'Roofing & Perimeter Framing': r'roof|truss|decking|sheathing|skylight',
    'Heavy Machinery & Haulage': r'forklift|bulldozer|excavator|truck|loader|struck by'
}

def categorize_activity(text):
    text_lower = str(text).lower()
    for category, regex in ACTIVITY_TAXONOMY.items():
        if re.search(regex, text_lower):
            return category
    return 'General Construction Activity'

@st.cache_data
def load_data():
    matrix = pd.read_csv('data/processed/activity_risk_matrix.csv')
    report = pd.read_csv('data/processed/classification_report.csv', index_col=0)
    feat_imp = pd.read_csv('data/processed/feature_importance.csv')
    return matrix, report, feat_imp

@st.cache_resource
def load_models():
    model = joblib.load('data/processed/rf_risk_model.pkl')
    vectorizer = joblib.load('data/processed/tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    matrix_df, report_df, feat_imp_df = load_data()
    rf_model, tfidf_vectorizer = load_models()
except FileNotFoundError:
    st.error("Required files missing. Run phase1_ml_engine.py first to generate data and models.")
    st.stop()

# ==========================================
# DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Interactive Risk Map", 
    "📋 Master Risk Matrix", 
    "📈 Model Performance",
    "⚡ Live ML Predictor", 
    "⚙️ Phase 2: RAG Audit"
])

# TAB 1: ALTAIR VISUALIZATION
with tab1:
    st.subheader("Frequency vs. Fatality Probability")
    st.write("Hover over the bubbles to view specific activity risks. Pan and zoom using your mouse.")
    
    # Create Altair Bubble Chart (UPDATED FOR CONTINUOUS COLOR SCALE)
    scatter_plot = alt.Chart(matrix_df).mark_circle(opacity=0.85).encode(
        x=alt.X('Total Historical Incidents:Q', title='Incident Frequency (Total Records)'),
        y=alt.Y('Fatality Probability:Q', title='Likelihood of Fatality'),
        size=alt.Size('Total Historical Incidents:Q', scale=alt.Scale(range=[200, 2000]), legend=None),
        
        # NEW: Continuous color scale from Yellow to Dark Red based on exact Fatality Probability
        color=alt.Color('Fatality Probability:Q', scale=alt.Scale(scheme='yelloworangered'), title='Fatality Risk Gradient'),
        
        tooltip=[
            alt.Tooltip('Construction Activity:N', title='Activity'),
            alt.Tooltip('Fatality Probability:Q', title='Fatality Prob', format='.3f'),
            alt.Tooltip('High Severity Probability:Q', title='High Sev Prob', format='.3f'),
            alt.Tooltip('Total Historical Incidents:Q', title='Total Incidents')
        ]
    ).interactive().properties(height=500)
    
    # Add a threshold line
    threshold_line = alt.Chart(pd.DataFrame({'y': [0.10]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')
    
    # Combine and render
    final_chart = scatter_plot + threshold_line
    st.altair_chart(final_chart, use_container_width=True)

# TAB 2: DATA TABLE
with tab2:
    st.subheader("ML Risk Engine Output Data")
    st.dataframe(matrix_df.style.background_gradient(subset=['Fatality Probability'], cmap='Reds'), use_container_width=True, hide_index=True)

# TAB 3: MODEL PERFORMANCE
with tab3:
    st.subheader("Classification Metrics")
    st.write("Precision, Recall, and F1-Score across risk severities. Evaluated on 20% unseen test data.")
    st.dataframe(report_df.style.format(precision=3).background_gradient(cmap='Blues'), use_container_width=True)
    
    st.subheader("Top 20 Predictive Features (TF-IDF)")
    st.write("The specific narrative keywords the Random Forest relied on most heavily to predict severity.")
    
    bar_chart = alt.Chart(feat_imp_df).mark_bar().encode(
        x=alt.X('Importance:Q', title='Gini Importance (Weight)'),
        y=alt.Y('Feature:N', sort='-x', title='NLP Token'),
        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['Feature', 'Importance']
    ).properties(height=500)
    
    st.altair_chart(bar_chart, use_container_width=True)

# TAB 4: LIVE ML PREDICTOR
with tab4:
    st.subheader("Test the ML Engine")
    user_input = st.text_area("Accident Narrative:", "Worker lost balance while adjusting a girder on the 4th floor staging and fell...")
    if st.button("Predict Severity"):
        if user_input.strip() != "":
            pred_category = categorize_activity(user_input)
            input_vector = tfidf_vectorizer.transform([user_input])
            pred_severity = rf_model.predict(input_vector)[0]
            confidence = rf_model.predict_proba(input_vector).max() * 100
            
            st.markdown("### Prediction Results")
            st.info(f"**Identified Activity:** {pred_category}")
            if pred_severity == 'Fatal':
                st.error(f"**Predicted Severity:** {pred_severity} (Confidence: {confidence:.1f}%)")
            elif pred_severity == 'High':
                st.warning(f"**Predicted Severity:** {pred_severity} (Confidence: {confidence:.1f}%)")
            else:
                st.success(f"**Predicted Severity:** {pred_severity} (Confidence: {confidence:.1f}%)")
        else:
            st.warning("Please enter a narrative.")

# TAB 5: PHASE 2 PLACEHOLDER
with tab5:
    st.subheader("Automated Contract Audit")
    st.info("This section will integrate the RAG Pipeline (Phase 2).")