import joblib
import pandas as pd
import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ============================================================
# HELPER: RESOLVE WILDCARDS & LISTS OF PATHS
# ============================================================
def resolve_file_paths(path_input):
    """Expands single strings, wildcards (*), or lists of patterns into distinct file paths."""
    if not path_input:
        return []
    if isinstance(path_input, str):
        path_input = [path_input]

    resolved = []
    for pattern in path_input:
        matched = glob.glob(pattern)
        if matched:
            resolved.extend(matched)
        elif glob.os.path.exists(pattern):
            resolved.append(pattern)
    return sorted(list(set(resolved)))

# ============================================================
# STEP 1: LOAD & HARMONIZE RAW OSHA DATASETS
# ============================================================
def load_and_standardize_osha(sir_path=None, fatal_paths=None, kaggle_paths=None):
    dfs = []

    # 1. Severe Injury Reports (SIR)
    sir_files = resolve_file_paths(sir_path)
    for f in sir_files:
        try:
            df_sir = pd.read_csv(f, encoding='latin1', engine='python', on_bad_lines='skip')
            naics_cols = [c for c in df_sir.columns if 'naics' in c.lower()]
            if not naics_cols:
                continue

            # Filter NAICS for Construction (Sector 23)
            df_sir = df_sir[df_sir[naics_cols[0]].astype(str).str.startswith('23')].copy()

            # Find descriptive narrative column
            candidate_cols = [c for c in df_sir.columns if any(k in c.lower() for k in ['description', 'narrative', 'event']) and 'date' not in c.lower()]
            if candidate_cols:
                text_col = candidate_cols[0]
                df_sir['Description'] = df_sir[text_col].fillna('').astype(str)
                df_sir['Severity'] = 'High'
                dfs.append(df_sir[['Description', 'Severity']])
                print(f"Loaded {len(df_sir)} records from SIR: {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # 2. Fatality Inspection Records
    fatal_files = resolve_file_paths(fatal_paths)
    for f in fatal_files:
        try:
            df_fat = pd.read_csv(f, encoding='latin1', engine='python', on_bad_lines='skip')
            text_cols = [c for c in df_fat.columns if any(k in c.lower() for k in ['summary', 'description', 'narrative', 'abstract', 'event']) and 'date' not in c.lower()]
            if text_cols:
                df_fat['Description'] = df_fat[text_cols[0]].fillna('').astype(str)
                df_fat['Severity'] = 'Fatal'
                dfs.append(df_fat[['Description', 'Severity']])
                print(f"Loaded {len(df_fat)} fatality records from: {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # 3. Kaggle Curated Datasets
    kaggle_files = resolve_file_paths(kaggle_paths)
    for f in kaggle_files:
        try:
            df_kag = pd.read_csv(f, encoding='latin1', engine='python', on_bad_lines='skip')
            text_cols = [c for c in df_kag.columns if any(k in c.lower() for k in ['abstract', 'description', 'narrative', 'summary']) and 'date' not in c.lower()]
            if text_cols:
                df_kag['Description'] = df_kag[text_cols[0]].fillna('').astype(str)

                # Check for degree of injury column
                deg_cols = [c for c in df_kag.columns if 'degree' in c.lower() or 'injury' in c.lower()]
                if deg_cols and 'degree_of_injury' in df_kag.columns:
                    mapping = {'Fatality': 'Fatal', 'Hospitalized': 'High', 'Non-Hospitalized': 'Moderate'}
                    df_kag['Severity'] = df_kag['degree_of_injury'].map(mapping).fillna('Low')
                else:
                    df_kag['Severity'] = 'Moderate'

                dfs.append(df_kag[['Description', 'Severity']])
                print(f"Loaded {len(df_kag)} summary records from: {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        raise ValueError("No valid datasets were loaded. Please check your folder paths.")

    master_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['Description'])
    master_df = master_df[master_df['Description'].str.strip().str.len() > 10].copy()
    print(f"\nTotal combined unique records: {len(master_df)}")
    return master_df

# ============================================================
# STEP 2: ACTIVITY CATEGORIZATION FROM NARRATIVE
# ============================================================
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

# ============================================================
# STEP 3: FEATURE ENGINEERING & MODEL TRAINING
# ============================================================
def train_ml_risk_engine(df):
    df['Activity'] = df['Description'].apply(categorize_activity)

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X = tfidf.fit_transform(df['Description'])
    y = df['Severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    # NEW: Extract Classification Report as a DataFrame
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # NEW: Extract Feature Importances
    feature_names = tfidf.get_feature_names_out()
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(20)

    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Return the metrics dataframes as well
    return clf, tfidf, df, report_df, feat_imp_df


# ============================================================
# STEP 4: GENERATE ACTIVITY RISK MATRIX
# ============================================================
def build_activity_risk_matrix(df, model, tfidf):
    matrix_rows = []

    for activity, group in df.groupby('Activity'):
        if len(group) < 5:
            continue

        severity_counts = group['Severity'].value_counts(normalize=True)
        modal_severity = severity_counts.idxmax()
        fatal_prob = severity_counts.get('Fatal', 0.0)

        matrix_rows.append({
            'Construction Activity': activity,
            'Total Historical Incidents': len(group),
            'Dominant Severity Index': modal_severity,
            'Fatality Probability': round(fatal_prob, 3),
            'High Severity Probability': round(severity_counts.get('High', 0.0), 3),
            'Moderate Severity Probability': round(severity_counts.get('Moderate', 0.0), 3),
            'Low Severity Probability': round(severity_counts.get('Low', 0.0), 3)
        })

    risk_matrix = pd.DataFrame(matrix_rows).sort_values(by='Fatality Probability', ascending=False)
    
    # Save directly to the processed data folder
    risk_matrix.to_csv('data/processed/activity_risk_matrix.csv', index=False)
    print("\nSaved 'activity_risk_matrix.csv' to data/processed/ successfully.")
    return risk_matrix

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    # Pointing to the local data/raw/ folder
    master_df = load_and_standardize_osha(
        sir_path="data/raw/January2015toNovember2025.csv",
        fatal_paths=[
            "data/raw/OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv",
            "data/raw/FatalitiesFY*.csv"
        ],
        kaggle_paths="data/raw/fy*_federal-state_summaries.csv"
    )

    # Capture the new metric dataframes
    model, vectorizer, processed_df, report_df, feat_imp_df = train_ml_risk_engine(master_df)
    matrix = build_activity_risk_matrix(processed_df, model, vectorizer)
    
    # Save the output matrix and models
    matrix.to_csv('data/processed/activity_risk_matrix.csv', index=False)
    import joblib
    joblib.dump(model, 'data/processed/rf_risk_model.pkl')
    joblib.dump(vectorizer, 'data/processed/tfidf_vectorizer.pkl')
    
    # NEW: Save the metrics to CSVs for the dashboard
    report_df.to_csv('data/processed/classification_report.csv', index=True)
    feat_imp_df.to_csv('data/processed/feature_importance.csv', index=False)
    
    print("\nSaved 'classification_report.csv' and 'feature_importance.csv' to data/processed/")