# 🚧 Construction Safety Risk Assessment & Automated Audit System

## 📖 Project Overview
This project is an end-to-end Machine Learning and Natural Language Processing (NLP) pipeline designed to enhance safety protocols in the construction industry. It bridges the gap between historical accident data and proactive contract auditing.

The system is developed in two distinct phases:
*   **Phase 1: ML Risk Engine (Active)** - Ingests ~70,000 historical OSHA incident reports to calculate empirical fatality probabilities across construction activities. It features a Random Forest NLP model that predicts the severity of hypothetical accidents in real-time.
*   **Phase 2: RAG Compliance Audit (Upcoming)** - An automated auditing tool that ingests Construction Tender/BOQ documents and uses a local Vector Database (ChromaDB) to cross-reference identified tasks against official ISO 45001 and OSHA 1926 safety regulations.

## 🏗️ System Architecture
The application uses a decoupled architecture, separating heavy backend data processing from the frontend user interface. 

*   **Offline Processing (`phase1_ml_engine.py`):** Handles data cleaning, TF-IDF vectorization, and Random Forest model training. Serializes the trained models (`.pkl`) and pre-computed metrics to disk.
*   **Online Presentation (`dashboard.py`):** A lightweight Streamlit frontend that reads the serialized artifacts to serve an interactive Altair risk map and a Live ML Predictor without retraining the model on the fly.
*   **Data Layers:** Strict separation between `data/raw/` (inputs), `data/knowledge_base/` (PDFs), and `data/processed/` (ephemeral outputs).

> 💡 **Note:** For deep technical dives, execution flows, and architectural trade-offs, please see the markdown files located in the `/docs` folder (`architecture.md`, `flow.md`, `decisions.md`).

## 🛠️ Tech Stack
*   **Language:** Python 3.12
*   **Machine Learning:** Scikit-Learn, Joblib
*   **Data Processing:** Pandas, NumPy
*   **NLP & RAG (Phase 2):** LangChain, ChromaDB, PyMuPDF, Sentence-Transformers
*   **Frontend UI:** Streamlit, Altair

---

## 💻 Local Setup & Installation

Follow these steps to configure your local VS Code environment. 

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/risk_assesment_construction.git](https://github.com/your-username/risk_assesment_construction.git)
cd risk_assesment_construction
python -m venv venv
Windows (PowerShell): .\venv\Scripts\activate
Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

🚀 Execution Guide
Step 1: Generate the ML Models (Offline Training)
Before launching the UI, you must process the raw CSV data into a mathematical matrix and train the predictive ML models. Ensure your raw OSHA data is placed inside the data/raw/ directory.

Run the ML engine from your terminal:
python phase1_ml_engine.py
Step 2: Launch the Dashboard (Online UI)
Once Step 1 is complete and the .pkl files exist, you can launch the Streamlit server:
streamlit run dashboard.py
