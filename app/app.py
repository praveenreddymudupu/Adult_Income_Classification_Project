import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Adult Income Predictor",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp { background-color: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.header-banner {
    background: linear-gradient(135deg, #1a1040 0%, #16213e 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.header-banner h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #a78bfa;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #8b949e;
    margin: 0;
    font-size: 0.95rem;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}
.result-high {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #34d399;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #1e1b4b, #2e1065);
    border: 1px solid #a78bfa;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-high h2 { color: #34d399; font-size: 1.6rem; margin: 0.3rem 0; }
.result-low h2  { color: #a78bfa; font-size: 1.6rem; margin: 0.3rem 0; }
.result-label {
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8b949e;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.2rem;
}
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #8b949e !important;
    font-size: 0.85rem !important;
}
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] div input {
    background-color: #21262d !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
.stButton button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: all 0.2s ease;
    letter-spacing: 0.5px;
}
.stButton button:hover {
    background: linear-gradient(135deg, #6d28d9, #5b21b6);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(167,139,250,0.3);
}
.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #34d399);
}
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "adult_income_best_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    bundle = load_model()
    pipeline     = bundle["pipeline"]
    model_name   = bundle["model_name"]
    feature_cols = bundle["feature_cols"]
    metrics      = bundle["metrics"]
except FileNotFoundError:
    st.error(
        "`adult_income_best_model.pkl` not found. "
        "Run the notebook's Step 12 save cell first, then place the file next to app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>💰 Adult Income Classification</h1>
    <p>Predict whether an individual earns <b>&gt;$50K/year</b> based on demographic
    and employment attributes from U.S. Census data.</p>
</div>
""", unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    age = c1.number_input("Age", min_value=17, max_value=90, value=35)
    sex = c2.selectbox("Sex", ["Male", "Female"])

    marital_status = st.selectbox(
        "Marital Status",
        ["Married-civ-spouse", "Never-married", "Divorced",
         "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    )

    # ── Employment ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Employment</div>', unsafe_allow_html=True)

    c6, c7 = st.columns(2)
    workclass = c6.selectbox(
        "Work Class",
        ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
         "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    )
    occupation = c7.selectbox(
        "Occupation",
        ["Tech-support", "Craft-repair", "Other-service", "Sales",
         "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
         "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
         "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
    )

    hours_per_week = st.slider(
        "Hours per Week", min_value=1, max_value=99, value=40
    )

    # ── Education & Finances ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Education & Finances</div>', unsafe_allow_html=True)

    education_num = st.slider(
        "Education Level (1=Preschool → 16=Doctorate)", min_value=1, max_value=16, value=10,
        help="1=Preschool, 5=5th-6th, 9=HS-grad, 10=Some-college, 13=Bachelors, 14=Masters, 16=Doctorate"
    )

    c10, c11 = st.columns(2)
    capital_gain = c10.number_input(
        "Capital Gain ($)", min_value=0, max_value=99999, value=0
    )
    capital_loss = c11.number_input(
        "Capital Loss ($)", min_value=0, max_value=4356, value=0
    )

    predict_btn = st.button("🔍 Predict Income", use_container_width=True)


# ── Results Panel ─────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        try:
            raw_input = {
    "age"            : int(age),
    "workclass"      : str(workclass),
    "fnlwgt"         : 189778,          # median value from training data
    "education.num"  : int(education_num),
    "marital.status" : str(marital_status),
    "occupation"     : str(occupation),
    "relationship"   : "Not-in-family", # most common default
    "race"           : "White",         # most common in dataset
    "sex"            : str(sex),
    "capital.gain"   : float(capital_gain),
    "capital.loss"   : float(capital_loss),
    "hours.per.week" : float(hours_per_week),
    "native.country" : "United-States", # most common default
}

            x_input = pd.DataFrame([raw_input])[feature_cols]

            prob       = float(pipeline.predict_proba(x_input)[0, 1])
            prediction = int(pipeline.predict(x_input)[0])

            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-label">Prediction</div>
                    <h2>Income &gt; $50K</h2>
                    <p style="color:#a7f3d0; margin:0;">
                        This individual is likely a <b>high earner</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-label">Prediction</div>
                    <h2>Income ≤ $50K</h2>
                    <p style="color:#ddd6fe; margin:0;">
                        This individual is likely earning <b>$50K or below</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="section-label">Probability of &gt;$50K</div>',
                unsafe_allow_html=True
            )
            st.progress(prob)
            st.markdown(f"""
            <div style="text-align:center; font-size:1.8rem; font-family:Syne;
                        font-weight:700; color:#a78bfa;">
                {prob:.1%}
            </div>
            """, unsafe_allow_html=True)

            # ── Model Metrics ─────────────────────────────────────────────────
            st.markdown(
                '<div class="section-label">Model Performance (Test Set)</div>',
                unsafe_allow_html=True
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""
            <div class="metric-card">
                <div class="value">{metrics['accuracy']:.2f}</div>
                <div class="label">Accuracy</div>
            </div>""", unsafe_allow_html=True)
            m2.markdown(f"""
            <div class="metric-card">
                <div class="value">{metrics['precision']:.2f}</div>
                <div class="label">Precision</div>
            </div>""", unsafe_allow_html=True)
            m3.markdown(f"""
            <div class="metric-card">
                <div class="value">{metrics['recall']:.2f}</div>
                <div class="label">Recall</div>
            </div>""", unsafe_allow_html=True)
            m4.markdown(f"""
            <div class="metric-card">
                <div class="value">{metrics['f1_score']:.2f}</div>
                <div class="label">F1-Score</div>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.markdown("""
        <div style="background:#161b22; border:1px dashed #30363d; border-radius:12px;
                    padding:3rem 2rem; text-align:center; color:#8b949e;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">💼</div>
            <div style="font-family:Syne; font-size:1rem; font-weight:600; color:#8b949e;">
                Fill in the individual's details and click Predict
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Active Model Card ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                padding:1rem 1.25rem;">
        <div style="font-size:0.7rem; letter-spacing:2px; color:#a78bfa;
                    text-transform:uppercase; font-weight:700; margin-bottom:0.5rem;">
            Active Model
        </div>
        <div style="font-family:Syne; font-weight:700; color:#e6edf3;">{model_name}</div>
        <div style="color:#8b949e; font-size:0.82rem; margin-top:0.25rem;">
            SMOTE · StandardScaler · OrdinalEncoder · StratifiedKFold(5) · RandomizedSearchCV
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Reference ─────────────────────────────────────────────────────
    with st.expander("📖 Education Level Reference"):
        edu_map = {
            1: "Preschool", 2: "1st-4th", 3: "5th-6th", 4: "7th-8th",
            5: "9th", 6: "10th", 7: "11th", 8: "12th", 9: "HS-grad",
            10: "Some-college", 11: "Assoc-voc", 12: "Assoc-acdm",
            13: "Bachelors", 14: "Masters", 15: "Prof-school", 16: "Doctorate"
        }
        for num, label in edu_map.items():
            st.markdown(
                f"<span style='color:#a78bfa;font-weight:700;'>{num}</span>"
                f" → {label}",
                unsafe_allow_html=True
            )
