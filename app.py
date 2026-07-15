import pandas as pd
import streamlit as st
import joblib
import os
import requests

# ---------- Preprocessing ----------

# Mapping to turn word answers into numbers
freq_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

def preprocess_input(raw_input, feature_cols):
    # Start with one row of all zeros, with the same columns the model was trained on
    row = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Ordinal columns - map them to numbers
    row["CAEC"] = freq_map[raw_input["CAEC"]]
    row["CALC"] = freq_map[raw_input["CALC"]]

    # Numeric columns - just copy the value the user typed in
    for col in numeric_cols:
        row[col] = raw_input[col]

    # One-hot columns - turn on the matching column if it exists.
    # If it doesn't exist, that means this answer was the "dropped" baseline
    # category from training, so leaving it as 0 is correct.
    gender_col = "Gender_" + raw_input["Gender"]
    if gender_col in row.columns:
        row[gender_col] = 1

    family_col = "family_history_with_overweight_" + raw_input["family_history_with_overweight"]
    if family_col in row.columns:
        row[family_col] = 1

    favc_col = "FAVC_" + raw_input["FAVC"]
    if favc_col in row.columns:
        row[favc_col] = 1

    smoke_col = "SMOKE_" + raw_input["SMOKE"]
    if smoke_col in row.columns:
        row[smoke_col] = 1

    scc_col = "SCC_" + raw_input["SCC"]
    if scc_col in row.columns:
        row[scc_col] = 1

    mtrans_col = "MTRANS_" + raw_input["MTRANS"]
    if mtrans_col in row.columns:
        row[mtrans_col] = 1

    return row


# ---------- UI and Model Loading ----------

# ---------- Risk spectrum reference ----------

risk_order = [
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]
risk_colors = {
    "Insufficient_Weight": "#4A90A4",
    "Normal_Weight": "#2E9E6B",
    "Overweight_Level_I": "#8FBF3F",
    "Overweight_Level_II": "#F2C14E",
    "Obesity_Type_I": "#EF9B4E",
    "Obesity_Type_II": "#E8703A",
    "Obesity_Type_III": "#D64550",
}

# ---------- Mappings for the natural-language dropdowns ----------

fcvc_options = {"Never": 1.0, "Sometimes": 2.0, "Always": 3.0}
ncp_options = {"1–2 meals a day": 1.0, "3 meals a day": 2.0, "More than 3 meals a day": 3.0}
ch2o_options = {"Less than 1 liter": 1.0, "1–2 liters": 2.0, "More than 2 liters": 3.0}
faf_options = {"I don't exercise": 0.0, "1–2 days a week": 1.0, "2–4 days a week": 2.0, "4–5 days a week": 3.0}
tue_options = {"0–2 hours": 0.0, "3–5 hours": 1.0, "More than 5 hours": 2.0}

GITHUB_URL = "https://github.com/Jahanzeb-abbasi/obesity-risk-classifier"
KAGGLE_DATASET = "https://www.kaggle.com/competitions/playground-series-s4e2"

FRIENDLY_NAMES = {
    "Age": "Age", "Height": "Height", "Weight": "Weight",
    "FCVC": "Vegetable consumption", "NCP": "Number of meals",
    "CH2O": "Water intake", "FAF": "Physical activity",
    "TUE": "Screen time", "CAEC": "Snacking frequency", "CALC": "Alcohol consumption",
}

def friendly_name(col):
    return FRIENDLY_NAMES.get(col, col.replace("_", " "))

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obesity Class I"
    elif bmi < 40:
        return "Obesity Class II"
    else:
        return "Obesity Class III"

def github_button_html():
    return f"""
    <a href="{GITHUB_URL}" target="_blank" style="
        display:inline-flex; align-items:center; gap:8px;
        background-color:#181717; color:#ffffff; text-decoration:none;
        padding:0.5rem 1rem; border-radius:8px; font-weight:600; font-size:0.9rem;">
        <svg height="18" width="18" viewBox="0 0 16 16" fill="#ffffff">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
            0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
            -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07
            -1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82
            .64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
            .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2
            0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"/>
        </svg>
        View on GitHub
    </a>
    """

# ---------- Page setup ----------

st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

.stButton>button, .stFormSubmitButton>button {
    background-color: #3FA796;
    color: #0E1512;
    border-radius: 10px;
    border: none;
    padding: 0.7rem 1.4rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 10px rgba(63, 167, 150, 0.25);
    transition: all 0.2s ease;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
    background-color: #57C4B0;
    color: #0E1512;
    box-shadow: 0 4px 14px rgba(63, 167, 150, 0.4);
    transform: translateY(-1px);
}

.mono-num { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

MODEL_URL = "https://huggingface.co/Zaibyyx/obesity-risk-classifier-model/resolve/main/random_forest_model.joblib"
MODEL_PATH = "artifacts/random_forest_model.joblib"


@st.cache_resource
def load_artifacts():

    os.makedirs("artifacts", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... (first run only)"):
            response = requests.get(MODEL_URL)
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    rf = joblib.load(MODEL_PATH)
    feature_cols = joblib.load("artifacts/feature_cols.joblib")

    return rf, feature_cols

rf, feature_cols = load_artifacts()

# ---------- About dialog (opens as a popup when the sidebar button is clicked) ----------

@st.dialog("About This Project", width="large")
def show_about():
    st.markdown("**Obesity Risk Predictor**")
    st.write(
        "A multi-class machine learning app that estimates obesity risk category "
        "from eating habits, activity levels, and body measurements."
    )
    st.markdown(
        "**Dataset:** Kaggle Playground Series S4E2 - 20,758 rows, 7 risk classes, "
        "16 lifestyle & body-measurement features."
    )
    st.markdown(f"Link → {KAGGLE_DATASET}")
    st.markdown("**Models compared:**")
    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": ["86%", "90%"],
        "Macro F1": ["0.85", "0.89"],
    })
    st.table(comparison_df.set_index("Model"))
    st.markdown("**Deployed model:** Random Forest (200 trees)")
    st.markdown("**Tech stack:** Python · Pandas · Scikit-learn · Streamlit")
    st.markdown(github_button_html(), unsafe_allow_html=True)

    st.divider()
    st.write("**What matters most to this model overall:**")
    st.caption(
        "This reflects the model's general behavior across all 20,758 training rows — "
        "not a personalized breakdown of this specific prediction."
    )
    importances = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    for col, imp in importances[:6]:
        st.markdown(
            f"""<div style="display:flex; align-items:center; margin-bottom:4px;">
            <div style="width:180px; font-size:0.85rem;">{friendly_name(col)}</div>
            <div style="flex:1; background:#2A3436; border-radius:4px; height:14px; margin:0 8px;">
                <div style="width:{imp*100:.1f}%; background:#3FA796; height:14px; border-radius:4px;"></div>
            </div>
            <div class="mono-num" style="width:50px; text-align:right;">{imp*100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
        
    st.divider()

    if st.button("Close"):
        st.rerun()

# ---------- Sidebar ----------

with st.sidebar:
    st.header("About This Project")
    st.write("A quick estimate of obesity risk from lifestyle and body measurements.")
    if st.button("📖 View Full Details", use_container_width=True):
        show_about()
    st.divider()
    st.caption("⚠️ Educational project only - Not medical advice.")

# ---------- Main form ----------

st.title("Obesity Risk Predictor 🩺")
st.write("Fill in your details below to estimate your obesity risk category.")

with st.form("prediction_form"):

    with st.container(border=True):
        st.subheader("🧍 Body Basics")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=14, max_value=90, value=25, step=1)
            gender = st.radio("Gender", ["Female", "Male"])
        with col2:
            height_cm = st.number_input("Height (cm)", min_value=140, max_value=220, value=170, step=1)
            family_history = st.radio(
                "Family history of overweight or obesity?", ["yes", "no"],
                format_func=lambda x: x.capitalize(),
                help="Any parents or siblings who have experienced overweight or obesity.",
            )
        with col3:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70, step=1)

    with st.container(border=True):
        st.subheader("🍽️ Eating Habits")
        col1, col2, col3 = st.columns(3)
        with col1:
            favc = st.radio(
                "Do you frequently eat high-calorie foods?", ["yes", "no"],
                format_func=lambda x: x.capitalize(),
                help="Fast food, fried food, sugary snacks, and similar foods on a regular basis.",
            )
            fcvc_label = st.selectbox("How often do you eat vegetables with meals?", list(fcvc_options.keys()))
        with col2:
            ncp_label = st.selectbox("How many main meals do you eat daily?", list(ncp_options.keys()))
            caec = st.selectbox("Do you snack between meals?", ["no", "Sometimes", "Frequently", "Always"])
        with col3:
            ch2o_label = st.selectbox("How much water do you drink daily?", list(ch2o_options.keys()))
            calc = st.selectbox("How often do you drink alcohol?", ["no", "Sometimes", "Frequently", "Always"])

    with st.container(border=True):
        st.subheader("🏃 Lifestyle & Activity")
        col1, col2, col3 = st.columns(3)
        with col1:
            smoke = st.radio(
                "Do you smoke?", ["yes", "no"],
                format_func=lambda x: x.capitalize(),
            )
        with col2:
            scc = st.radio(
                "Do you monitor your daily calorie intake?", ["yes", "no"],
                format_func=lambda x: x.capitalize(),
                help="Actively tracking calories eaten, for example with an app or food diary.",
            )
            faf_label = st.selectbox("How often do you engage in physical activity?", list(faf_options.keys()))
        with col3:
            mtrans = st.selectbox(
                "What is your primary mode of transportation?",
                ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
            )
            tue_label = st.selectbox("How much daily screen time do you have?", list(tue_options.keys()))

    submitted = st.form_submit_button("🔍 Get My Risk Assessment", use_container_width=True, type="primary")

# ---------- Prediction + results ----------

if submitted:
    raw_input = {
        "Age": age,
        "Height": height_cm / 100,
        "Weight": weight,
        "FCVC": fcvc_options[fcvc_label],
        "NCP": ncp_options[ncp_label],
        "CH2O": ch2o_options[ch2o_label],
        "FAF": faf_options[faf_label],
        "TUE": tue_options[tue_label],
        "Gender": gender,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "SMOKE": smoke,
        "SCC": scc,
        "MTRANS": mtrans,
        "CAEC": caec,
        "CALC": calc,
    }

    with st.spinner("Analyzing your inputs..."):
        row = preprocess_input(raw_input, feature_cols)
        prediction = rf.predict(row)[0]
        proba = rf.predict_proba(row)[0]
        proba_dict = dict(zip(rf.classes_, proba))

    st.subheader("Your Risk Assessment")

    color = risk_colors[prediction]
    col_result, col_bmi = st.columns([2, 1])

    with col_result:
        st.markdown(
            f"""<div style="background-color:{color}22; border-left: 6px solid {color};
            padding: 1rem; border-radius: 8px; height: 100%;">
            <span style="font-size:0.85rem; color:#9BA6A2;">Predicted category</span><br/>
            <span style="font-family:'Space Grotesk'; font-size: 1.3rem; font-weight:700; color:{color};">
            {prediction.replace('_', ' ')}</span></div>""",
            unsafe_allow_html=True,
        )

    with col_bmi:
        bmi = weight / (height_cm / 100) ** 2
        st.metric("Your BMI", f"{bmi:.1f}", bmi_category(bmi), delta_color="off")

    st.caption(
        "BMI is a simple weight-to-height ratio. The model's prediction can differ from this "
        "because it also accounts for lifestyle factors BMI alone doesn't capture."
    )

    marker_pct = risk_order.index(prediction) / (len(risk_order) - 1) * 100
    gradient = ", ".join(risk_colors[c] for c in risk_order)
    st.markdown(
        f"""
        <div style="position:relative; height:24px; border-radius:12px;
        background: linear-gradient(90deg, {gradient}); margin: 1rem 0 1.5rem 0;">
            <div style="position:absolute; left:{marker_pct}%; top:-6px; transform:translateX(-50%);
            width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
            border-top:10px solid #E8ECEA;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#9BA6A2; margin-top:-20px;">
            <span>Insufficient Weight</span><span>Obesity Type III</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
    top_class, top_p = sorted_probs[0]
    second_class, second_p = sorted_probs[1]
    if top_p - second_p < 0.15:
        st.info(
            f"**Close call:** the model was fairly split between **{top_class.replace('_', ' ')}** "
            f"({top_p*100:.0f}%) and **{second_class.replace('_', ' ')}** ({second_p*100:.0f}%)."
        )

    st.divider()
    st.write("**Probability by category**, from lowest to highest risk:")
    for cls in risk_order:
        p = proba_dict[cls]
        bar_color = risk_colors[cls]
        st.markdown(
            f"""<div style="display:flex; align-items:center; margin-bottom:4px;">
            <div style="width:180px; font-size:0.85rem;">{cls.replace('_', ' ')}</div>
            <div style="flex:1; background:#2A3436; border-radius:4px; height:14px; margin:0 8px;">
                <div style="width:{p*100:.1f}%; background:{bar_color}; height:14px; border-radius:4px;"></div>
            </div>
            <div class="mono-num" style="width:50px; text-align:right;">{p*100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )



# ---------- Footer ----------

st.divider()
st.caption("⚠️ This is a machine learning estimate for educational purposes - not medical advice.")
st.write("Built as an Applied AI Engineering capstone project")
st.markdown(github_button_html(), unsafe_allow_html=True)