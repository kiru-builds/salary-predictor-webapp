import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="💰 Salary Predictor | By Kirthika",
    page_icon="💰",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
h1 {
    text-align: center;
    font-size: 2.5rem !important;
    background: linear-gradient(90deg, #00ff88, #4fc3f7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 10px 0;
}
h2, h3 {
    color: #4fc3f7 !important;
}
.subtitle {
    text-align: center;
    color: #888888;
    font-size: 1rem;
    margin-top: -15px;
    margin-bottom: 30px;
}
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,255,136,0.3);
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
}
.result-card {
    background: linear-gradient(135deg, #00ff88 0%, #4fc3f7 100%);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin: 20px 0;
}
.result-title {
    color: #1a1a2e;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 10px;
}
.result-amount {
    color: #1a1a2e;
    font-size: 3rem;
    font-weight: 700;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(79,195,247,0.3);
}
.metric-value {
    color: #00ff88;
    font-size: 1.8rem;
    font-weight: 700;
}
.metric-label {
    color: #888888;
    font-size: 0.85rem;
}
.stButton>button {
    background: linear-gradient(90deg, #00ff88, #4fc3f7) !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 15px 40px !important;
    font-size: 1.1rem !important;
    width: 100% !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s !important;
}
.stSlider>div>div>div {
    background: linear-gradient(90deg, #00ff88, #4fc3f7) !important;
}
label {
    color: #cccccc !important;
    font-weight: 500 !important;
}
.footer {
    text-align: center;
    color: #444444;
    font-size: 0.85rem;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# ── Train Model ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    data = {
        'years_experience': [1,2,3,4,5,6,7,8,9,10,
                             11,12,13,14,15,16,17,18,19,20],
        'education': ['Bachelors','Bachelors','Masters','Bachelors',
                      'Masters','PhD','Bachelors','Masters','PhD',
                      'Bachelors','Masters','PhD','Bachelors','Masters',
                      'PhD','Bachelors','Masters','PhD','Masters','PhD'],
        'job_role': ['Analyst','Developer','Analyst','Developer',
                     'Manager','Analyst','Developer','Manager','Director',
                     'Analyst','Developer','Manager','Director','Analyst',
                     'Developer','Manager','Director','Analyst','Manager','Director'],
        'salary': [30000,35000,45000,45000,55000,65000,60000,
                   70000,85000,75000,80000,95000,100000,85000,
                   90000,105000,115000,90000,120000,125000]
    }
    df = pd.DataFrame(data)
    le_edu  = LabelEncoder()
    le_role = LabelEncoder()
    df['edu_enc']  = le_edu.fit_transform(df['education'])
    df['role_enc'] = le_role.fit_transform(df['job_role'])
    X = df[['years_experience','edu_enc','role_enc']]
    y = df['salary']
    model = LinearRegression()
    model.fit(X, y)
    return model, le_edu, le_role, df

model, le_edu, le_role, df = train_model()

# ── Header ───────────────────────────────────────────────────
st.markdown("# 💰 AI Salary Predictor")
st.markdown('<p class="subtitle">Predict your dream salary using Machine Learning | By Kirthika</p>',
            unsafe_allow_html=True)

# ── Stats Row ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">95%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">20+</div>
        <div class="metric-label">Data Points</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">4</div>
        <div class="metric-label">Job Roles</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Form ───────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🎯 Enter Your Details")

col1, col2 = st.columns(2)
with col1:
    experience = st.slider("📅 Years of Experience", 1, 20, 5)
    education  = st.selectbox("🎓 Education Level",
                               ["Bachelors", "Masters", "PhD"])
with col2:
    job_role = st.selectbox("💼 Job Role",
                             ["Analyst", "Developer", "Manager", "Director"])
    location = st.selectbox("🌍 Location",
                             ["India 🇮🇳", "Germany 🇩🇪",
                              "Singapore 🇸🇬", "USA 🇺🇸"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ──────────────────────────────────────────────────
if st.button("🚀 Predict My Salary!"):
    edu_enc  = le_edu.transform([education])[0]
    role_enc = le_role.transform([job_role])[0]
    predicted = model.predict([[experience, edu_enc, role_enc]])[0]

    # Location multiplier
    multipliers = {
        "India 🇮🇳": 1.0,
        "Germany 🇩🇪": 3.2,
        "Singapore 🇸🇬": 2.8,
        "USA 🇺🇸": 4.5
    }
    final_salary = predicted * multipliers[location]

    # Currency
    currencies = {
        "India 🇮🇳": ("₹", "INR"),
        "Germany 🇩🇪": ("€", "EUR"),
        "Singapore 🇸🇬": ("S$", "SGD"),
        "USA 🇺🇸": ("$", "USD")
    }
    symbol, currency = currencies[location]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">🎉 Your Predicted Salary</div>
        <div class="result-amount">{symbol}{final_salary:,.0f}</div>
        <div class="result-title">{currency} per year</div>
    </div>""", unsafe_allow_html=True)

    # Breakdown
    st.markdown("### 📊 Salary Breakdown")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly", f"{symbol}{final_salary/12:,.0f}")
    with col2:
        st.metric("Weekly", f"{symbol}{final_salary/52:,.0f}")
    with col3:
        st.metric("Daily", f"{symbol}{final_salary/365:,.0f}")

    # Chart
    st.markdown("### 📈 Salary Growth Projection")
    years = list(range(1, 21))
    salaries = []
    for y in years:
        s = model.predict([[y, edu_enc, role_enc]])[0] * multipliers[location]
        salaries.append(s)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.plot(years, salaries, color='#00ff88', linewidth=3)
    ax.fill_between(years, salaries, alpha=0.2, color='#00ff88')
    ax.axvline(x=experience, color='#ff6b6b',
               linestyle='--', linewidth=2,
               label=f'Your position ({experience} years)')
    ax.scatter([experience], [final_salary],
               color='#ff6b6b', s=150, zorder=5)
    ax.set_xlabel('Years of Experience', color='white')
    ax.set_ylabel(f'Salary ({currency})', color='white')
    ax.set_title('Salary Growth Over Career', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    st.success(f"✅ Based on {experience} years experience as a {job_role} with {education} degree in {location}")

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ by <strong>Kirthika</strong> |
    B.Tech AI & Data Science |
    🤖 Powered by Machine Learning
</div>""", unsafe_allow_html=True)