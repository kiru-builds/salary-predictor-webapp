import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Navigator | By Kirthika",
    page_icon="🚀",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00ff88, #4fc3f7, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 20px 0 5px 0;
}
.subtitle {
    text-align: center;
    color: #888888;
    font-size: 1.1rem;
    margin-bottom: 30px;
}
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 20px;
    padding: 25px;
    margin: 10px 0;
}
.result-card {
    background: linear-gradient(135deg, #00ff88, #4fc3f7);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}
.result-amount {
    color: #1a1a2e;
    font-size: 2.5rem;
    font-weight: 700;
}
.result-label {
    color: #1a1a2e;
    font-size: 1rem;
    font-weight: 600;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(79,195,247,0.3);
}
.metric-value { color: #00ff88; font-size: 1.8rem; font-weight: 700; }
.metric-label { color: #888888; font-size: 0.85rem; }
.skill-tag {
    display: inline-block;
    background: rgba(0,255,136,0.15);
    border: 1px solid #00ff88;
    color: #00ff88;
    padding: 5px 15px;
    border-radius: 20px;
    margin: 5px;
    font-size: 0.85rem;
}
.skill-tag-blue {
    display: inline-block;
    background: rgba(79,195,247,0.15);
    border: 1px solid #4fc3f7;
    color: #4fc3f7;
    padding: 5px 15px;
    border-radius: 20px;
    margin: 5px;
    font-size: 0.85rem;
}
.stButton>button {
    background: linear-gradient(90deg, #00ff88, #4fc3f7) !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 12px 30px !important;
    font-size: 1rem !important;
    width: 100% !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    color: #888888 !important;
    font-size: 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: #00ff88 !important;
    border-bottom: 2px solid #00ff88 !important;
}
label { color: #cccccc !important; }
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

# ── Data ─────────────────────────────────────────────────────
salary_data = {
    'Data Scientist': {
        'India 🇮🇳':       {'min': 800000,  'max': 2500000,  'currency': '₹'},
        'Germany 🇩🇪':     {'min': 55000,   'max': 90000,    'currency': '€'},
        'Singapore 🇸🇬':   {'min': 70000,   'max': 120000,   'currency': 'S$'},
        'USA 🇺🇸':         {'min': 100000,  'max': 180000,   'currency': '$'},
        'UK 🇬🇧':          {'min': 50000,   'max': 90000,    'currency': '£'},
        'Canada 🇨🇦':      {'min': 80000,   'max': 130000,   'currency': 'CA$'},
    },
    'ML Engineer': {
        'India 🇮🇳':       {'min': 1000000, 'max': 3000000,  'currency': '₹'},
        'Germany 🇩🇪':     {'min': 60000,   'max': 100000,   'currency': '€'},
        'Singapore 🇸🇬':   {'min': 80000,   'max': 140000,   'currency': 'S$'},
        'USA 🇺🇸':         {'min': 120000,  'max': 220000,   'currency': '$'},
        'UK 🇬🇧':          {'min': 60000,   'max': 110000,   'currency': '£'},
        'Canada 🇨🇦':      {'min': 90000,   'max': 150000,   'currency': 'CA$'},
    },
    'Data Analyst': {
        'India 🇮🇳':       {'min': 400000,  'max': 1200000,  'currency': '₹'},
        'Germany 🇩🇪':     {'min': 40000,   'max': 65000,    'currency': '€'},
        'Singapore 🇸🇬':   {'min': 50000,   'max': 85000,    'currency': 'S$'},
        'USA 🇺🇸':         {'min': 65000,   'max': 110000,   'currency': '$'},
        'UK 🇬🇧':          {'min': 35000,   'max': 60000,    'currency': '£'},
        'Canada 🇨🇦':      {'min': 55000,   'max': 90000,    'currency': 'CA$'},
    },
    'AI Engineer': {
        'India 🇮🇳':       {'min': 1200000, 'max': 4000000,  'currency': '₹'},
        'Germany 🇩🇪':     {'min': 65000,   'max': 110000,   'currency': '€'},
        'Singapore 🇸🇬':   {'min': 90000,   'max': 160000,   'currency': 'S$'},
        'USA 🇺🇸':         {'min': 140000,  'max': 250000,   'currency': '$'},
        'UK 🇬🇧':          {'min': 65000,   'max': 120000,   'currency': '£'},
        'Canada 🇨🇦':      {'min': 100000,  'max': 170000,   'currency': 'CA$'},
    },
    'Software Engineer': {
        'India 🇮🇳':       {'min': 500000,  'max': 2000000,  'currency': '₹'},
        'Germany 🇩🇪':     {'min': 50000,   'max': 85000,    'currency': '€'},
        'Singapore 🇸🇬':   {'min': 65000,   'max': 110000,   'currency': 'S$'},
        'USA 🇺🇸':         {'min': 110000,  'max': 200000,   'currency': '$'},
        'UK 🇬🇧':          {'min': 45000,   'max': 85000,    'currency': '£'},
        'Canada 🇨🇦':      {'min': 80000,   'max': 140000,   'currency': 'CA$'},
    },
}

skills_data = {
    'Data Scientist': {
        'must': ['Python', 'Statistics', 'Machine Learning', 'SQL', 'Pandas', 'NumPy'],
        'good': ['TensorFlow', 'PyTorch', 'Tableau', 'R', 'Spark', 'Docker'],
        'soft': ['Communication', 'Problem Solving', 'Storytelling with Data']
    },
    'ML Engineer': {
        'must': ['Python', 'TensorFlow', 'PyTorch', 'MLOps', 'Docker', 'Kubernetes'],
        'good': ['AWS/GCP/Azure', 'Spark', 'Kafka', 'FastAPI', 'CI/CD'],
        'soft': ['System Design', 'Collaboration', 'Attention to Detail']
    },
    'Data Analyst': {
        'must': ['SQL', 'Excel', 'Python/R', 'Tableau/PowerBI', 'Statistics'],
        'good': ['Google Analytics', 'Looker', 'Airflow', 'Pandas'],
        'soft': ['Critical Thinking', 'Communication', 'Business Acumen']
    },
    'AI Engineer': {
        'must': ['Python', 'Deep Learning', 'LLMs', 'NLP', 'Computer Vision', 'MLOps'],
        'good': ['LangChain', 'HuggingFace', 'OpenAI API', 'Vector DBs', 'RAG'],
        'soft': ['Research mindset', 'Creativity', 'Fast Learning']
    },
    'Software Engineer': {
        'must': ['Python/Java/C++', 'Data Structures', 'Algorithms', 'Git', 'SQL'],
        'good': ['Cloud (AWS/GCP)', 'Microservices', 'React', 'Docker', 'REST APIs'],
        'soft': ['Team Collaboration', 'Code Review', 'Agile/Scrum']
    },
}

cost_of_living = {
    'India 🇮🇳':     {'rent': 15000,  'food': 8000,   'transport': 3000,  'currency': '₹', 'monthly': 35000},
    'Germany 🇩🇪':   {'rent': 900,    'food': 400,    'transport': 80,    'currency': '€', 'monthly': 1500},
    'Singapore 🇸🇬': {'rent': 1800,   'food': 600,    'transport': 120,   'currency': 'S$','monthly': 3000},
    'USA 🇺🇸':       {'rent': 2000,   'food': 500,    'transport': 150,   'currency': '$', 'monthly': 3500},
    'UK 🇬🇧':        {'rent': 1200,   'food': 400,    'transport': 150,   'currency': '£', 'monthly': 2200},
    'Canada 🇨🇦':    {'rent': 1500,   'food': 450,    'transport': 120,   'currency': 'CA$','monthly': 2500},
}

internship_platforms = {
    'India 🇮🇳':     ['Internshala', 'LinkedIn', 'Naukri', 'AngelList', 'Indeed'],
    'Germany 🇩🇪':   ['LinkedIn', 'XING', 'StepStone', 'Indeed DE', 'Make it in Germany'],
    'Singapore 🇸🇬': ['LinkedIn', 'JobStreet', 'MyCareersFuture', 'Glassdoor', 'Indeed SG'],
    'USA 🇺🇸':       ['LinkedIn', 'Indeed', 'Glassdoor', 'Handshake', 'WayUp'],
    'UK 🇬🇧':        ['LinkedIn', 'Gradcracker', 'RateMyPlacement', 'Indeed UK', 'Prospects'],
    'Canada 🇨🇦':    ['LinkedIn', 'Indeed CA', 'Glassdoor', 'Workopolis', 'Monster CA'],
}

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🚀 AI Career Navigator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your complete guide to salaries, skills & opportunities worldwide | Built by Kirthika</div>', unsafe_allow_html=True)

# ── Stats ────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Countries</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-value">5</div><div class="metric-label">Job Roles</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-value">50+</div><div class="metric-label">Skills Mapped</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-value">Free</div><div class="metric-label">Always</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Salary Explorer",
    "🌍 Country Comparator",
    "📚 Skills Roadmap",
    "💼 Profile Scorer",
    "🎯 Internship Guide"
])

# ══ TAB 1: Salary Explorer ═══════════════════════════════════
with tab1:
    st.markdown("### 💰 Explore Salaries by Role & Country")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        role = st.selectbox("💼 Job Role", list(salary_data.keys()))
    with col2:
        country = st.selectbox("🌍 Country", list(cost_of_living.keys()))
    with col3:
        experience = st.selectbox("📅 Experience Level",
                                   ["Fresher (0-1 yr)", "Junior (1-3 yrs)",
                                    "Mid (3-7 yrs)", "Senior (7+ yrs)"])

    exp_multiplier = {"Fresher (0-1 yr)": 0.7, "Junior (1-3 yrs)": 0.9,
                      "Mid (3-7 yrs)": 1.2, "Senior (7+ yrs)": 1.6}
    mult = exp_multiplier[experience]

    if st.button("💰 Show My Salary Range!"):
        data = salary_data[role][country]
        min_sal = data['min'] * mult
        max_sal = data['max'] * mult
        avg_sal = (min_sal + max_sal) / 2
        curr    = data['currency']
        col = salary_data[role][country]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">💰 {role} in {country}</div>
            <div class="result-amount">{curr}{avg_sal:,.0f}</div>
            <div class="result-label">Average Annual Salary</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Minimum", f"{curr}{min_sal:,.0f}")
        with c2:
            st.metric("Average", f"{curr}{avg_sal:,.0f}")
        with c3:
            st.metric("Maximum", f"{curr}{max_sal:,.0f}")

        # Monthly breakdown
        monthly = avg_sal / 12
        col_info = cost_of_living[country]
        savings  = monthly - col_info['monthly']

        st.markdown("#### 📊 Monthly Breakdown")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Monthly Income", f"{curr}{monthly:,.0f}")
        with c2:
            st.metric("Monthly Expenses", f"{curr}{col_info['monthly']:,}")
        with c3:
            st.metric("Monthly Savings", f"{curr}{savings:,.0f}",
                      delta="Positive" if savings > 0 else "Negative")

        # Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        categories = ['Min Salary', 'Avg Salary', 'Max Salary', 'Monthly Expenses x12']
        values = [min_sal, avg_sal, max_sal, col_info['monthly']*12]
        colors = ['#4fc3f7', '#00ff88', '#ff6b6b', '#ffa726']
        bars = ax.bar(categories, values, color=colors, edgecolor='none', width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.01,
                    f'{curr}{val:,.0f}', ha='center',
                    color='white', fontsize=9)
        ax.set_title(f'{role} Salary vs Cost of Living in {country}',
                     color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ══ TAB 2: Country Comparator ════════════════════════════════
with tab2:
    st.markdown("### 🌍 Compare Salaries Across Countries")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    role2 = st.selectbox("💼 Select Role to Compare",
                          list(salary_data.keys()), key='role2')

    if st.button("🌍 Compare All Countries!"):
        countries  = list(salary_data[role2].keys())
        avg_sals   = [(salary_data[role2][c]['min'] +
                       salary_data[role2][c]['max']) / 2
                      for c in countries]
        currencies = [salary_data[role2][c]['currency'] for c in countries]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#1a1a2e')

        # Bar chart
        colors = ['#00ff88','#4fc3f7','#ff6b6b','#ffa726','#ab47bc','#26c6da']
        axes[0].set_facecolor('#16213e')
        bars = axes[0].bar(range(len(countries)), avg_sals,
                           color=colors, edgecolor='none', width=0.6)
        axes[0].set_xticks(range(len(countries)))
        axes[0].set_xticklabels(countries, rotation=30, ha='right', color='white')
        axes[0].set_title(f'Average {role2} Salary by Country',
                          color='white', fontweight='bold')
        axes[0].tick_params(colors='white')
        axes[0].spines['bottom'].set_color('#444')
        axes[0].spines['left'].set_color('#444')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # Pie chart
        axes[1].set_facecolor('#16213e')
        axes[1].pie(avg_sals, labels=countries, colors=colors,
                    autopct='%1.0f%%', startangle=90,
                    textprops={'color': 'white'})
        axes[1].set_title('Salary Distribution', color='white', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        # Table
        st.markdown("#### 📋 Detailed Comparison")
        table_data = []
        for c, s, curr in zip(countries, avg_sals, currencies):
            col_info = cost_of_living[c]
            monthly  = s / 12
            savings  = monthly - col_info['monthly']
            table_data.append({
                'Country': c,
                'Avg Annual Salary': f"{curr}{s:,.0f}",
                'Monthly Salary': f"{curr}{monthly:,.0f}",
                'Monthly Expenses': f"{curr}{col_info['monthly']:,}",
                'Monthly Savings': f"{curr}{savings:,.0f}"
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══ TAB 3: Skills Roadmap ════════════════════════════════════
with tab3:
    st.markdown("### 📚 Skills Roadmap for Your Dream Role")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    role3 = st.selectbox("💼 Choose Your Target Role",
                          list(skills_data.keys()), key='role3')
    skills = skills_data[role3]

    st.markdown(f"#### 🎯 Roadmap for {role3}")

    st.markdown("*✅ Must-Have Skills:*")
    tags = ''.join([f'<span class="skill-tag">{s}</span>'
                    for s in skills['must']])
    st.markdown(tags, unsafe_allow_html=True)

    st.markdown("*⭐ Good to Have:*")
    tags2 = ''.join([f'<span class="skill-tag-blue">{s}</span>'
                     for s in skills['good']])
    st.markdown(tags2, unsafe_allow_html=True)

    st.markdown("*🤝 Soft Skills:*")
    for s in skills['soft']:
        st.markdown(f"- {s}")

    # Learning path
    st.markdown("#### 📅 6-Month Learning Plan")
    months = {
        "Month 1-2": f"Master {skills['must'][0]} and {skills['must'][1]}",
        "Month 3-4": f"Learn {skills['must'][2]} and {skills['must'][3]}",
        "Month 5":   f"Build 2-3 projects using {skills['must'][0]}",
        "Month 6":   "Apply for internships & update GitHub/LinkedIn"
    }
    for month, plan in months.items():
        st.info(f"*{month}:* {plan}")

    st.markdown('</div>', unsafe_allow_html=True)

# ══ TAB 4: Profile Scorer ════════════════════════════════════
with tab4:
    st.markdown("### 💼 Score Your Career Profile")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("Answer honestly to get your career readiness score!")

    q1 = st.slider("📁 GitHub Projects (how many?)", 0, 10, 0)
    q2 = st.slider("📜 Certifications completed", 0, 10, 0)
    q3 = st.selectbox("🎓 Education", ["High School", "Diploma",
                                        "Bachelor's", "Master's", "PhD"])
    q4 = st.slider("💼 Internships/Work Experience (months)", 0, 24, 0)
    q5 = st.selectbox("🌐 LinkedIn profile?", ["No", "Basic", "Optimized"])
    q6 = st.selectbox("🗣️ English proficiency", ["Beginner", "Intermediate",
                                                   "Advanced", "Native"])

    if st.button("🎯 Calculate My Score!"):
        score = 0
        score += min(q1 * 8, 30)
        score += min(q2 * 5, 20)
        edu_scores = {"High School": 5, "Diploma": 8,
                      "Bachelor's": 15, "Master's": 18, "PhD": 20}
        score += edu_scores[q3]
        score += min(q4 * 1.5, 15)
        linkedin_scores = {"No": 0, "Basic": 5, "Optimized": 10}
        score += linkedin_scores[q5]
        eng_scores = {"Beginner": 0, "Intermediate": 3,
                      "Advanced": 7, "Native": 10}
        score += eng_scores[q6]
        score = min(int(score), 100)

        if score >= 80:
            color = "#00ff88"
            msg   = "🌟 Excellent! You're ready for international opportunities!"
            level = "Job Ready"
        elif score >= 60:
            color = "#4fc3f7"
            msg   = "👍 Good profile! A few more projects and you're set!"
            level = "Almost Ready"
        elif score >= 40:
            color = "#ffa726"
            msg   = "📈 Keep building! Focus on projects and certifications!"
            level = "In Progress"
        else:
            color = "#ff6b6b"
            msg   = "🚀 Just getting started! Follow the Skills Roadmap tab!"
            level = "Getting Started"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}44);
             border:2px solid {color}; border-radius:20px;
             padding:30px; text-align:center;">
            <div style="color:{color}; font-size:3rem; font-weight:700;">{score}/100</div>
            <div style="color:{color}; font-size:1.2rem; font-weight:600;">{level}</div>
            <div style="color:white; margin-top:10px;">{msg}</div>
        </div>""", unsafe_allow_html=True)

        # Breakdown chart
        st.markdown("#### 📊 Score Breakdown")
        categories = ['Projects', 'Certifications',
                      'Education', 'Experience', 'LinkedIn', 'English']
        values = [min(q1*8, 30), min(q2*5, 20),
                  edu_scores[q3], min(q4*1.5, 15),
                  linkedin_scores[q5], eng_scores[q6]]
        max_vals = [30, 20, 20, 15, 10, 10]

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        x = range(len(categories))
        ax.bar(x, max_vals, color='#333333', edgecolor='none', width=0.5, label='Max')
        ax.bar(x, values,   color='#00ff88', edgecolor='none', width=0.5, label='Your Score')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, color='white')
        ax.set_title('Your Score Breakdown', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ══ TAB 5: Internship Guide ══════════════════════════════════
with tab5:
    st.markdown("### 🎯 Internship Finder Guide")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    country5 = st.selectbox("🌍 Target Country",
                              list(internship_platforms.keys()), key='c5')
    role5    = st.selectbox("💼 Target Role",
                              list(skills_data.keys()), key='r5')

    if st.button("🎯 Show My Internship Guide!"):
        platforms = internship_platforms[country5]
        skills5   = skills_data[role5]['must']
        col_info5 = cost_of_living[country5]
        curr5     = col_info5['currency']

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🌐 Top Platforms to Apply")
            for i, p in enumerate(platforms, 1):
                st.markdown(f"*{i}.* {p}")

        with c2:
            st.markdown("#### 💰 Expected Stipend")
            sal_range = salary_data[role5][country5]
            intern_min = sal_range['min'] * 0.3 / 12
            intern_max = sal_range['min'] * 0.5 / 12
            st.metric("Monthly Stipend Range",
                      f"{curr5}{intern_min:,.0f} - {curr5}{intern_max:,.0f}")
            st.metric("Monthly Living Cost",
                      f"{curr5}{col_info5['monthly']:,}")

        st.markdown("#### ✅ Skills to Highlight in Application")
        tags = ''.join([f'<span class="skill-tag">{s}</span>'
                        for s in skills5])
        st.markdown(tags, unsafe_allow_html=True)

        st.markdown("#### 📝 Application Checklist")
        checklist = [
            "✅ Updated Resume (1 page, English)",
            "✅ LinkedIn profile optimized",
            f"✅ GitHub with {role5} projects",
            "✅ Cover letter written",
            "✅ References from professors",
            "✅ Valid passport (6+ months)",
            "✅ Bank statement ready",
        ]
        for item in checklist:
            st.markdown(item)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ by <strong>Kirthika</strong> |
    B.Tech AI & Data Science |
    🚀 Helping students navigate their AI career worldwide
</div>""", unsafe_allow_html=True)
