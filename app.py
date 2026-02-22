import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="AI Career Guidance System",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# Extended Training Data
# ----------------------------
data = {
    'Maths':[85,70,90,60,40,88,65,75,92,55,78,82],
    'Physics':[85,60,88,65,45,80,60,70,95,55,72,85],
    'Chemistry':[80,70,85,60,50,75,65,68,92,55,70,80],
    'Computer':[90,50,95,40,30,85,60,65,98,50,88,90],
    'Biology':[70,65,60,80,85,75,70,72,65,80,60,58],
    'Accounts':[0,0,0,0,0,0,80,85,75,65,90,88],
    'Economics':[0,0,0,0,0,0,70,80,75,60,85,84],
    'Business':[0,0,0,0,0,0,75,85,80,60,88,82],
    'English':[75,80,90,60,50,78,70,72,85,60,75,77],
    'Coding':[1,0,1,0,0,1,0,0,1,0,1,1],
    'Communication':[1,1,0,1,1,0,1,1,0,1,1,1],
    'Creativity':[0,1,0,1,1,0,1,1,0,1,1,0],
    'Art':[0,1,0,1,1,0,1,1,0,1,1,0],
    'Sports':[0,0,0,0,1,0,1,1,0,1,0,0],
    'Leadership':[0,1,0,1,0,0,1,1,0,1,1,0],
    'Teamwork':[1,1,0,1,1,0,1,1,0,1,1,1],
    'ProblemSolving':[1,0,1,0,0,1,1,0,1,0,1,1],
    'Career':[
        'Software Engineer','Teacher','Data Scientist','Designer','Clerk',
        'Software Engineer','Doctor','Accountant','Entrepreneur',
        'Sports Coach','Cyber Security Analyst','Business Analyst'
    ]
}

df = pd.DataFrame(data)

features = df.columns[:-1]
X = df[features]
y = df['Career']

model = DecisionTreeClassifier()
model.fit(X, y)

# ----------------------------
# UI Layout
# ----------------------------
st.title("🎓 AI Career Guidance System")
st.markdown("### Discover the Best Career Path Based on Your Skills & Marks")

col1, col2 = st.columns(2)

with col1:
    stream = st.selectbox(
        "Select Your Stream",
        ["Computer Science", "Biology", "Commerce", "Diploma", "Arts"]
    )

with col2:
    st.info("Fill your marks and select skills to get prediction.")

st.markdown("---")

# ----------------------------
# Marks Input
# ----------------------------
st.subheader("📚 Academic Marks")

marks = {}
subject_list = ['Maths','Physics','Chemistry','Computer',
                'Biology','Accounts','Economics',
                'Business','English']

for subject in subject_list:
    marks[subject] = st.slider(f"{subject}", 0, 100, 50)

# ----------------------------
# Skills Section
# ----------------------------
st.subheader("🧠 Skills & Interests")

skills = {}
skill_list = ['Coding','Communication','Creativity',
              'Art','Sports','Leadership',
              'Teamwork','ProblemSolving']

cols = st.columns(4)
for i, skill in enumerate(skill_list):
    with cols[i % 4]:
        skills[skill] = st.checkbox(skill)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Career"):

    input_data = list(marks.values()) + [int(v) for v in skills.values()]
    probabilities = model.predict_proba([input_data])[0]
    classes = model.classes_

    top_indices = probabilities.argsort()[-3:][::-1]

    st.success("### 🎯 Top 3 Career Suggestions")

    for i in top_indices:
        st.write(f"✅ *{classes[i]}*")

    st.markdown("---")

    st.subheader("📌 Career Guidance Roadmap")

    career_info = {
        "Software Engineer": "Learn Programming → Build Projects → Internship → IT Company Job",
        "Doctor": "NEET Exam → MBBS → Internship → Hospital Practice",
        "Accountant": "B.Com → CA / CMA → Finance Sector Job",
        "Entrepreneur": "Business Knowledge → Startup Idea → Funding → Launch Business",
        "Data Scientist": "Python → Statistics → Machine Learning → Data Projects",
        "Teacher": "Bachelor Degree → B.Ed → School/College Job",
        "Designer": "Design Tools → Portfolio → Creative Agency Job",
        "Cyber Security Analyst": "Networking → Ethical Hacking → Security Certification",
        "Business Analyst": "Business Knowledge → Data Skills → Corporate Role",
        "Sports Coach": "Sports Training → Certification → Coaching Role",
        "Clerk": "Government Exams → Office Administration Job"
    }

    for i in top_indices:
        career = classes[i]
        if career in career_info:
            st.info(f"*{career} Roadmap:* {career_info[career]}")
