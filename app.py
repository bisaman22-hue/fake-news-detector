import streamlit as st
import pickle

# Load model
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Page settings
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# CSS Styling
st.markdown("""
<style>

.main {
    background-color: #f4f6f9;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #ff4b4b;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 20px;
}

.stButton>button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">📰 Fake News Detector</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">AI & Data Analytics Project</p>',
    unsafe_allow_html=True
)

# Input
news = st.text_area(
    "Enter News Article",
    height=200,
    placeholder="Paste news here..."
)

# Prediction
if st.button("Detect News"):

    if news.strip() == "":
        st.warning("Please enter news.")

    else:

        news_vector = vectorizer.transform([news])

        result = model.predict(news_vector)

        if result[0] == "FAKE":
            st.error("❌ FAKE NEWS DETECTED")

        else:
            st.success("✅ REAL NEWS DETECTED")

# Footer
st.write("---")
st.write("Made using AI, Machine Learning & Streamlit")