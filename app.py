import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")

# Text input
news_text = st.text_area("Paste a news article or headline:")

if st.button("Check"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        X = vectorizer.transform([news_text])
        prediction = model.predict(X)[0]

        if prediction == "FAKE":
            st.error("❌ This news seems FAKE!")
        else:
            st.success("✅ This news seems REAL!")