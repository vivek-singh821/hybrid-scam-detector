import streamlit as st
import pickle
import re
def detect_urls(text):
    url_pattern = r"(https?://\S+|www\.\S+)"
    return re.findall(url_pattern, text)
def url_risk_score(url):
    risk = 0

    suspicious_domains = [".xyz", ".ru", ".tk", ".ml"]
    shorteners = ["bit.ly", "tinyurl", "goo.gl"]

    for domain in suspicious_domains:
        if domain in url:
            risk += 60

    for short in shorteners:
        if short in url:
            risk += 40

    if sum(c.isdigit() for c in url) > 5:
        risk += 30

    return min(risk, 100)    
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    font-size: 42px !important;
}
textarea {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Smart Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Sidebar
st.sidebar.title("ℹ About")
st.sidebar.write("This AI model detects whether a message is Spam or Not Spam.")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Accuracy: ~97%")
st.sidebar.write("Spam Recall: ~89%")

# Main Title
st.markdown("<h1 style='text-align: center;'>📩 Smart Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-powered SMS Classification System</p>", unsafe_allow_html=True)

st.divider()

# Input
user_input = st.text_area("Enter your message:", height=150)
st.caption("💡 Press Ctrl + Enter to analyze the message.")

if user_input.strip() != "":

    # 1️⃣ ML Prediction
    transformed = vectorizer.transform([user_input])
    prediction = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]

    spam_prob = probabilities[1] * 100
    ham_prob = probabilities[0] * 100

    # 2️⃣ URL Detection
    urls = detect_urls(user_input)

    if urls:
        url_score = url_risk_score(urls[0])
    else:
        url_score = 0

    # 3️⃣ Hybrid Risk Calculation
    final_risk = (spam_prob * 0.6) + (url_score * 0.4)

    if final_risk >= 75:
        risk_level = "🔴 HIGH RISK"
    elif final_risk >= 40:
        risk_level = "🟠 MEDIUM RISK"
    else:
        risk_level = "🟢 LOW RISK"

    # 4️⃣ Display ML Result
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("🚨 Spam Detected")
        else:
            st.success("✅ Not Spam")

    with col2:
        st.metric(label="Confidence", value=f"{max(spam_prob, ham_prob):.2f}%")

    st.divider()

    st.subheader("Probability Breakdown")
    st.write(f"Spam Probability: {spam_prob:.2f}%")
    st.write(f"Not Spam Probability: {ham_prob:.2f}%")

    st.progress(int(max(spam_prob, ham_prob)))

    # 5️⃣ Display Hybrid Risk
    st.divider()
    st.subheader("🛡 Hybrid Scam Risk Analysis")
    st.write(f"Final Scam Risk Score: {final_risk:.2f}%")
    st.write(f"Risk Level: {risk_level}")
    st.progress(int(final_risk))

    # 6️⃣ Show URL Info (if exists)
    if urls:
        st.divider()
        st.subheader("🔗 URL Risk Analysis")
        st.write(f"Detected URL: {urls[0]}")
        st.write(f"URL Risk Score: {url_score}%")