import streamlit as st
import pickle
import re
import string
#ui

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
    <h1 style='text-align: center;'>📰 Fake News Detection System</h1>
    <p style='text-align: center; color: grey;'>
    AI Powered | Logistic Regression | TF-IDF
    </p>
""", unsafe_allow_html=True)

st.divider()

#load saved model
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

#text cleaning funtiom
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"\d+","",text)
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

#page title
st.title("Fake News Detection App")

#inoput box
news_input = st.text_area("Enter News Text Here")

#button
if st.button("check news"):
    cleaned = clean_text(news_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    probability = model.predict_proba(vectorized)[0]

    if prediction[0] == 1:
        st.success("🟢 This is Real News")
        confidence = probability[1] * 100
    else:
        st.error("🔴 This is Fake News")
        confidence = probability[0] * 100

    st.write(f"### Confidence Score: {round(confidence,2)}%")

    st.progress(int(confidence))



