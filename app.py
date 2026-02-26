import streamlit as st
import pickle
import re
import string

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
        st.write("Confidence:", round(probability[1]*100, 2), "%")
    else:
        st.error("🔴 This is Fake News")
        st.write("Confidence:", round(probability[0]*100, 2), "%")


