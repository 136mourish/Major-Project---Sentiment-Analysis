import streamlit as st
import joblib
model = joblib.load('sentiment')
st.title('Sentiment Analysis')
ip = st.text_input("Enter the review")
op = model.predict([ip])
if st.button('Predict'):
  st.title(op[0])