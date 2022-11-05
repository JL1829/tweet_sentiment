import streamlit as st
from MLEngineBert import MlEngineBert

st.title("Tweet Sentiment Analysis")
st.subheader("By NUS-ISS PLP Group 12")

engine = MlEngineBert()

text_input = st.text_area("Enter the COVID-19 Related Tweet: ", height=50)
if st.button("Predict"):
    returns = engine(text_input)
    st.write("These are sentiment:")
    st.write(returns)