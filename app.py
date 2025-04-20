import streamlit as st
import helper
import pickle as pkl
import joblib

model = joblib.load('model_rf.pkl')

st.header("Duplicate Question Pairs")

q1 = st.text_input('Enter Question 1')
q2 = st.text_input('Enter Question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    if result:
        st.header("Duplicates")
    else:
        st.header("Not Duplicates")