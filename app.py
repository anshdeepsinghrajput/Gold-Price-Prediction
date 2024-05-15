import streamlit as st
import numpy as np
import joblib

st.title("Gold Price Predictor")

s,u,sl,eu = st.columns(4)
with s:
    s = st.number_input('SPX',value=0.0,min_value=0.0,format='%.6f')
with u:
    u = st.number_input("USO",value=0.0,min_value=0.0,format='%.6f')
with sl:
    sl = st.number_input('SLV',value=0.0,min_value=0.0,format='%.3f')
with eu:
    eu = st.number_input('EUR/USD',value=0.0,min_value=0.0,format='%.6f')

arr = np.array([s,u,sl,eu])
rf = joblib.load('random_forest')
if st.button("Predict"):
    pred = rf.predict(arr.reshape(1,-1))
    st.success(f"Predicted gold price is {pred[0]}")





