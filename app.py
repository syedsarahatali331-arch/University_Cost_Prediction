import streamlit as st, pandas as pd, joblib
st.title("International Education Cost Predictor")
model=joblib.load("model.pkl")
country=st.text_input("Country","USA")
city=st.text_input("City","Cambridge")
university=st.text_input("University","Harvard University")
program=st.text_input("Program","Computer Science")
level=st.selectbox("Level",["Bachelor","Master","PhD"])
duration=st.number_input("Duration (Years)",1.0,8.0,2.0)
tuition=st.number_input("Tuition (USD)",0.0,200000.0,50000.0)
living=st.number_input("Living Cost Index",0.0,200.0,80.0)
rent=st.number_input("Monthly Rent (USD)",0.0,10000.0,2000.0)
visa=st.number_input("Visa Fee (USD)",0.0,5000.0,200.0)
insurance=st.number_input("Annual Insurance (USD)",0.0,10000.0,1000.0)
exchange=st.number_input("Exchange Rate",0.01,200.0,1.0)
if st.button("Predict"):
    x=pd.DataFrame([{"Country":country,"City":city,"University":university,"Program":program,"Level":level,
                     "Duration_Years":duration,"Tuition_USD":tuition,"Living_Cost_Index":living,
                     "Rent_USD":rent,"Visa_Fee_USD":visa,"Insurance_USD":insurance,"Exchange_Rate":exchange}])
    st.metric("Estimated total cost",f"${model.predict(x)[0]:,.0f}")
