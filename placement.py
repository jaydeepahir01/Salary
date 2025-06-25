import numpy as np
import pickle
import streamlit as st

placement_model = pickle.load(open("placement_model.sav", "rb"))

def Predict(input_data):
    inpute_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = inpute_data_as_numpy_array.reshape(1, -1)
    prediction = placement_model.predict(input_data_reshaped)
    return prediction[0]

st.title("Placement Prediction")

col1, col2 = st.columns(2)

with col1:
    gender = st.text_input('Enter Gender (0 for Male, 1 for Female)')
    ssc_p = st.text_input('Enter Your 10th %')
    hsc_p = st.text_input('Enter Your 12th %')
    hsc_s = st.text_input('Enter Stream (0-Commerce, 1-Science, 2-Arts)')
    degree_p = st.text_input('Enter Your Degree %')

with col2:
    degree_t = st.text_input('Enter Degree Type (0-Sci&Tech, 1-Comm&Mgmt, 2-Others)')
    workex = st.text_input('Work Experience (1-Yes, 0-No)')
    etest_p = st.text_input('Enter Etest %')
    specialisation = st.text_input('Specialisation (0-Mkt&Fin, 1-Mkt&HR)')
    mba_p = st.text_input('Enter MBA %')

if st.button('Predict salary'):
    try:
        input_data = (
            float(gender), float(ssc_p), float(hsc_p), float(hsc_s), float(degree_p),
            float(degree_t), float(workex), float(etest_p), float(specialisation), float(mba_p)
        )
        w_pre = Predict(input_data)
        st.success(f'Prediction: {w_pre}')
    except ValueError:
        st.error("Please enter numeric values only.")
