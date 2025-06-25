import numpy as np
import pickle
import streamlit as st

# Load the trained model
try:
    placement_model = pickle.load(open("placement_model.sav", "rb"))
except Exception as e:
    st.error("Error loading model: " + str(e))
    st.stop()

# Prediction function
def Predict(input_data):
    try:
        input_array = np.asarray(input_data, dtype=np.float32)
        if input_array.shape[0] != placement_model.n_features_in_:
            raise ValueError(f"Expected {placement_model.n_features_in_} features, but got {input_array.shape[0]}")
        input_reshaped = input_array.reshape(1, -1)
        prediction = placement_model.predict(input_reshaped)
        return prediction[0]
    except Exception as e:
        st.error("Prediction error: " + str(e))
        return None

# App title
st.title("🎓 Placement Prediction")

# Input columns
col1, col2 = st.columns(2)

with col1:
    gender = st.text_input('Gender (0 = Male, 1 = Female)')
    ssc_p = st.text_input('10th Percentage')
    hsc_p = st.text_input('12th Percentage')
    hsc_s = st.text_input('Stream (0-Commerce, 1-Science, 2-Arts)')
    degree_p = st.text_input('Degree Percentage')

with col2:
    degree_t = st.text_input('Degree Type (0-Sci&Tech, 1-Comm&Mgmt, 2-Others)')
    workex = st.text_input('Work Experience (0-No, 1-Yes)')
    etest_p = st.text_input('Etest Percentage')
    specialisation = st.text_input('Specialisation (0-Mkt&Fin, 1-Mkt&HR)')
    mba_p = st.text_input('MBA Percentage')

# Predict button
if st.button('Predict Placement'):
    try:
        input_data = [
            float(gender), float(ssc_p), float(hsc_p), float(hsc_s), float(degree_p),
            float(degree_t), float(workex), float(etest_p), float(specialisation), float(mba_p)
        ]
        result = Predict(input_data)
        if result == 1:
            st.success("✅ The student is likely to be placed.")
        elif result == 0:
            st.warning("❌ The student is unlikely to be placed.")
        elif result is None:
            pass  # already handled in Predict
        else:
            st.info(f"Model output: {result}")
    except ValueError:
        st.error("⚠️ Please enter only numeric values in all fields.")
