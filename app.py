# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model

# # Load the trained model, transformer, and label encoder
# model = load_model("my_model.h5")
# transformer = joblib.load("transformer_X.pkl")
# label_encoder = joblib.load("label_encoder.pkl")

# # Set the title of the app
# st.title('Cancer Risk Prediction')

# # Create the user interface for inputting data
# st.header('Enter Patient Information')

# # Create columns for a more organized layout
# col1, col2, col3 = st.columns(3)

# with col1:
#     age = st.number_input('Age', min_value=0, max_value=120, value=25)
#     bmi = st.number_input('BMI', min_value=0.0, value=22.0)
#     sleep_hours = st.number_input('Sleep Hours', min_value=0.0, max_value=24.0, value=7.0)


# with col2:
#     smoker = st.selectbox('Smoker', ('No', 'Yes'))
#     alcohol_consumption = st.selectbox('Alcohol Consumption', ('Unknown', 'Low', 'Moderate', 'High'))
#     diet_type = st.selectbox('Diet Type', ('Fatty', 'Mixed', 'Healthy'))
#     physical_activity_level = st.selectbox('Physical Activity Level', ('Low', 'Moderate', 'High'))

# with col3:
#     family_history = st.selectbox('Family History', ('No', 'Yes'))
#     mental_stress_level = st.selectbox('Mental Stress Level', ('Low', 'Medium', 'High'))
#     regular_health_checkup = st.selectbox('Regular Health Checkup', ('No', 'Yes'))
#     prostate_exam_done = st.selectbox('Prostate Exam Done', ('No', 'Yes'))


# # Create a button to make predictions
# if st.button('Predict Cancer Risk'):
#     # Create a dataframe from the user's input
#     input_data = pd.DataFrame({
#         'age': [age],
#         'bmi': [bmi],
#         'smoker': [smoker],
#         'alcohol_consumption': [alcohol_consumption],
#         'diet_type': [diet_type],
#         'physical_activity_level': [physical_activity_level],
#         'family_history': [family_history],
#         'mental_stress_level': [mental_stress_level],
#         'sleep_hours': [sleep_hours],
#         'regular_health_checkup': [regular_health_checkup],
#         'prostate_exam_done': [prostate_exam_done]
#     })

#     # Transform the input data using the loaded transformer
#     input_data_transformed = transformer.transform(input_data)

#     # Make a prediction
#     prediction_proba = model.predict(input_data_transformed)
#     prediction = np.argmax(prediction_proba, axis=1)
#     confidence = np.max(prediction_proba) * 100


#     # Decode the prediction to the original label
#     predicted_risk_level = label_encoder.inverse_transform(prediction)[0]

#     # Display the prediction
#     st.subheader('Prediction')
#     if predicted_risk_level == 'High':
#         st.error(f'The predicted cancer risk level is: **{predicted_risk_level}**')
#     elif predicted_risk_level == 'Medium':
#         st.warning(f'The predicted cancer risk level is: **{predicted_risk_level}**')
#     else:
#         st.success(f'The predicted cancer risk level is: **{predicted_risk_level}**')
    
#     st.write(f"Confidence: **{confidence:.2f}%**")






























import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained model, transformer, and label encoder
model = load_model("my_model.h5")
transformer = joblib.load("transformer_X.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Set the title of the app
st.title('Cancer Risk Prediction')

# Create the user interface for inputting data
st.header('Enter Patient Information')

# Create columns for a more organized layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    bmi = st.number_input('BMI', min_value=0.0, value=22.0)
    sleep_hours = st.number_input('Sleep Hours', min_value=0.0, max_value=24.0, value=7.0)


with col2:
    smoker = st.selectbox('Smoker', ('No', 'Yes'))
    alcohol_consumption = st.selectbox('Alcohol Consumption', ('Unknown', 'Low', 'Moderate', 'High'))
    diet_type = st.selectbox('Diet Type', ('Fatty', 'Mixed', 'Healthy'))
    physical_activity_level = st.selectbox('Physical Activity Level', ('Low', 'Moderate', 'High'))

with col3:
    family_history = st.selectbox('Family History', ('No', 'Yes'))
    mental_stress_level = st.selectbox('Mental Stress Level', ('Low', 'Medium', 'High'))
    regular_health_checkup = st.selectbox('Regular Health Checkup', ('No', 'Yes'))
    prostate_exam_done = st.selectbox('Prostate Exam Done', ('No', 'Yes'))


# Create a button to make predictions
if st.button('Predict Cancer Risk'):
    # Define the exact order of columns your model was trained on
    column_order = [
        'age', 'bmi', 'smoker', 'alcohol_consumption', 'diet_type', 
        'physical_activity_level', 'family_history', 'mental_stress_level', 
        'sleep_hours', 'regular_health_checkup', 'prostate_exam_done'
    ]
    
    # Create a dataframe from the user's input
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'smoker': [smoker],
        'alcohol_consumption': [alcohol_consumption],
        'diet_type': [diet_type],
        'physical_activity_level': [physical_activity_level],
        'family_history': [family_history],
        'mental_stress_level': [mental_stress_level],
        'sleep_hours': [sleep_hours],
        'regular_health_checkup': [regular_health_checkup],
        'prostate_exam_done': [prostate_exam_done]
    })

    # Ensure the DataFrame columns are in the correct order
    input_data = input_data[column_order]

    # Transform the input data using the loaded transformer
    input_data_transformed = transformer.transform(input_data)

    # Make a prediction
    prediction_proba = model.predict(input_data_transformed)
    prediction = np.argmax(prediction_proba, axis=1)
    confidence = np.max(prediction_proba) * 100


    # Decode the prediction to the original label
    predicted_risk_level = label_encoder.inverse_transform(prediction)[0]

    # Display the prediction
    st.subheader('Prediction')
    if predicted_risk_level == 'High':
        st.error(f'The predicted cancer risk level is: **{predicted_risk_level}**')
    elif predicted_risk_level == 'Medium':
        st.warning(f'The predicted cancer risk level is: **{predicted_risk_level}**')
    else:
        st.success(f'The predicted cancer risk level is: **{predicted_risk_level}**')
    
    st.write(f"Confidence: **{confidence:.2f}%**")