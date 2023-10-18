import streamlit as st

# Create a dictionary to map model names to their corresponding pickle files
model_files = {
    "Ensemble": "./models/Ensemble.pkl",
    "KNNClassifier": "./models/KNNClassifier.pkl",
    "NBClassifier": "./models/NBClassifier.pkl",
    "RandomForest": "./models/RandomForest.pkl",
    "SVMClassifier": "./models/SVMClassifier.pkl"
}

# Sidebar with model selection
st.sidebar.title("Select Model")
selected_model = st.sidebar.selectbox("Choose a model for prediction", list(model_files.keys()))

# Load the selected model
model_file = model_files[selected_model]
model = joblib.load(model_file)

# Main content
st.title("Crop Prediction Using Machine Learning")

st.write("This application allows you to predict the most suitable crop based on input parameters using different models.")

# Input parameters
st.header("Input Parameters")
param1 = st.number_input("Nitrogen", value=0.0)
param2 = st.number_input("Phosphorus", value=0.0)
param3 = st.number_input("Potassium", value=0.0)
param4 = st.number_input("Temperature", value=0.0)
param5 = st.number_input("Humidity", value=0.0)
param6 = st.number_input("PH", value=0.0)
param7 = st.number_input("Rainfall(in mm)", value=0.0)

# Create a list of input parameters for prediction
input_data = np.array([[param1, param2, param3, param4, param5, param6, param7]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.subheader("Prediction")
    st.write(f"The predicted crop using '{selected_model}' is: {prediction[0]}")

# Add a disclaimer
st.sidebar.subheader("About CPA")
st.sidebar.write('''Welcome to the Crop Prediction App! This tool leverages machine learning models to help you predict the most suitable crop based on seven essential agricultural parameters. Whether you're a farmer planning your next planting season or an enthusiast interested in crop predictions, this app has you covered. Simply input the parameters, select a model, and get accurate crop recommendations for your specific conditions. Explore the power of data-driven agriculture with ease!.''')

