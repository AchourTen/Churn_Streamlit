import streamlit as st
import pandas as pd
import pickle
import os
from urllib.request import urlretrieve

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Function to download model
def download_model():
    # Replace this with your actual drive link
    MODEL_URL = "YOUR_DIRECT_DOWNLOAD_LINK"
    
    try:
        if not os.path.exists('models'):
            os.makedirs('models')
        
        if not os.path.exists('models/churn_model.pkl'):
            with st.spinner('Downloading model... This might take a few minutes...'):
                urlretrieve(MODEL_URL, 'models/churn_model.pkl')
                st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        st.error("Please download the model manually and place it in a 'models' folder")
        return False
    return True

# Load the model
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('models/churn_model.pkl'):
            st.warning("Model file not found. Please follow the download instructions.")
            return None
        with open('models/churn_model.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("📱 Expresso Customer Churn Prediction")
    st.markdown("Enter customer information to predict their likelihood of churning.")
    
    # Check for model and show download instructions if needed
    if not os.path.exists('models/churn_model.pkl'):
        st.warning("⚠️ Model not found!")
        st.markdown("""
        ### Setup Instructions
        
        1. Download the model file from [this link](YOUR_DRIVE_LINK)
        2. Create a folder named 'models' in the same directory as this app
        3. Place the downloaded model file in the 'models' folder
        4. Rename the file to 'churn_model.pkl'
        5. Refresh this page
        """)
        return
    
    model = load_model()
    
    if model:
        with st.form("prediction_form"):
            st.subheader("Customer Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tenure = st.number_input("Tenure (months)", min_value=0)
                monthly_spend = st.number_input("Monthly Spend ($)", min_value=0.0)
                data_usage = st.number_input("Data Usage (GB)", min_value=0.0)
            
            with col2:
                plan_type = st.selectbox(
                    "Plan Type",
                    options=['Basic', 'Premium', 'Ultimate']
                )
                contract_type = st.selectbox(
                    "Contract Type",
                    options=['Month-to-Month', 'One Year', 'Two Year']
                )
            
            submitted = st.form_submit_button("Predict Churn")
            
            if submitted:
                try:
                    input_data = {
                        'tenure': tenure,
                        'monthly_spend': monthly_spend,
                        'data_usage': data_usage,
                        'plan_type': plan_type,
                        'contract_type': contract_type
                    }
                    
                    input_df = pd.DataFrame([input_data])
                    
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0][1]
                    
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    if prediction == 1:
                        st.error("⚠️ High Risk of Churn")
                        st.metric("Churn Probability", f"{probability:.1%}")
                    else:
                        st.success("✅ Low Risk of Churn")
                        st.metric("Churn Probability", f"{probability:.1%}")
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("Please ensure all inputs are filled correctly.")

if __name__ == "__main__":
    main()
