import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
import gdown

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Model download function
def download_model():
    model_path = 'churn_model.pkl'
    if not os.path.exists(model_path):
        # Google Drive link (replace with your actual Google Drive link)
        drive_link = "YOUR_GOOGLE_DRIVE_LINK"
        
        with st.spinner('Downloading model... This might take a few minutes...'):
            try:
                # For Google Drive links
                gdown.download(drive_link, model_path, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                st.error("Please download the model manually from the link below and place it in the same directory as this app.")
                st.markdown("""
                    ### Manual Download Instructions:
                    1. Click the link below to download the model:
                    2. [Download Churn Model (3.05 GB)](YOUR_GOOGLE_DRIVE_LINK)
                    3. Place the downloaded `churn_model.pkl` file in the same directory as this app
                    4. Refresh this page
                """)
                return False
    return True

# Load the model
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('churn_model.pkl'):
            st.warning("Model file not found. Please follow the download instructions below.")
            return None
        return joblib.load('churn_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Page title and description
    st.title("📱 Expresso Customer Churn Prediction")
    st.markdown("Enter customer information to predict their likelihood of churning.")
    
    # Check if model exists
    model = load_model()
    
    if not model:
        st.warning("⚠️ Model not found!")
        st.markdown("""
        ### First Time Setup Instructions
        
        This application requires a pre-trained model file (`churn_model.pkl`, 3.05 GB) to make predictions.
        
        **Option 1: Automatic Download**
        - Click the button below to download the model automatically
        - This might take a few minutes depending on your internet connection
        
        **Option 2: Manual Download**
        1. Download the model manually from [this link](YOUR_GOOGLE_DRIVE_LINK)
        2. Place the downloaded `churn_model.pkl` file in the same directory as this app
        3. Refresh this page
        
        Note: The model file is large (3.05 GB) because it contains complex patterns learned from extensive customer data.
        """)
        
        if st.button("Download Model"):
            download_model()
            st.rerun()
            
        return
    
    # Rest of your existing code for the form and predictions
    with st.form("prediction_form"):
        st.subheader("Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Numeric inputs
            st.markdown("### Usage Metrics")
            tenure = st.number_input("Tenure (months)", min_value=0)
            monthly_spend = st.number_input("Monthly Spend ($)", min_value=0.0)
            total_spend = st.number_input("Total Spend ($)", min_value=0.0)
            data_usage = st.number_input("Data Usage (GB)", min_value=0.0)
            call_minutes = st.number_input("Call Minutes", min_value=0)
            
        with col2:
            # Categorical inputs
            st.markdown("### Customer Details")
            plan_type = st.selectbox(
                "Plan Type",
                options=['Basic', 'Premium', 'Ultimate']
            )
            payment_method = st.selectbox(
                "Payment Method",
                options=['Credit Card', 'Bank Transfer', 'Cash']
            )
            contract_type = st.selectbox(
                "Contract Type",
                options=['Month-to-Month', 'One Year', 'Two Year']
            )
            customer_service_calls = st.number_input("Customer Service Calls", min_value=0)
            
        # Submit button
        submitted = st.form_submit_button("Predict Churn Probability")
        
        if submitted:
            try:
                # Prepare input data
                input_data = {
                    'tenure': tenure,
                    'monthly_spend': monthly_spend,
                    'total_spend': total_spend,
                    'data_usage': data_usage,
                    'call_minutes': call_minutes,
                    'plan_type': plan_type,
                    'payment_method': payment_method,
                    'contract_type': contract_type,
                    'customer_service_calls': customer_service_calls
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns([1,2,1])
                
                with col2:
                    if prediction == 1:
                        st.error("⚠️ High Risk of Churn")
                        st.metric(
                            label="Churn Probability",
                            value=f"{probability:.1%}"
                        )
                        st.markdown("""
                            #### Recommended Actions:
                            - Reach out to customer for feedback
                            - Offer personalized retention deals
                            - Review service quality and usage patterns
                        """)
                    else:
                        st.success("✅ Low Risk of Churn")
                        st.metric(
                            label="Churn Probability",
                            value=f"{probability:.1%}"
                        )
                        st.markdown("""
                            #### Recommended Actions:
                            - Continue monitoring usage patterns
                            - Consider upselling opportunities
                            - Maintain regular engagement
                        """)
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.info("Please ensure all input fields are filled correctly.")

if __name__ == "__main__":
    main()