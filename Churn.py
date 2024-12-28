import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        # Ensure model is loaded from a valid path
        model_path = 'model/churn_model.pkl'
        if not os.path.exists(model_path):
            st.warning("Model file not found. Please follow the download instructions.")
            return None
        model = joblib.load(model_path)
        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            return model
        else:
            st.error("Invalid model format")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Debug: Error details:", str(e))
        return None

def main():
    st.title("📱 Expresso Customer Churn Prediction")
    st.markdown("Enter customer information to predict their likelihood of churning.")
    
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the model file and try again.")
        return
    
    # Load the pre-trained LabelEncoders for categorical features
    le_region = LabelEncoder()
    le_tenure = LabelEncoder()
    le_mrg = LabelEncoder()
    le_top_pack = LabelEncoder()

    # Load pre-fitted encoders or fit them based on training data
    # For example, if you have access to training data, fit the label encoders here

    with st.form("prediction_form"):
        st.subheader("Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            region = st.selectbox("REGION", options=['FATICK', 'DAKAR', 'THIES', 'SAINT-LOUIS', 'LOUGA', 'DIOURBEL', 'KAOLACK', 'ZIGUINCHOR', 'KOLDA', 'TAMBACOUNDA'])
            tenure = st.selectbox("TENURE", options=['K > 24 month', '0 < K <= 6 month', '6 < K <= 12 month', '12 < K <= 24 month'])
            montant = st.number_input("MONTANT", min_value=0.0, value=4250.0)
            frequence_rech = st.number_input("FREQUENCE_RECH", min_value=0, value=15)
           
        
        with col2:
            frequence = st.number_input("FREQUENCE", min_value=0, value=17)
            on_net = st.number_input("ON_NET", min_value=0.0, value=388.0)
            revenue = st.number_input("REVENUE", min_value=0.0, value=4251.0)

        with col3:
            mrg = st.selectbox("MRG", options=['NO', 'YES'])
            regularity = st.number_input("REGULARITY", min_value=0, value=54)
            arpu_segment = st.number_input("ARPU_SEGMENT", min_value=0.0, value=1417.0)
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            try:
                # Prepare input data for prediction
                input_data = {
                    'REGION': region,
                    'TENURE': tenure,
                    'MONTANT': montant,
                    'FREQUENCE_RECH': frequence_rech,
                    'REVENUE': revenue,
                    'ARPU_SEGMENT': arpu_segment,
                    'FREQUENCE': frequence,
                    'ON_NET': on_net,
                    'MRG': mrg,
                    'REGULARITY': regularity
                
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Apply label encoding to categorical features
                input_df['REGION'] = le_region.fit_transform([region])
                input_df['TENURE'] = le_tenure.fit_transform([tenure])
                input_df['MRG'] = le_mrg.fit_transform([mrg])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                # Display results
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
                st.write("Debug: Error details:", str(e))

if __name__ == "__main__":
    main()
