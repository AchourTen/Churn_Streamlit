# Customer Churn Prediction - Expresso Dataset

## Overview
This project focuses on predicting customer churn using a dataset from Expresso. It includes data preprocessing, feature engineering, and model training using a Random Forest Classifier. A Streamlit web application is provided for interactive predictions.

## Dataset
- **Source:** Expresso Churn Dataset
https://drive.google.com/file/d/12_KUHr5NlHO_6bN5SylpkxWc-JvpJNWe/view

- **Key Features:**
  - Customer demographic and usage data.
  - Target variable: `CHURN` (1 - churned, 0 - not churned).
| French                 | English                                      |
|------------------------|----------------------------------------------|
| **user_id**            | User ID                                      |
| **REGION**             | The location of each client                  |
| **TENURE**             | Duration in the network                      |
| **MONTANT**            | Top-up amount                                |
| **FREQUENCE_RECH**     | Number of times the customer refilled        |
| **REVENUE**            | Monthly income of each client                |
| **ARPU_SEGMENT**       | Income over 90 days / 3                      |
| **FREQUENCE**          | Number of times the client has made an income|
| **DATA_VOLUME**        | Number of connections                        |
| **ON_NET**             | Inter-Expresso call                          |
| **ORANGE**             | Call to Orange                               |
| **TIGO**               | Call to Tigo                                 |
| **ZONE1**              | Call to zone1                                |
| **ZONE2**              | Call to zone2                                |
| **MRG**                | A client who is going                        |
| **REGULARITY**         | Number of times the client is active for 90 days |
| **TOP_PACK**           | The most active packs                        |
| **FREQ_TOP_PACK**      | Number of times the client has activated top pack packages |
| **CHURN**              | Target variable to predict (1 - churned, 0 - not churned) |
## Project Structure
```
│  
├── app.py               # Streamlit web application
├── Model_training.py  # Data preprocessing and model training script
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── model/               # NEED TO CREATE THE Directory for storing trained models
    └── churn_model.pkl  # Saved Random Forest model
```

## Setup Instructions
1. **Clone Repository:**
   ```bash
   git clone https://github.com/AchourTen/Churn_Streamlit
   cd https://github.com/AchourTen/Churn_Streamlit
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Preprocessing and Model Training:**
   ```bash
   python preprocess_model.py
   ```

4. **Launch Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

## Preprocessing and Model Training
- **Key Steps:**
  1. Load and inspect the dataset.
  2. Handle missing values and outliers.
  3. Encode categorical features using LabelEncoder.
  4. Train a Random Forest Classifier.
  5. Evaluate the model using accuracy metrics.
  6. Save the model using `joblib` in the `model/` directory as `churn_model.pkl`.

- **Model Accuracy:** Displayed in the console during training.

## Streamlit Web App
The web app provides a user-friendly interface to predict churn probabilities based on customer inputs.

### Features:
- Select and input customer details.
- Predict churn and display results with probabilities.

### Instructions:
1. Open the app using Streamlit (`streamlit run app.py`).
2. Fill in the form with customer details.
3. Click "Predict Churn" to see the results.

### Example Inputs:
- REGION: 'DAKAR'
- TENURE: '0 < K <= 6 month'
- MONTANT: 4250.0
- FREQUENCE_RECH: 15
- MRG: 'YES'

### Example Output:
- Prediction: High/Low Risk of Churn
- Probability: 80%

## Troubleshooting
- **Model Not Found:** Ensure the `churn_model.pkl` file exists in the `model/` directory.
- **Prediction Errors:** Check input formats and categorical values.

## Future Improvements
- Add additional models for comparison.
- Incorporate feature selection techniques.
- Enhance the web app with visualization tools.

## License
This project is licensed under the MIT License.

---

