# ExcelR-Internship-Project-2

# Predictive Modeling for Attorney Involvement in Claims

## Introduction
This project focuses on building predictive models to determine whether an attorney will be involved in an insurance claim based on various claim-related factors. The goal is to optimize the claims process, reduce litigation costs, and enhance resource allocation for insurance companies.

---

## Project Objectives
1. Understand and preprocess the dataset for modeling.
2. Perform exploratory data analysis (EDA) to gain insights into key predictors.
3. Engineer features to improve model performance.
4. Train and evaluate machine learning models to predict attorney involvement.
5. Select the best model and deploy it using Streamlit for real-time predictions.

---

## Dataset Information
The dataset contains 13 features and 1,340 rows:
- **Numerical Features:** Financial Loss (`LOSS`), Age (`CLMAGE`), etc.
- **Categorical Features:** Gender (`CLMSEX`), Driving Record (`Driving_Record`), etc.
- **Target Variable:** `ATTORNEY` (1 = Attorney Involved, 0 = No Attorney).

---

## Exploratory Data Analysis (EDA)
1. **Correlation Analysis:** Identified relationships between predictors and the target variable.
2. **Visualizations:** Distribution plots, bar charts, and box plots were used to highlight key insights.
3. **Findings:** Factors such as accident severity and driving records have significant impact on attorney involvement.

---

## Data Preprocessing & Visualizations
- **Steps Taken:**
  - Imputed missing values.
  - Scaled numerical features.
  - Encoded categorical variables using one-hot encoding.
  - Balanced the dataset to handle class imbalances.

---

## Train & Test Split
- Data was split into training (80%) and testing (20%) sets to ensure unbiased evaluation.
- The split ensured stratified sampling for balanced representation of target classes.

---

## Feature Engineering
- Created new features to capture relationships between existing variables.
- Encoded categorical data to numerical forms for compatibility with machine learning models.

---

## Applying Machine Learning Algorithms
The following models were trained and evaluated:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Machines (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes

---

## Model Evaluation
- Metrics such as Accuracy, Precision, Recall, F1-Score, and AUC-ROC were used.
- Logistic Regression emerged as the best model with an accuracy of ~55%.
- Visualizations included confusion matrices and bar plots for metric comparisons.

---

## Best Model Selection
- **Logistic Regression:**
  - Accuracy: 55%.
  - Precision: 56%.
  - Recall: 51%.
  - F1-Score: 53%.

---

## Output
The model predicts whether an attorney is likely to be involved in a claim:
- `1` = Attorney Involved
- `0` = No Attorney

---

## Deployment
- **Framework:** Streamlit.
- **Process:**
  1. Trained models (`Logistic Regression` and `Random Forest`) were saved as pickle files.
  2. A Streamlit app was created to take user inputs and provide predictions in real-time.
  3. Ngrok was used to expose the app to external users during testing.

---

## Conclusion
This project demonstrates how machine learning can transform insurance workflows by predicting attorney involvement in claims. Future work could include:
- Collecting more data for improved accuracy.
- Exploring advanced models and deep learning techniques.
- Integrating the model into live insurance systems for end-to-end automation.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/attorney-involvement-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Interact with the app through the generated URL.

---

## Files in the Repository
- `app.py`: Streamlit app for deployment.
- `best_lr.pkl`: Pickled Logistic Regression model.
- `best_rf.pkl`: Pickled Random Forest model.
- `Updated_Claimants_Dataset.csv`: Dataset used for the project.
- `README.md`: Project documentation.

---

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
- **Deployment Tool:** Ngrok

