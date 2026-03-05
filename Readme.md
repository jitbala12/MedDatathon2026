# Link
https://meddatathon2026.streamlit.app/

# Methods

We implemented the models using **Python and Scikit-learn** with a pipeline architecture.

## Libraries Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

## Workflow

1. Load dataset
2. Data preprocessing
3. Train/Test split
4. Feature scaling
5. Model training using Scikit-learn pipelines
6. Model evaluation
7. Deployment using Streamlit

## Models Implemented

We evaluated three classification models based on literature recommendations:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

Each model was trained and evaluated using consistent preprocessing pipelines.

# Evaluation Metrics

To compare model performance we used:

- **Accuracy**
- **Precision**
- **Recall**
- **ROC-AUC**

In a clinical context, **recall for malignant tumors is especially important** because false negatives (missing cancer) can have serious consequences.

---

# Results

All models performed strongly on this dataset, but their behavior differed.

### Logistic Regression
- Very strong overall accuracy
- Well calibrated probabilities

### Random Forest
- High accuracy and strong stability
- Effective at capturing nonlinear relationships

### Support Vector Machine
- Good classification performance
- More conservative probability estimates

Overall, **Random Forest and Logistic Regression demonstrated the strongest diagnostic performance**, which aligns with findings reported in the research literature.

---

# Streamlit Application

We built an interactive **Streamlit dashboard** that allows users to test the models.

Users can:

- Input tumor measurement values
- Select a machine learning model
- Generate a prediction
- Compare model outputs

This allows users to explore how different algorithms respond to the same diagnostic data.

---
## Dataset

UCI Machine Learning Repository  
Breast Cancer Wisconsin (Diagnostic)

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
