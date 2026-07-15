# Obesity Risk Classifier

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://obesity-risk-classifier.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Jahanzeb-abbasi/obesity-risk-classifier)

A machine learning web application that predicts an individual's obesity risk category based on lifestyle habits, physical characteristics, and health-related factors.

This project was built as part of my Applied AI Engineering portfolio to practice multi-class classification, model evaluation, and end-to-end deployment.

---

## Live Demo

**Streamlit App:**  
https://obesity-risk-classifier.streamlit.app/

---

## Dataset

**Kaggle Playground Series S4E2 – Multi-Class Prediction of Obesity Risk**

- **20,758** samples
- **16 input features**
- **7 target classes**

Target classes:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

---

## Features

The model uses a combination of:

### Personal Information
- Gender
- Age
- Height
- Weight

### Lifestyle Habits
- Family history of overweight
- High-calorie food consumption
- Vegetable consumption
- Number of main meals
- Eating between meals
- Water intake
- Physical activity
- Screen time
- Smoking
- Alcohol consumption
- Calories monitoring
- Transportation method

---

## Model

Two models were evaluated:

- Logistic Regression (Baseline)
- Random Forest (Final Model)

Random Forest achieved the best overall performance and was selected for deployment.

### Final Performance

| Metric | Score |
|---------|------:|
| Accuracy | **0.90** |
| Macro Precision | **0.89** |
| Macro Recall | **0.89** |
| Macro F1 Score | **0.89** |

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## Project Structure

```
.
├── app.py
├── artifacts/
│   └── feature_cols.joblib
├── notebooks/
│   └── 01_EDA.ipynb
│   └── 02_model_training.ipynb
├── requirements.txt
└── README.md
```

> The trained Random Forest model is hosted externally due to GitHub's file size limit and is automatically downloaded on the first application startup.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Jahanzeb-abbasi/obesity-risk-classifier.git
cd obesity-risk-classifier
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

Since the trained Random Forest model exceeds GitHub's 100 MB file limit, it is hosted separately on Hugging Face and automatically downloaded during the first launch. Subsequent runs use the cached local copy.

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Explainable AI using SHAP
- Additional ensemble models
- Docker deployment

---

## Author

**Jahanzeb Abbasi**

GitHub: https://github.com/Jahanzeb-abbasi
