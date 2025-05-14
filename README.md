# 🩺 Medical Insurance Cost Prediction using Regression

## 🎯 Project Objective

Build a machine learning model that can **predict the insurance cost (charges)** for an individual based on their **demographics and lifestyle factors**.

This helps:
- Insurance companies better estimate risk
- Customers understand what affects their insurance costs

---

## 📄 Dataset Overview

- **Source**: [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
- **Columns**:
  - `age`: Age of the individual
  - `sex`: Gender (male/female)
  - `bmi`: Body Mass Index (float)
  - `children`: Number of children covered by insurance
  - `smoker`: Smoker status (yes/no)
  - `region`: Residential area in the US
  - `charges`: Target column — medical insurance cost (float)

---

## ⚙️ Problem Type

This is a **supervised regression** problem, where the goal is to predict a continuous numeric variable: `charges`.

---

## 🧱 Project Steps

### 1. ETL (Extract, Transform, Load)
- Extract data from CSV
- Clean and preprocess the dataset
- Save cleaned version for analysis

### 2. EDA (Exploratory Data Analysis)
- Check for missing values and outliers
- Explore distributions of key features
- Visualize relationships between features and `charges`

### 3. Feature Engineering
- Encode categorical variables (e.g., `smoker`, `region`)
- Scale or normalize features if needed
- Create new features if useful

### 4. Model Building
- Train different regression models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Evaluate with metrics like:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score

### 5. Model Evaluation
- Compare models
- Choose best-performing one
- Analyze feature importance

### 6. (Optional) Deployment
- Save trained model (e.g., using `joblib`)
- Build a CLI tool or basic web app for prediction

---

## 📈 Expected Output

- A trained regression model
- Visual reports showing:
  - Correlation between features and cost
  - Impact of smoking, age, BMI, etc.
- Insights like:
  - Smokers tend to pay significantly more
  - Higher BMI leads to higher insurance cost

---

## 🛠 Tools Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 📌 Author Notes

This is a beginner-friendly data science project to get hands-on experience with:
- ETL pipelines
- EDA & visualizations
- Machine learning model building
- End-to-end thinking for solving real business problems

