🎓 Student Dropout Prediction System (Machine Learning Project)
📌 Project Overview

Student Dropout Prediction is a machine learning–based project designed to identify students who are at risk of dropping out from educational institutions. Early prediction helps institutions take timely academic and psychological actions to reduce dropout rates and improve student success.

This project applies supervised and unsupervised machine learning techniques on real-world student data to predict student outcomes effectively.

🎯 Objectives

Analyze student academic, demographic, and financial data

Identify key factors contributing to student dropout

Build and compare multiple machine learning models

Predict student status as:

Dropout

Enrolled

Graduate

🧠 Machine Learning Models Used
🔹 Supervised Learning

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naïve Bayes

Support Vector Machine (SVM)

🔹 Unsupervised Learning

K-Means Clustering

📂 Dataset Information

Dataset Name: Student Dropout Dataset

Source: UCI Machine Learning Repository / Kaggle

Type: Multiclass Classification

Dataset Size: 4,000+ records

🎯 Target Variable Mapping

0 → Dropout

1 → Enrolled

2 → Graduate

🔑 Important Features

Age at enrollment

Previous qualification and grades

Curricular units (1st & 2nd semester performance)

Tuition fees payment status

Scholarship holder

Debtor status

⚙️ Data Preprocessing

Handling missing values

Label encoding for categorical features

Feature selection

Feature scaling using StandardScaler

Train–test split

📊 Exploratory Data Analysis (EDA)

Correlation heatmap

Feature distribution analysis

Class imbalance visualization

Relationship analysis between features and target

📈 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

🏆 Best Model Identification

After training and evaluating all models, Decision Tree and SVM achieved the highest accuracy (around 90%+) and performed best in predicting student dropout risk.

🚀 Expected Outcomes

Early identification of students at risk

Improved academic decision-making

Support for student retention strategies

Better understanding of student performance patterns

🛠️ Tools & Technologies

Programming Language: Python

Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

Development Environment: Jupyter Notebook

Version Control: Git & GitHub
```
📁 Project Structure
Student-Dropout-Prediction/
│
├── dataset/
│   └── student_dropout.csv
│
├── notebooks/
│   └── student_dropout_prediction.ipynb
│
├── README.md
└── requirements.txt
```
▶️ How to Run the Project

Clone the repository

git clone https://github.com/your-username/student-dropout-prediction.git


Install dependencies

pip install -r requirements.txt


Open Jupyter Notebook

jupyter notebook


Run student_dropout_prediction.ipynb

📌 Conclusion

This project demonstrates how machine learning can be applied in the education domain to predict student dropout risks. Accurate predictions enable institutions to take preventive actions and improve overall student success rates.

🔮 Future Scope

Deploy the model as a web application

Use ensemble and deep learning models

Integrate real-time student data

Add explainable AI (XAI) techniques

# 📜 License
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project for educational and research purposes.


👨‍💻 Author

Project By: Shivam Kumar Mishra
Course: B.Tech (CSE)
Project Type: Machine Learning Academic Project
