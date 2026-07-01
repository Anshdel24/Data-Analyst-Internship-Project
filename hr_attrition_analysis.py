import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(records=1200):
    print("[1/4] Synthesizing HR dataset...")
    departments = ['Sales', 'Engineering', 'HR', 'Marketing', 'Finance', 'Operations']
    salary_bands = ['Low', 'Medium', 'High']
    
    data = {
        'Employee_ID': [f'EMP{1000 + i}' for i in range(records)],
        'Department': [np.random.choice(departments) for _ in range(records)],
        'Salary_Band': [np.random.choice(salary_bands, p=[0.4, 0.45, 0.15]) for _ in range(records)],
        'Satisfaction_Level': np.random.uniform(0.1, 1.0, records),
        'Average_Monthly_Hours': np.random.randint(120, 310, records),
        'Promotion_Last_5Years': np.random.choice([0, 1], size=records, p=[0.94, 0.06]),
        'Years_At_Company': np.random.randint(1, 11, records)
    }
    df = pd.DataFrame(data)
    
    # Attrition logic definition
    attrition_prob = (
        (df['Satisfaction_Level'] < 0.4) * 0.45 +
        (df['Average_Monthly_Hours'] > 260) * 0.25 +
        (df['Promotion_Last_5Years'] == 0) * 0.10 +
        (df['Salary_Band'] == 'Low') * 0.15
    )
    df['Attrition'] = np.where(attrition_prob + np.random.uniform(0, 0.2, records) > 0.5, 1, 0)
    df.to_csv('hr_attrition_data.csv', index=False)
    print("-> Saved 'hr_attrition_data.csv'")
    return df

def run_eda(df):
    print("[2/4] Performing Exploratory Data Analysis & generating charts...")
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Department vs Attrition
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Department', hue='Attrition', palette='Set2')
    plt.title('Employee Attrition Distribution across Departments')
    plt.tight_layout()
    plt.savefig('department_attrition.png')
    plt.close()
    
    # Chart 2: Salary vs Attrition
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Salary_Band', hue='Attrition', palette='viridis')
    plt.title('Employee Attrition Analysis by Salary Band')
    plt.tight_layout()
    plt.savefig('salary_attrition.png')
    plt.close()
    print("-> Saved 'department_attrition.png' and 'salary_attrition.png'")

def train_predictive_model(df):
    print("[3/4] Preprocessing features and splitting data...")
    le_dept = LabelEncoder()
    le_salary = LabelEncoder()
    
    df_model = df.copy()
    df_model['Department'] = le_dept.fit_transform(df['Department'])
    df_model['Salary_Band'] = le_salary.fit_transform(df['Salary_Band'])
    
    X = df_model.drop(columns=['Employee_ID', 'Attrition'])
    y = df_model['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print("[4/4] Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n================ MODEL PERFORMANCE REPORT ================")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("==========================================================")

if __name__ == "__main__":
    hr_df = generate_data()
    run_eda(hr_df)
    train_predictive_model(hr_df)
