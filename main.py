import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. MLflow ნაწილი - მოდელის მომზადება
# ==========================================

# ექსპერიმენტისთვის სახელის დარქმევა
mlflow.set_experiment("Customer_Churn_Baseline")

# ხელოვნური, პატარა დატასეტი (EDA-დან გამომდინარე)
# features: tenure (თვეები), MonthlyCharges (გადასახადი)
# target: churn (0 = არა, 1 = კი)
data = {
    "tenure": [1, 3, 12, 24, 2, 36, 48, 5, 60, 1],
    "MonthlyCharges": [85, 75, 40, 60, 90, 50, 30, 95, 100, 20],
    "churn": [1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
}
df = pd.DataFrame(data)

X = df[["tenure", "MonthlyCharges"]]
y = df["churn"]

# MLflow-ში რბოლის (Run) დაწყება
with mlflow.start_run():
    # მოდელის შექმნა და გაწვრთნა
    model = LogisticRegression()
    model.fit(X, y)

    # პროგნოზის გაკეთება მეტრიკისთვის
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # დავალების მოთხოვნა: მინიმუმ ერთი პარამეტრის შენახვა
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("threshold_tenure", 6)

    # დავალების მოთხოვნა: მინიმუმ ერთი მეტრიკის შენახვა
    mlflow.log_metric("accuracy", accuracy)

    # დავალების მოთხოვნა: Model Logging (მოდელის შენახვა)
    mlflow.sklearn.log_model(model, "churn_baseline_model")

    print(f"მოდელი წარმატებით ჩაიწერა! Accuracy: {accuracy}")


# ==========================================
# 2. FastAPI ნაწილი - სერვისის აწყობა
# ==========================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI სერვისი კლიენტების გადინების პროგნოზირებისთვის",
)


# Pydantic მოდელი შემავალი მონაცემების ვალიდაციისთვის
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float


@app.post("/predict")
def predict_churn(customer: CustomerData):
    # შემავალი მონაცემების DataFrame-ად გადაქცევა
    input_data = pd.DataFrame(
        [[customer.tenure, customer.MonthlyCharges]],
        columns=["tenure", "MonthlyCharges"],
    )

    # პროგნოზის გაკეთება ჩვენი მოდელით
    prediction = model.predict(input_data)[0]
    # ალბათობის გამოთვლა
    probability = model.predict_proba(input_data)[0][1]

    # პასუხის დაბრუნება JSON ფორმატში
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 2),
        "status": "High Risk" if prediction == 1 else "Low Risk",
    }