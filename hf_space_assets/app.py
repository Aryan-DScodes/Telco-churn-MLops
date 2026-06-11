import gradio as gr
import joblib
import pandas as pd

# Load artifacts
model = joblib.load("model.pkl")
preprocess = joblib.load("preprocessing.pkl")
with open("feature_columns.txt", "r") as f:
    feature_columns = [line.strip() for line in f.readlines()]
def predict_ui(*args):
    input_data = {
        "gender": args[0],
        "Partner": args[1],
        "Dependents": args[2],
        "PhoneService": args[3],
        "MultipleLines": args[4],
        "InternetService": args[5],
        "OnlineSecurity": args[6],
        "OnlineBackup": args[7],
        "DeviceProtection": args[8],
        "TechSupport": args[9],
        "StreamingTV": args[10],
        "StreamingMovies": args[11],
        "Contract": args[12],
        "PaperlessBilling": args[13],
        "PaymentMethod": args[14],
        "tenure": float(args[15]),
        "MonthlyCharges": float(args[16]),
        "TotalCharges": float(args[17]),
    }

    df = pd.DataFrame([input_data])

    # IMPORTANT: apply preprocessing (if your pipeline uses it)
    if preprocess:
        df = preprocess.transform(df)

    pred = model.predict(df)[0]

    return "Churn" if pred == 1 else "No Churn"


demo = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method"
        ),
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges"),
    ],
    outputs="text",
    title="Telco Churn Predictor"
)

demo.launch()