import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import gradio as gr
from serving.inference import predict

def predict_ui(*args):
    payload = {
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
        "tenure": int(args[15]),
        "MonthlyCharges": float(args[16]),
        "TotalCharges": float(args[17]),
    }

    return predict(payload)

demo = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Dropdown(["Male", "Female"]),
        gr.Dropdown(["Yes", "No"]),
        gr.Dropdown(["Yes", "No"]),
        gr.Dropdown(["Yes", "No"]),
        gr.Dropdown(["Yes", "No", "No phone service"]),
        gr.Dropdown(["DSL", "Fiber optic", "No"]),
        gr.Dropdown(["Yes", "No", "No internet service"]),
        gr.Dropdown(["Yes", "No", "No internet service"]),
        gr.Dropdown(["Yes", "No", "No internet service"]),
        gr.Dropdown(["Yes", "No", "No internet service"]),
        gr.Dropdown(["Yes", "No", "No internet service"]),
        gr.Dropdown(["Yes", "No", "No internet service"]),
        gr.Dropdown(["Month-to-month", "One year", "Two year"]),
        gr.Dropdown(["Yes", "No"]),
        gr.Dropdown([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]),
        gr.Number(),
        gr.Number(),
        gr.Number(),
    ],
    outputs="text",
    title="Telco Churn Predictor"
)

demo.launch()