import gradio as gr
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load the trained model
try:
    pipeline = joblib.load('model.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    print("Model and features loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None
    feature_cols = []

def predict_stock(date_str, stock_1, stock_2, stock_3, stock_4):
    if pipeline is None:
        return "Model not loaded."
    
    try:
        # Process Date
        try:
            date_obj = pd.to_datetime(date_str)
        except:
            return "Invalid Date Format. Please use YYYY-MM-DD."
            
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        day_of_week = date_obj.dayofweek
        
        # Create Dataframe
        data = {
            'Stock_1': [stock_1],
            'Stock_2': [stock_2],
            'Stock_3': [stock_3],
            'Stock_4': [stock_4],
            'Year': [year],
            'Month': [month],
            'Day': [day],
            'DayOfWeek': [day_of_week]
        }
        
        df = pd.DataFrame(data)
        
        # Ensure columns are in correct order
        # We need to make sure we match the feature columns from training
        # If Validation set was used, checking feature_cols is good.
        if len(feature_cols) > 0:
            df = df[feature_cols]
            
        prediction = pipeline.predict(df)[0]
        return f"{prediction:.4f}"
        
    except Exception as e:
        return f"Error during prediction: {e}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 Stock Price Prediction App")
    gr.Markdown("Enter the date and values for Stock 1-4 to predict the value of **Stock 5**.")
    
    with gr.Row():
        with gr.Column():
            date_input = gr.Textbox(label="Date (YYYY-MM-DD)", value=datetime.today().strftime('%Y-%m-%d'))
            s1 = gr.Number(label="Stock 1 Value", value=6.0)
            s2 = gr.Number(label="Stock 2 Value", value=70.0)
            s3 = gr.Number(label="Stock 3 Value", value=80.0)
            s4 = gr.Number(label="Stock 4 Value", value=100.0)
            predict_btn = gr.Button("Predict Stock 5", variant="primary")
            
        with gr.Column():
            output = gr.Textbox(label="Predicted Stock 5 Value")

    predict_btn.click(
        fn=predict_stock,
        inputs=[date_input, s1, s2, s3, s4],
        outputs=output
    )

    gr.Markdown("### Application Details")
    gr.Markdown("- **Model**: Random Forest Regressor")
    gr.Markdown("- **Input Features**: Stock 1, Stock 2, Stock 3, Stock 4, Date Analysis")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=True)

    
