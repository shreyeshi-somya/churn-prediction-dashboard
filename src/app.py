import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

import anthropic
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title and Introduction
st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("### Predict customer churn using machine learning and get AI-powered insights")

# Overview section
with st.expander("ℹ️ About This Dashboard", expanded=False):
    st.markdown("""
    This dashboard predicts customer churn for telecom companies using machine learning.
    
    **Features:**
    - 📊 **4 ML Models**: Compare Logistic Regression, Random Forest, and XGBoost
    - 🎯 **Batch Predictions**: Upload customer data and get churn predictions
    - 📈 **Performance Metrics**: View confusion matrices, ROC curves, and key metrics
    - 🤖 **AI Insights**: Get business recommendations powered by Claude AI
    
    **How to use:**
    1. Select a model from the sidebar
    2. Upload your customer data CSV in the "Make Predictions" tab
    3. Review performance metrics and AI-generated insights
    
    **Models Available:**
    - **Logistic Regression (Tuned)**: Best balanced approach (70% recall, 0.842 AUC)
    - **Logistic Regression (Default)**: Conservative, high precision (66% precision)
    - **Random Forest**: Stable predictions (0.843 AUC)
    - **XGBoost (SMOTE)**: Catches most churners (71% recall)
    """)

# Sidebar
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression (Tuned Threshold)",
        "Logistic Regression (Default)",
        "Random Forest",
        "XGBoost (SMOTE)"
    ]
)

# Model performance info
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Expected Performance")

model_info = {
    "Logistic Regression (Tuned Threshold)": {
        "ROC-AUC": "0.842",
        "Recall": "70%",
        "Precision": "54%",
        "Best for": "Balanced approach"
    },
    "Logistic Regression (Default)": {
        "ROC-AUC": "0.842",
        "Recall": "56%",
        "Precision": "66%",
        "Best for": "Conservative predictions"
    },
    "Random Forest": {
        "ROC-AUC": "0.843",
        "Recall": "52%",
        "Precision": "66%",
        "Best for": "High precision"
    },
    "XGBoost (SMOTE)": {
        "ROC-AUC": "0.824",
        "Recall": "71%",
        "Precision": "52%",
        "Best for": "Catching more churners"
    }
}

info = model_info[model_choice]
st.sidebar.metric("ROC-AUC", info["ROC-AUC"])
st.sidebar.metric("Recall (Churn)", info["Recall"])
st.sidebar.metric("Precision (Churn)", info["Precision"])
st.sidebar.info(f"💡 **{info['Best for']}**")

st.sidebar.markdown("---")
st.sidebar.success("✅ Models loaded")

# Load models
@st.cache_resource
def load_models():
    lr_config = joblib.load('models/best_model_config.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    lr_default = joblib.load('models/logistic_regression_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    
    return lr_config, xgb_model, scaler, feature_names, lr_default, rf_model

lr_config, xgb_model, scaler, feature_names, lr_default, rf_model = load_models()

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Make Predictions", "📊 Model Performance", "🤖 AI Insights"])

with tab1:
    st.header("Upload Customer Data for Predictions")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load the data
        df_predict = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(df_predict)} customers")
        
        # Show preview
        with st.expander("📋 Preview Data"):
            st.dataframe(df_predict.head())
        
        # Predict button
        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Making predictions..."):
                # Prepare the data (same preprocessing as training)
                df_processed = df_predict.copy()
                
                # Drop customerID if it exists
                customer_ids = df_processed['customerID'] if 'customerID' in df_processed.columns else None
                if 'customerID' in df_processed.columns:
                    df_processed = df_processed.drop('customerID', axis=1)
                
                # Drop Churn column if it exists (for evaluation later)
                actual_churn = None
                if 'Churn' in df_processed.columns:
                    actual_churn = df_processed['Churn'].map({'Yes': 1, 'No': 0})
                    df_processed = df_processed.drop('Churn', axis=1)
                
                # One-hot encode
                df_encoded = pd.get_dummies(df_processed, drop_first=True)
                
                # Align columns with training data
                for col in feature_names:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                df_encoded = df_encoded[feature_names]
                
                # Scale numerical features
                numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
                df_encoded[numerical_features] = scaler.transform(df_encoded[numerical_features])
                
                # Make predictions based on selected model
                if model_choice == "Logistic Regression (Tuned Threshold)":
                    model = lr_config['model']
                    threshold = lr_config['optimal_threshold']
                    predictions_proba = model.predict_proba(df_encoded)[:, 1]
                    predictions = (predictions_proba >= threshold).astype(int)
                elif model_choice == "Logistic Regression (Default)":
                    predictions_proba = lr_default.predict_proba(df_encoded)[:, 1]
                    predictions = lr_default.predict(df_encoded)
                elif model_choice == "Random Forest":
                    predictions_proba = rf_model.predict_proba(df_encoded)[:, 1]
                    predictions = rf_model.predict(df_encoded)
                elif model_choice == "XGBoost (SMOTE)":
                    predictions_proba = xgb_model.predict_proba(df_encoded)[:, 1]
                    predictions = xgb_model.predict(df_encoded)
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Customer_ID': customer_ids if customer_ids is not None else range(len(predictions)),
                    'Churn_Probability': predictions_proba,
                    'Churn_Prediction': ['Yes' if p == 1 else 'No' for p in predictions],
                    'Risk_Level': ['High' if p >= 0.7 else 'Medium' if p >= 0.4 else 'Low' for p in predictions_proba]
                })
                
                if actual_churn is not None:
                    results_df['Actual_Churn'] = ['Yes' if a == 1 else 'No' for a in actual_churn]

                st.session_state.results_df = results_df    
                
                # Display results
                st.success("✅ Predictions complete!")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Customers", len(predictions))
                with col2:
                    churn_count = predictions.sum()
                    st.metric("Predicted Churners", f"{churn_count} ({churn_count/len(predictions)*100:.1f}%)")
                with col3:
                    high_risk = (results_df['Risk_Level'] == 'High').sum()
                    st.metric("High Risk Customers", high_risk)
                
                # Show results table
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

with tab2:
    st.header("Model Performance Metrics")
    
    # Check if we have actual churn data from the predictions
    if st.session_state.results_df is not None and 'Actual_Churn' in st.session_state.results_df.columns:
        st.success("✅ Evaluation data available!")
        
        # Get predictions and actuals
        y_true = (st.session_state.results_df['Actual_Churn'] == 'Yes').astype(int)
        y_pred = (st.session_state.results_df['Churn_Prediction'] == 'Yes').astype(int)
        y_proba = st.session_state.results_df['Churn_Probability']
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
        with col2:
            st.metric("Precision", f"{precision_score(y_true, y_pred):.2%}")
        with col3:
            st.metric("Recall", f"{recall_score(y_true, y_pred):.2%}")
        with col4:
            st.metric("F1-Score", f"{f1_score(y_true, y_pred):.2%}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Churn', 'Predicted Churn'],
            y=['Actual No Churn', 'Actual Churn'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        st.subheader("ROC Curve")
        from sklearn.metrics import roc_curve, auc
        
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        # Add shaded area under ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, 
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='blue', width=2),
            name=f'ROC Curve (AUC = {roc_auc:.3f})'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], 
            name='Random Classifier', 
            line=dict(color='red', dash='dash')
        ))
        
        # Add AUC text annotation in the center
        fig.add_annotation(
            x=0.6, y=0.3,
            text=f"<b>AUC = {roc_auc:.3f}</b>",
            showarrow=False,
            font=dict(size=20, color='blue'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='blue',
            borderwidth=2,
            borderpad=3
        )
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Upload data with actual churn labels to see performance metrics")

with tab3:
    st.header("🤖 AI-Powered Insights")
    
    # Check if we have predictions
    if st.session_state.results_df is None:
        st.info("👆 Upload data and generate predictions in the 'Make Predictions' tab first")
    else:
        st.write("Generate AI-powered insights about your churn predictions")
        
        if st.button("✨ Generate AI Insights", type="primary"):
            with st.spinner("Analyzing predictions with AI..."):
                try:
                    # Prepare summary statistics
                    total_customers = len(st.session_state.results_df)
                    predicted_churners = (st.session_state.results_df['Churn_Prediction'] == 'Yes').sum()
                    churn_rate = predicted_churners / total_customers * 100
                    high_risk = (st.session_state.results_df['Risk_Level'] == 'High').sum()
                    medium_risk = (st.session_state.results_df['Risk_Level'] == 'Medium').sum()
                    low_risk = (st.session_state.results_df['Risk_Level'] == 'Low').sum()
                    avg_churn_prob = st.session_state.results_df['Churn_Probability'].mean()
                    
                    # Create prompt for Claude
                    prompt = f"""You are a business analyst reviewing customer churn predictions. 

Here's the summary:
- Total customers analyzed: {total_customers}
- Predicted to churn: {predicted_churners} ({churn_rate:.1f}%)
- High risk customers: {high_risk}
- Medium risk customers: {medium_risk}
- Low risk customers: {low_risk}
- Average churn probability: {avg_churn_prob:.2%}
- Model used: {model_choice}

Please provide:
1. A brief executive summary (2-3 sentences)
2. Key insights about the churn risk distribution
3. Recommended actions for the business (3-4 specific recommendations)

Keep it concise and actionable."""

                    # Call Claude API
                    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1024,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Display insights
                    st.success("✅ AI Insights Generated!")
                    st.markdown(message.content[0].text)
                    
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    st.info("Make sure your ANTHROPIC_API_KEY is set correctly in the .env file")
