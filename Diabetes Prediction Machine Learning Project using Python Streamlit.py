import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model_class import Model  # Ensure this class definition is available
import Diabetes_info  # Assuming this module exists
import dill

# Load the model using joblib
try:
    loaded_model = joblib.load('model.pkl')  # Use .pkl and joblib
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.pkl' is in the correct path.")
    st.stop()


def predict_diabetes(input_data):
    input_df = pd.DataFrame([{
        'Pregnancies': input_data[0],
        'Glucose': input_data[1],
        'Insulin': input_data[2],
        'BMI': input_data[3],
        'Age': input_data[4]
    }])
    try:
        prediction = loaded_model.predict(input_df)[0]
        prob = loaded_model.predict_proba(input_df)[0][1]
        return prediction, prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


def load_real_data():
    try:
        df = pd.read_csv("diabetes.csv")  # Use relative path if possible
        df = df[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age', 'Outcome']]
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'diabetes.csv' is in the correct path.")
        return pd.DataFrame()


st.title("Diabetes Prediction Web App")

with st.sidebar.expander("Enter Your Health Parameters", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])

    # For females, show the pregnancies input widget.
    if gender == "Female":
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=1)
    else:
        # For males, hide the pregnancy row and automatically set pregnancies to 0.
        pregnancies = 0

    glucose = st.number_input("Glucose", min_value=0.0, step=1.0, value=100.0)
    insulin = st.number_input("Insulin", min_value=0.0, step=1.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1, value=25.0)
    age = st.number_input("Age", min_value=0, step=1, value=30)

    if st.button("Predict and Explain"):
        input_features = [pregnancies, glucose, insulin, bmi, age]
        pred, prob = predict_diabetes(input_features)
        if pred is not None:
            result_text = "Diabetic" if pred == 1 else "Non Diabetic"
            st.subheader("Prediction")
            st.write(f"**Result:** {result_text}")
            st.write(f"**Probability of Diabetes:** {prob:.2f}")

            # -------------------------
            # SHAP Visualizations (for user input)
            # -------------------------
            st.subheader("Model Explanation with SHAP")

            input_df = pd.DataFrame([{
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'Insulin': insulin,
                'BMI': bmi,
                'Age': age
            }])

            # Use the same features from df_real for background. Fallback to diabetes.csv if needed.
            X = load_real_data()[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]
            if X.empty:
                data = pd.read_csv('diabetes.csv')
                X = data[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]

            # Select a background sample (100 samples) for SHAP explanations.
            background = shap.sample(X, 100)
            # Select the user input instance for explanation.
            sample = input_df


            # Define a function to return the probability for class 1.
            def proba_class_1(x):
                return loaded_model.predict_proba(x)[:, 1]


            # Initialize the KernelExplainer.
            explainer = shap.KernelExplainer(proba_class_1, background)
            shap_values = explainer.shap_values(sample)

            # -------------------------
            # Force Plot Visualization
            # -------------------------
            st.markdown("### Force Plot")
            force_plot = shap.force_plot(explainer.expected_value, shap_values[0], sample, matplotlib=False)
            force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(force_html, height=350)

            # -------------------------
            # Waterfall Plot Visualization
            # -------------------------
            st.markdown("### Waterfall Plot")
            exp = shap.Explanation(values=shap_values[0],
                                   base_values=explainer.expected_value,
                                   data=sample.values[0],
                                   feature_names=['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age'])
            plt.figure()
            shap.plots.waterfall(exp, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

Diabetes_info.show_diabetes_info()

# -------------------------
# Model Performance Metrics
# -------------------------
df_real = load_real_data()
if not df_real.empty:
    y_real = df_real['Outcome']
    X_real = df_real[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]
    try:
        y_pred = loaded_model.predict(X_real)
        acc = accuracy_score(y_real, y_pred)
        prec = precision_score(y_real, y_pred, zero_division=0)
        rec = recall_score(y_real, y_pred, zero_division=0)
        f1 = f1_score(y_real, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_real, loaded_model.predict_proba(X_real)[:, 1])

        st.subheader("Model Performance")
        metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'ROC AUC': roc_auc}
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics_df['Metric'], metrics_df['Value'],
               color=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral'])
        ax.set_ylabel('Value')
        ax.set_title('Model Performance Metrics')
        ax.set_ylim(0, 1.05)  # Set y-axis limit for readability.
        for index, value in enumerate(metrics_df['Value']):
            ax.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error calculating model performance: {e}")
else:
    st.error("No valid data available from the CSV file.")
    st.stop()