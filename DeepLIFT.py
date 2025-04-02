import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from model_class import Model  # Ensure this class definition is available

# Load the dataset (used for selecting a background sample)
data = pd.read_csv('diabetes.csv')
# Use only the raw features; the pipeline will do any additional transformations.
X = data[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]

# Load the saved model from file
model_instance = joblib.load('model.pkl')

# Select a background sample from the training data for the explainer.
# Here we use 100 random samples. Adjust as needed.
background = shap.sample(X, 150)

# Select one instance to explain.
# In this example, we use the first instance.
sample = X.iloc[0:1]

# Wrapper function for KernelExplainer
def proba_class_1(x):
    return model_instance.predict_proba(x)[:, 1]

# Initialize the SHAP KernelExplainer.
explainer = shap.KernelExplainer(proba_class_1, background)

# Compute SHAP values for the sample.
shap_values = explainer.shap_values(sample)

# ------------------------
# Generate the Force Plot (Static Image)
# ------------------------
try:
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], sample, matplotlib=True)
    plt.savefig("force_plot.png")  # Save as PNG
    print("Force plot saved as 'force_plot.png'")
    plt.clf() #clear figure to avoid overlap with next plot.
except Exception as e:
    print(f"Error generating force plot: {e}")
    print("Skipping force plot generation.")

# ------------------------
# Generate the Waterfall Plot (Static Image)
# ------------------------
try:
    exp = shap.Explanation(values=shap_values[0],
                           base_values=explainer.expected_value,
                           data=sample.values[0],
                           feature_names=['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age'])

    shap.plots.waterfall(exp, show=False)  # show=False to prevent immediate display
    plt.savefig("waterfall_plot.png")  # Save as PNG
    print("Waterfall plot saved as 'waterfall_plot.png'")
    plt.clf() #clear figure to avoid overlap with next plot.

except Exception as e:
    print(f"Error generating waterfall plot: {e}")
    print("Skipping waterfall plot generation.")

# ------------------------
# Data Table (for sample)
# ------------------------
print("\nSample Data:")
print(sample)