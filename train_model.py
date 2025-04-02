import joblib
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from custom_transformers import FeatureEngineering, WoEEncoding, ColumnSelector

class Model:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self,X):
        return self.pipeline.predict(X)

# Load data
data = pd.read_csv('diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]
y = data['Outcome']

# Create the pipeline
def create_pipeline(params):
    """
    Create a pipeline with preprocessing and a neural network classifier.
    """
    mlp_params = {
        'hidden_layer_sizes': (int(params['hidden_layer_size']),),
        'activation': 'relu',  # fixed for now
        'solver': 'adam',  # fixed for now
        'alpha': params['alpha'],
        'learning_rate_init': params['learning_rate_init'],
        'max_iter': 500,  # increased iterations for better convergence
        'random_state': 42
    }

    mlp_clf = MLPClassifier(**mlp_params)

    pipeline = Pipeline(steps=[
        ('feature_eng', FeatureEngineering()),
        ('woe', WoEEncoding()),
        ('selector', ColumnSelector(columns=[
            'Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age',
            'PregnancyRatio', 'RiskScore', 'InsulinEfficiency',
            'Glucose_BMI', 'BMI_Age', 'Pregnancies_woe', 'Glucose_woe', 'BMI_woe', 'RiskScore_woe'
        ])),
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', mlp_clf)
    ])
    return pipeline

def objective(hidden_layer_size, alpha, learning_rate_init):
    """
    Objective function for Bayesian Optimization.
    """
    params = {
        'hidden_layer_size': hidden_layer_size,
        'alpha': alpha,
        'learning_rate_init': learning_rate_init
    }
    pipeline = create_pipeline(params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=make_scorer(roc_auc_score))
    return scores.mean()

# Define the bounds for Bayesian Optimization:
pbounds = {
    'hidden_layer_size': (50, 200),
    'alpha': (1e-5, 1e-1),
    'learning_rate_init': (1e-4, 1e-1)
}

if __name__ == '__main__':
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=10, n_iter=90)

    print("Best hyperparameters found:", optimizer.max)

    best_params = optimizer.max['params']
    best_params['hidden_layer_size'] = int(best_params['hidden_layer_size'])

    best_pipeline = create_pipeline(best_params)
    model_instance = Model(best_pipeline)

    model_instance.fit(X, y)

    y_pred = model_instance.predict_proba(X)[:, 1]

    print("ROC_AUC Score: ", (roc_auc_score(y, y_pred) * 100).round(2))

    joblib.dump(model_instance, 'model.pkl')
    print("Model saved as 'model.pkl'")