from sklearn.pipeline import Pipeline

class Model:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)