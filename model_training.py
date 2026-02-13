import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.special import expit
import joblib

class SentimentModel:
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = None
        self.history = {}
        
        # Create model immediately upon initialization
        self.create_model()

    def create_model(self):
        """Initialize the specific machine learning model"""
        models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=20
            ),
            'svm': LinearSVC(
                max_iter=1000,
                random_state=42,
                C=1.0,
                dual='auto'
            )
        }
        
        self.model = models.get(self.model_type)
        if self.model is None:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """Train the model and optionally evaluate on validation set"""
        if self.model is None:
            self.create_model()
            
        # Fit the model
        self.model.fit(x_train, y_train)

        # Calculate training accuracy
        train_pred = self.model.predict(x_train)
        train_acc = accuracy_score(y_train, train_pred)

        print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        self.history['train_accuracy'] = train_acc

        # Evaluation on validation data if provided
        if x_val is not None and y_val is not None:
            val_pred = self.model.predict(x_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
            self.history['val_accuracy'] = val_acc
            
        return self.model

    def evaluate(self, x_test, y_test):
        """Evaluate model performance on test set"""
        y_pred = self.model.predict(x_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        return metrics

    def predict(self, x):
        """Return Class Labels (0 or 1)"""
        return self.model.predict(x)

    def predict_proba(self, x):
        """Return Probabilities (Confidence scores)"""
        # 1. Models that support predict_proba natively (LR, NB, RF)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(x)
        
        # 2. Models that need decision_function conversion (LinearSVC)
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(x)
            proba = expit(decision)
            # Create a 2D array [prob_class_0, prob_class_1]
            return np.vstack([1 - proba, proba]).T
            
        else:
            return None

if __name__ == "__main__":
    model = SentimentModel(model_type="logistic_regression")
    print(f"Model Initialized: {model.model}")