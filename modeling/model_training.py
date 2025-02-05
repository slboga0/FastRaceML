# FastRaceML/modeling/model_training.py

"""
Machine learning training utilities, e.g., logistic regression, 
model evaluation, etc.
"""

import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

def binary_regression_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train, validate, and test a logistic regression model.
    Returns:
        model (LogisticRegression)
        feature_importance (pd.DataFrame)
    """
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    # Validation
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    logger.info(f"Validation Accuracy: {val_acc:.2f}")

    # Test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    logger.info(f"Test Accuracy: {test_acc:.2f}")

    # Classification report
    logger.info("Classification Report (Test):\n" + classification_report(y_test, y_test_pred))

    # Feature importance
    coefs = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)
    logger.info(f"Top 10 features:\n{coefs.head(10)}")

    return model, coefs
