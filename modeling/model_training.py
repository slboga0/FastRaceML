import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import label_binarize
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

def remap_labels(y):
    """
    Remap labels. For numeric or multi-dimensional targets (regression),
    return as a NumPy array; for classification, implement mapping as needed.
    """
    y_array = np.array(y)
    if np.issubdtype(y_array.dtype, np.number) or y_array.ndim > 1:
        return y_array, {}
    else:
        mapping = {}  # Implement mapping logic if needed.
        y_mapped = np.array([mapping[label] for label in y])
        return y_mapped, mapping

def lgb_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    num_class = len(np.unique(y_train))
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42,
        'metric': 'multi_logloss'
    }
    
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000)
    feature_importance = model.feature_importance(importance_type='gain')
    
    test_preds = model.predict(X_test)
    n_classes = test_preds.shape[1]
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    test_auc = roc_auc_score(y_test_bin, test_preds, multi_class='ovr')
    logger.info(f'Test AUC: {test_auc:.4f}')
    
    y_pred_class = np.argmax(test_preds, axis=1)
    accuracy = accuracy_score(y_test, y_pred_class)
    logger.info(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred_class)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred_class))
    
    roc_auc = roc_auc_score(y_test_bin, test_preds, multi_class='ovr')
    logger.info(f"Multiclass ROC AUC (OvR): {roc_auc:.4f}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"Cross-Validation Accuracy Scores: {scores}")
    logger.info(f"Mean CV Accuracy: {scores.mean()}")

    return model, feature_importance

def random_forest_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    logger.info(f'Validation AUC: {val_auc:.4f}')
    logger.info(f'Test AUC: {test_auc:.4f}')
    return model, feature_importance

def binary_regression_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    logger.info(f"Validation Accuracy: {val_acc:.2f}")

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    logger.info(f"Test Accuracy: {test_acc:.2f}")

    logger.info("Classification Report (Test):\n" + classification_report(y_test, y_test_pred))

    coefs = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)
    logger.info(f"Top 10 features:\n{coefs.head(10)}")

    return model, coefs
