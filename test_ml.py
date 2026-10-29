import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def sample_data():
    """Create a minimal representative dataframe for testing."""
    df = pd.DataFrame(
        {
            "age": [39, 50, 38, 53],
            "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private"],
            "education": ["Bachelors", "Bachelors", "HS-grad", "11th"],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
            ],
            "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Handlers-cleaners"],
            "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband"],
            "race": ["White", "White", "White", "Black"],
            "sex": ["Male", "Male", "Male", "Male"],
            "native-country": ["United-States", "United-States", "United-States", "United-States"],
            "salary": ["<=50K", "<=50K", "<=50K", "<=50K"],
        }
    )
    return df


def test_process_data_shapes(sample_data):
    """Test process_data returns valid shapes and consistent row counts."""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == sample_data.shape[0]
    assert len(y) == sample_data.shape[0]
    assert X.shape[1] > 0
    assert encoder is not None
    assert lb is not None


def test_train_model_algorithm():
    """Test train_model fits a RandomForestClassifier and runs inference."""
    X_dummy = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    y_dummy = np.array([0, 1, 1, 0])

    model = train_model(X_dummy, y_dummy)

    assert isinstance(model, RandomForestClassifier)
    preds = inference(model, X_dummy)
    assert len(preds) == len(y_dummy)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics():
    """Test compute_model_metrics outputs exact expected numerical values."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    # 2 TP, 0 FP, 1 FN -> Precision: 1.0, Recall: 2/3, F1: 0.8
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)
