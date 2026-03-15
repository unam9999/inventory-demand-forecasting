import pytest
import pandas as pd
import numpy as np
from src.models import NaiveBaseline, MovingAverageBaseline

def test_naive_baseline():
    model = NaiveBaseline()
    X = pd.DataFrame({'lag_1': [10, 20, 30]})
    preds = model.predict(X)
    assert np.array_equal(preds, [10, 20, 30])

def test_moving_average_baseline():
    model = MovingAverageBaseline()
    X = pd.DataFrame({'rolling_mean_7': [15, 25, 35]})
    preds = model.predict(X)
    assert np.array_equal(preds, [15, 25, 35])
