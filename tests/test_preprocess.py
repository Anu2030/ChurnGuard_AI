import os
import sys
import pandas as pd
import numpy as np
import pytest

# Add base directory to path so we can import src.config and src.preprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.preprocess import clean_data


def make_raw_df():
    """Helper: returns a small DataFrame mimicking raw telco data."""
    return pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003'],
        'tenure':     [0,      10,     5],
        'MonthlyCharges': [50.0, 20.0, 30.0],
        'TotalCharges':   [' ', '200.0', '150.0'],  # blank for new customer
        'SeniorCitizen':  [0, 1, 0],
        'Churn':          ['No', 'Yes', 'No'],
    })


def test_clean_data_returns_dataframe():
    """clean_data should always return a DataFrame."""
    df_clean = clean_data(make_raw_df())
    assert isinstance(df_clean, pd.DataFrame)


def test_total_charges_is_float():
    """TotalCharges must be numeric (float) after cleaning."""
    df_clean = clean_data(make_raw_df())
    assert df_clean['TotalCharges'].dtype == float


def test_blank_total_charges_filled_not_dropped():
    """
    Blank TotalCharges (new customers with tenure=0) should be filled with
    MonthlyCharges, NOT dropped, so row count is preserved.
    """
    raw = make_raw_df()
    df_clean = clean_data(raw)
    # All 3 rows should still be present
    assert len(df_clean) == 3
    # No NaN values should remain
    assert not df_clean['TotalCharges'].isna().any()
    # The blank cell should now equal the MonthlyCharges of that row
    assert df_clean.loc[0, 'TotalCharges'] == pytest.approx(50.0)


def test_senior_citizen_mapped_to_yes_no():
    """SeniorCitizen integers should be mapped to 'Yes'/'No' strings."""
    df_clean = clean_data(make_raw_df())
    valid_values = {'Yes', 'No'}
    assert set(df_clean['SeniorCitizen'].unique()).issubset(valid_values)


def test_customer_id_preserved():
    """customerID column must be preserved in the cleaned output (needed for dashboard/RFM)."""
    df_clean = clean_data(make_raw_df())
    assert 'customerID' in df_clean.columns

