"""
Water Contamination  Prediction Model Training Script
This script loads the car dataset, preprocesses it, trains multiple ML models,
and saves the best performing model along with preprocessing objects.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import re

def load_and_clean_data(filepath):
    """Load and clean the data"""
    print("Loading data...")
    df = pd.read_csv(filepath)

    print(f"Initial dataset shape: {df.shape}")
    print(f"columns: {df.columns}")
    print("\n Cleaning data...")

