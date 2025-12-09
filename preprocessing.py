# preprocessing.py
from simulation import create_feature_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def prepare_data(n_mice=5, n_trials_per_mouse=40, test_size=0.2, random_state=42):
    """
    Prepare data for ML models.
    
    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training labels (1=Correct, 0=Incorrect)
        y_test: Test labels
        feature_cols: List of feature column names
        scaler: Fitted StandardScaler object
    """
    
    # Generate data
    feature_matrix, trial_labels = create_feature_matrix(n_mice, n_trials_per_mouse, random_state=random_state)
    
    # Create target
    feature_matrix['Correct'] = (feature_matrix['Hit'] + feature_matrix['CR']).astype(int)
    
    # One-hot encode learning stage
    # Converts: "Naive", "Learning", "Proficient" → binary columns
    # drop_first=True: Uses Naive as reference (encoded as [0,0])
    feature_matrix_encoded = pd.get_dummies(feature_matrix, 
                                            columns=['Learning_Stage'], 
                                            drop_first=True)
    
    # Select features
    feature_cols = ['theta_frequency', 'beta_frequency', 'gamma_frequency', 
                    'noise_level', 'Trial_Number',
                    'Learning_Stage_Naive', 'Learning_Stage_Proficient']
    
    X = feature_matrix_encoded[feature_cols]
    y = feature_matrix_encoded['Correct']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def prepare_data_proficient_only(n_mice=5, n_trials_per_mouse=40, test_size=0.2, random_state=42):
    """
    Prepare data using only Proficient stage trials.
    This removes the learning stage confound and focuses on neural predictors.
    
    Returns:
        X_train_scaled: Scaled training features (NO learning stage columns)
        X_test_scaled: Scaled test features
        y_train: Training labels
        y_test: Test labels
        feature_cols: List of feature column names
        scaler: Fitted StandardScaler object
    """
    
    # Generate data
    feature_matrix, trial_labels = create_feature_matrix(n_mice, n_trials_per_mouse, random_state=random_state)
    
    # Filter only proficient trials
    feature_matrix = feature_matrix[feature_matrix['Learning_Stage'] == 'Proficient'].copy()
    
    print(f"Proficient-only dataset: {len(feature_matrix)} trials")
    
    # Create binary target
    feature_matrix['Correct'] = (feature_matrix['Hit'] + feature_matrix['CR']).astype(int)
    
    # Check class balance
    correct_count = feature_matrix['Correct'].sum()
    incorrect_count = (feature_matrix['Correct'] == 0).sum()
    print(f"Correct trials: {correct_count} ({correct_count/len(feature_matrix)*100:.1f}%)")
    print(f"Incorrect trials: {incorrect_count} ({incorrect_count/len(feature_matrix)*100:.1f}%)")
    
    # Select features - No learning stage
    feature_cols = ['theta_frequency', 'beta_frequency', 'gamma_frequency', 
                    'noise_level', 'Trial_Number']
    
    X = feature_matrix[feature_cols]
    y = feature_matrix['Correct']
    
    # Check if we have enough data for both classes
    if correct_count < 2 or incorrect_count < 2:
        raise ValueError("Not enough samples in one or both classes for stratified split!")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def prepare_data_neural_only(n_mice=5, n_trials_per_mouse=40, test_size=0.2, random_state=42):
    """
    Prepare data using ONLY neural features (no behavioral/trial info).
    This is the purest test of whether neural activity predicts behavior.
    
    Returns:
        X_train_scaled: Scaled training features (ONLY neural)
        X_test_scaled: Scaled test features
        y_train: Training labels
        y_test: Test labels
        feature_cols: List of feature column names
        scaler: Fitted StandardScaler object
    """
    
    # Generate data
    feature_matrix, trial_labels = create_feature_matrix(n_mice, n_trials_per_mouse, random_state=random_state)
    
    # Create binary target
    feature_matrix['Correct'] = (feature_matrix['Hit'] + feature_matrix['CR']).astype(int)
    
    # Select only neural features
    feature_cols = ['theta_frequency', 'beta_frequency', 'gamma_frequency', 'noise_level']
    
    X = feature_matrix[feature_cols]
    y = feature_matrix['Correct']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler