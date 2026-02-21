import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_model(X, y):
    """
    Train a Random Forest Classifier and save the best model.
    
    Parameters:
    -----------
    X : pandas DataFrame or numpy array
        Feature matrix
    y : pandas Series or numpy array
        Target vector (should be 1D, binary in this case)
    
    Returns:
    --------
    Trained model object
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42,
        stratify=y          # important for imbalanced classes (machine failure is rare)
    )
    
    # Initialize model - using class_weight='balanced' because failure cases are rare
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1               # use all CPU cores → faster training
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    predictions = model.predict(X_test)
    
    # Print evaluation
    print("\n" + "="*60)
    print("Classification Report on Test Set:")
    print("="*60)
    print(classification_report(y_test, predictions))
    
    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model