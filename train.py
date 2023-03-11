import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def run_training(): 
    """
    Train the model
    """
    # Read the training data
    dataset = pd.read_csv(
        filepath_or_buffer= "/workspaces/msds434Final2/data.csv"
        )

    # Split into labels and targets
    X = dataset.drop("species", axis=1).copy()
    y = dataset["species"].copy()

    # Create train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=26)

    # Training the model
    model = LogisticRegression(random_state=26)
    model.fit(X_train, y_train)

    # Persist the trained model 
    joblib.dump(model, "logistic_regression_v1.pkl")

if __name__ == "__main__":
    run_training()
