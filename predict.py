import joblib 
import pandas as pd

model = joblib.load("logistic_regression_v1.pkl")

def make_prediction(inputs): 
    """
    Make a prediction using the trained model 
    """
    inputs_df = pd.DataFrame(
        inputs, 
        columns=["id","sepal_length", "sepal_width", "petal_length", "petal_width"]
        )
    predictions = model.predict(inputs_df)
    
    return predictions
