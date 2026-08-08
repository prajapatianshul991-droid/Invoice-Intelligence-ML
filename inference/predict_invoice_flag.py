import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\dell\OneDrive\Desktop\New Folder\Projects using DS\Invoice\invoice_flagging\models\predict_flag_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    model = joblib.load(model_path)
    return model


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df


# Example inference run (local testing)
if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [10, 500],
        "invoice_dollars": [200, 15000],
        "Freight": [5, 300],
        "total_item_quantity": [10, 480],
        "total_item_dollars": [195, 15000]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)