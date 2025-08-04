import pandas as pd
import joblib

# Load model and scaler
rf_model = joblib.load("C:/Users/IreOluwa/Desktop/jupyter/airbnb_price_model.pkl")
scaler = joblib.load("C:/Users/IreOluwa/Desktop/jupyter/albany-airbnb-project/outputs/scaler.pkl")
feature_columns = joblib.load("C:/Users/IreOluwa/Desktop/jupyter/albany-airbnb-project/outputs/feature_columns.pkl")

def predict_new_price(new_listing):
    new_df = pd.DataFrame([new_listing])
    new_df_encoded = pd.get_dummies(new_df, columns=['neighbourhood', 'room_type'], drop_first=True)
    for col in feature_columns:
        if col not in new_df_encoded.columns:
            new_df_encoded[col] = 0
    new_df_encoded = new_df_encoded[feature_columns]
    new_df_encoded[['number_of_reviews', 'number_of_reviews_ltm', 'availability_365', 'minimum_nights']] = scaler.transform(
        new_df_encoded[['number_of_reviews', 'number_of_reviews_ltm', 'availability_365', 'minimum_nights']]
    )
    new_price = rf_model.predict(new_df_encoded)[0]
    print(f"\nPredicted Price for New Listing: ${new_price:.2f}")

if __name__ == "__main__":
    new_listing = {
        'neighbourhood': 'SIXTH WARD',
        'room_type': 'Private room',
        'number_of_reviews': 60,
        'number_of_reviews_ltm': 300,
        'availability_365': 200,
        'minimum_nights': 10
    }
    predict_new_price(new_listing)