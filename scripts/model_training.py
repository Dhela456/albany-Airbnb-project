import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    updated_df = df.copy()
    updated_df = updated_df.drop(columns=['license', 'neighbourhood_group'], errors='ignore')
    updated_df['price'].fillna(updated_df['price'].median(), inplace=True)
    updated_df['number_of_reviews_ltm'].fillna(updated_df['number_of_reviews_ltm'].median(), inplace=True)
    updated_df['reviews_per_month'].fillna(updated_df['reviews_per_month'].mean(), inplace=True)
    updated_df['last_review'].ffill(inplace=True)
    updated_df['availability_365'] = updated_df['availability_365'].clip(lower=0)
    updated_df['price'] = pd.to_numeric(updated_df['price'].replace('[$]', '', regex=True), errors='coerce')
    # Calculate IQR for price
    upper_limit = updated_df['price'].quantile(0.95)
    cleaned_df = updated_df[updated_df['price'] <= upper_limit]
    updated_df = cleaned_df.copy()
    # Calculate IQR for number_of_reviews
    upper_limit = updated_df['number_of_reviews'].quantile(0.95)
    cleaned_df = updated_df[updated_df['number_of_reviews'] <= upper_limit]
    updated_df = cleaned_df.copy()
    updated_df_encoded = pd.get_dummies(updated_df, columns=['neighbourhood', 'room_type'], drop_first=True)
    return updated_df_encoded

def train_model(updated_df_encoded):
    features = ['number_of_reviews', 'number_of_reviews_ltm', 'availability_365', 'minimum_nights'] + \
               [col for col in updated_df_encoded.columns if col.startswith('neighbourhood_') or col.startswith('room_type_')]
    X = updated_df_encoded[features]
    y = updated_df_encoded['price']
    scaler = StandardScaler()
    X[['number_of_reviews', 'number_of_reviews_ltm', 'availability_365', 'minimum_nights']] = scaler.fit_transform(
        X[['number_of_reviews', 'number_of_reviews_ltm', 'availability_365', 'minimum_nights']]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print("\nRandom Forest Model Performance:")
    print(f"MAE: {mean_absolute_error(y_test, rf_pred):.2f} $")
    print(f"MSE: {mean_squared_error(y_test, rf_pred):.2f}")
    print(f"R-squared: {r2_score(y_test, rf_pred):.2f}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=rf_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Random Forest: Actual vs. Predicted Prices')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.tight_layout()
    plt.close()
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.close()
    joblib.dump(rf_model, 'outputs/airbnb_price_model.pkl')
    joblib.dump(scaler, 'outputs/scaler.pkl')
    joblib.dump(X.columns, 'outputs/feature_columns.pkl')
    return rf_model, scaler, X.columns

if __name__ == "__main__":
    updated_df_encoded = load_and_preprocess_data("C:/Users/IreOluwa/Desktop/jupyter/albany-airbnb-project/Data/listings.csv")
    rf_model, scaler, feature_columns = train_model(updated_df_encoded)