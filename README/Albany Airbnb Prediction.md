
# Albany Airbnb Price Prediction Analysis

**Objective**: Predict Airbnb listing prices in Albany using various features such as Neighbourhood, Room Type, Reviews and Availabilty.

**Features**:
- Neighbourhood(e.g, SIXTH WARD, FIFTEENTH WARD)
- Room Type (e.g, entire home/apt, private room)
- Number of Reviews, Reviews LTM, Availability
- Minimum Nights.

**Model**: Random Forest Regressor
- MAE (Mean Asolute Error): 26.09
- R-Squared: 0.38

**Insights**: 
- The Private Room in the SEVENTH WARD has very budget friendly listing with an average price of $50 which explains why it is one of the most reviewed listing with an average of 198 reviews.
- The most expensive listing is in the EIGHTH WARD with an average price of $199.83 for an entire home/apt. Also, the private Room has the highest number of reviews despite having a high average price of $93.
- Room type, specifically the Private Room strongly influences price, with Entire home/apt being pricier.
- The first ward didn't sell any private room.
- The NUMBER OF REVIEWS has a strong negative correlation with the MINIMUM NIGHT spent in each rooms, which means that wards with high number of reviews have low minimum nights spent.
- The private room of both the FIRST and SIXTH ward have zero availability.
- The SIXTH WARD has the highest BUDGET and  MID RANGE price category frequency in the neighbourhood which means the average customer prefers rooms in the SIXTH WARD.
- The Total reviews for each ward in the last 12 months is slightly negatively correlated with the average availability of each ward as ward with higher total reviews in the last 12 months tend to have low availability and wards with higher availability tend to have lower total reviews in the last 12 months. For example: The FIFTH ward has a very low total reviews in the last 12 months but a very high availaility in fact, it is the most available ward of 365 days.

**Visualizations**:
- Actual vs Predicted Prices: 'actual_vs_predicted_prices.png'
- Feature Importance: 'feature_importance.png'
