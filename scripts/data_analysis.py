import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    updated_df = df.copy()
# Remove null columns
    updated_df = updated_df.drop(columns=['license', 'neighbourhood_group'], errors='ignore', axis=1)

# Handle missing values
    updated_df['price'].fillna(updated_df['price'].median(), inplace=True)
    updated_df['number_of_reviews_ltm'].fillna(updated_df['number_of_reviews_ltm'].median(), inplace=True)
    updated_df['reviews_per_month'].fillna(updated_df['reviews_per_month'].mean(), inplace=True)
    updated_df['last_review'].ffill(inplace=True)
    updated_df['availability_365'] = updated_df['availability_365'].clip(lower=0)

# Correct data types
    updated_df['last_review'] = pd.to_datetime(updated_df['last_review'], format='%Y-%m-%d', errors='coerce')
    updated_df['neighbourhood'] = updated_df['neighbourhood'].astype('category')
    updated_df['room_type'] = updated_df['room_type'].astype('category')
    updated_df['availability_365'] = pd.to_numeric(updated_df['availability_365'], errors='coerce')
    updated_df['price'] = pd.to_numeric(updated_df['price'].replace('[$]', '', regex=True), errors='coerce')

# Handle outliers
# Calculate IQR for price
    upper_limit = updated_df['price'].quantile(0.95)
    cleaned_df = updated_df[updated_df['price'] <= upper_limit]
    updated_df = cleaned_df.copy()

# Calculate IQR for number_of_reviews
    upper_limit = updated_df['number_of_reviews'].quantile(0.95)
    cleaned_df = updated_df[updated_df['number_of_reviews'] <= upper_limit]
    updated_df = cleaned_df.copy()
# Create price_category
    updated_df['price_category'] = pd.cut(updated_df['price'], bins=[0, 100, 175, np.inf], 
                                         labels=['Budget', 'Mid-range', 'Luxury'], include_lowest=True)
    updated_df['price_category'] = updated_df['price_category'].astype('category')
    return updated_df


def analyze_data(updated_df):
# Relationship between variables
    price_stats = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['price'].size().unstack(fill_value=0)
    prices = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['price'].agg(['min', 'max']).round(2)
    price_stats_2 = updated_df.groupby(['neighbourhood', 'price_category'], observed=True)['price'].mean().unstack(fill_value=0).round(2)
    avg_price_stats = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['price'].mean().round(2)
    avg_price_stats2 = updated_df.groupby('neighbourhood', observed=True)['price'].mean().round(2)
    price_summary1 = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['price'].agg(['size', 'min', 'max', 'mean']).round(2)
    price_summary2 = updated_df.groupby(['neighbourhood', 'price_category'], observed=True)['price'].agg(['size', 'min', 'max', 'mean']).round(2)

# Price by neighbourhood and price_category and room_type
    price_stats = updated_df.groupby(['neighbourhood', 'price_category'], observed=True)['price'].mean().unstack(fill_value=0).round(2)
    print("\nAverage Price by Neighbourhood and Price Category:")
    print(price_stats)    
# Plot price by neighbourhood and room_type
    price_by_neighbourhood_room = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['price'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=price_by_neighbourhood_room, x='neighbourhood', y='price', hue='room_type')
    plt.title('Average Price by Neighbourhood and Room Type')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.savefig('outputs/price_by_neighbourhood_room_type.png')
    plt.close()

# Price by neighbourhood
    cheapest_avg_ward = avg_price_stats2.idxmin()
    cheapest_avg_price = avg_price_stats2.min()
    most_expensive_avg_ward = avg_price_stats2.idxmax()
    most_expensive_avg_price = avg_price_stats2.max()
    print(f"cheapest ward: {cheapest_avg_ward}, average price: ${cheapest_avg_price:.2f}")
    print(f"most expensive ward: {most_expensive_avg_ward}, average price: ${most_expensive_avg_price:.2f}")
#Wards with the highest price in each price category
    max_price_by_category = updated_df.groupby(['price_category', 'neighbourhood'], observed=True)['price'].max().reset_index()
    highest_prices = max_price_by_category.loc[max_price_by_category.groupby('price_category', observed=True)['price'].idxmax()]
    highest_prices = highest_prices.merge(updated_df[['neighbourhood', 'room_type', 'price']], on=['neighbourhood', 'price'], how='left')
    print("\nWards with the highest price in each price category")
    print(highest_prices[['neighbourhood', 'room_type', 'price_category', 'price']])
#Wards with the lowest price in each price category
    min_price_by_category = updated_df.groupby(['price_category', 'neighbourhood'], observed=True)['price'].min().reset_index()
    lowest_prices = min_price_by_category.loc[min_price_by_category.groupby('price_category', observed=True)['price'].idxmin()]
    lowest_prices = lowest_prices.merge(updated_df[['neighbourhood', 'room_type', 'price']], on=['neighbourhood', 'price'], how='left')
    print("\nWards with the lowest price in each price category")
    print(lowest_prices[['neighbourhood', 'room_type', 'price_category', 'price']])
# plot of price by neighbourhood
    sns.barplot(x='neighbourhood', y='price', color='skyblue', data=updated_df, errorbar=None)
    plt.title('Average prices of each Neighborhood')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Price($)')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.figure(figsize=(20, 10))
    plt.savefig('average_prices_neighbourhood.png')
    plt.show()
# BoxPlot: Price Distribution by Room Type
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='room_type', y='price', data=updated_df, color='skyblue')
    plt.title('Price Distribution by Room Type')
    plt.xlabel('Room Type')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
# ScatterPlot: Price vs Number of reviews
    sns.scatterplot(x='number_of_reviews', y='price', data=updated_df, hue='room_type', palette='Set2', alpha=0.6)
    plt.title('Price vs. Number of Reviews')    
    plt.xlabel('Number of Reviews') 
    plt.ylabel('Price ($)')
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.show() 
# ScatterPlot: Price vs Availability
    sns.scatterplot(x='availability_365', y='price', hue='room_type', data=updated_df)
    plt.title('Price vs. Availability (Days)')
    plt.xlabel('Availability (Days)')
    plt.ylabel('Price ($)')
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.show()
# Historgram: Distribution of Prices
    sns.histplot(updated_df['price'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Prices')
    plt.xlabel('Price ($)') 
    plt.ylabel('Frequency')
    plt.tight_layout() 
    plt.show()
# Countplot: Distribution of Price category
    sns.countplot(x='price_category', hue='room_type', data=updated_df, palette='Set2')
    plt.title('Distribution of Price Categories')
    plt.xlabel('Price Category')
    plt.ylabel('Frequency')
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.show()
# Plot for Price Category by Neighbourhood
    plt.figure(figsize=(16, 8))
    sns.countplot(x='neighbourhood', hue='price_category', data=updated_df, palette='viridis')
    plt.title('Distribution of Neighborhood and Price Category')
    plt.xlabel('Neighborhood')
    plt.ylabel('Count')
    plt.legend(title='Price Category')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
# Plot for Avg Price of each Room Type
    sns.barplot(x='price_category', y='price', hue='room_type', data=updated_df, palette='Set2', errorbar=None)
    plt.title('Distribution of Price Categories')
    plt.xlabel('Price Category')
    plt.ylabel('Price ($)')
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.show()

# Availability of each Neighbourhood by RoomType
    avail_room_stats = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['availability_365'].mean().round(2)
    min_avail_room = avail_room_stats.idxmin()
    min_avail_details = avail_room_stats.min()
    print(f"The least available room is: {min_avail_room}, Availability: {min_avail_details:.0f}")
    max_avail_room = avail_room_stats.idxmax()
    max_avail_details = avail_room_stats.max()
    print(f"The most available room is: {max_avail_room}, Availability: {max_avail_details:.0f}")
# plot of availability by neighbourhood and room type
    avail_room_stats.plot(kind='bar', figsize=(12, 8))
    plt.xlabel('Neighbourhood')
    plt.ylabel('Availability (Days)')
    plt.title('Average Availability by Neighbourhood and Room Type')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('availability_by_neighbourhood_roomtype.png')
    plt.show()

# Availability of each RoomType by Price Category
    avail_room_price = updated_df.groupby(['room_type', 'price_category'], observed=True)['availability_365'].mean().round(2)
    min_avail_room_price = avail_room_price.idxmin()
    min_avail_price = avail_room_price.min()
    print(f"The least available room by price category is: {min_avail_room_price}, Availability: {min_avail_price:.0f}")
    max_avail_room_price = avail_room_price.idxmax()
    max_avail_price = avail_room_price.max()
    print(f"The most available room by price category is: {max_avail_room_price}, Availability: {max_avail_price:.0f}")
#plot of availability by room type and price category
    sns.barplot(x='room_type', y='availability_365', data=updated_df, hue='price_category', palette='Set2', errorbar=None)
    plt.xlabel('Room Type')
    plt.ylabel('Availability (Days)')
    plt.title('Room Type by Price Category and Availability')
    plt.legend(title='Price Category')
    plt.tight_layout()  
    plt.savefig('room_type_by_price_category_and_availability.png')
    plt.show()

# Availability of each Neighbourhood
    avail_ward_stats = updated_df.groupby('neighbourhood', observed=True)['availability_365'].mean().round(0)
    min_avg_avail_ward = avail_ward_stats.idxmin()
    min_avg_avail_ward_details = avail_ward_stats.min()
    max_avg_avail_ward = avail_ward_stats.idxmax()
    max_avg_avail_details = avail_ward_stats.max()
    print(f"\nThe least available ward (Avg) is: {min_avg_avail_ward}, Availability: {min_avg_avail_ward_details}")
    print(f"The most available ward (Avg) is: {max_avg_avail_ward}, Availability: {max_avg_avail_details}")
#plot of availability by neighbourhood by RoomType
    avail_ward_stats.plot(kind='bar', figsize=(12, 8))
    plt.xlabel('Neighbourhood')
    plt.ylabel('Availability (Days)')
    plt.title('Average Availability by Neighbourhood by Room Type')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
#plot Availability by Neighbourhood (Avg)
    avail_ward_stats.plot(kind='bar', figsize=(12, 8))
    plt.xlabel('Neighbourhood')
    plt.ylabel('Availability (Days)')
    plt.title('Average Availability by Neighbourhood')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
# Plot for Room Type by Price category and Availability
    sns.barplot(x='room_type', y='availability_365', data=updated_df, hue='price_category', palette='Set2', errorbar=None)
    plt.xlabel('Room Type')
    plt.ylabel('Availability (Days)')
    plt.title('Room Type by Price Category and Availability')
    plt.legend(title='Price Category')
    plt.tight_layout()
    plt.show()

# Number of Reviews (Avg)
    review_summary = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['number_of_reviews'].agg(['size', 'min', 'max', 'mean']).round(2)
    review_stats = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['number_of_reviews'].mean().round(2)
    print(f"\nMost Reviewed Neighbourhood: {review_stats.idxmax()}, Reviews={review_stats.max()}")
    print(f"\nLeast Reviewed Neighbourhood: {review_stats.idxmin()}. Reviews={review_stats.min()}")
#plot of number of reviews of neighbourhood (Avg)
    sns.barplot(x='neighbourhood', y='number_of_reviews', color='skyblue', data=updated_df, errorbar=None)
    plt.title('Number of Reviews by Neighborhood in Albany')
    plt.xlabel('Neighborhood')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
#plot of number of reviews of neighbourhood by room type
    sns.barplot(x='neighbourhood', y='number_of_reviews', hue='room_type', palette='Set2', data=updated_df, errorbar=None)
    plt.title('Number of Reviews by Neighborhood by Room Type')
    plt.xlabel('Neighborhood')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=90)
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.show()
# Plot for Number of Reviews of Room Type by Price Category
    sns.barplot(x='room_type', y='number_of_reviews', hue='price_category', data=updated_df, palette='Set1', errorbar=None)
    plt.title('Room Type by Price Category and Number of Reviews')
    plt.xlabel('Room Type')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=0)
    plt.legend(title='Price Category')
    plt.tight_layout()
    plt.show()

# Monthly Reviews (Avg)
    monthly_review_summary = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['reviews_per_month'].agg(['size', 'min', 'max', 'mean']).round(2)
    monthly_review_stats = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['reviews_per_month'].mean().round(2)
    print(f"\nMost reviewed ward per month: {monthly_review_stats.idxmax()}, Reviews= {monthly_review_stats.max()}")
    print(f"\nLeast reviewed ward per month: {monthly_review_stats.idxmin()}, Reviews= {monthly_review_stats.min()}")
#plot of Monthly Reviews of each Neighbourhood
    sns.barplot(x='neighbourhood', y='reviews_per_month', data=updated_df, color='skyblue', errorbar=None)
    plt.title('Avg Monthly Reviews of each Neighbourhood')
    plt.xlabel('Neighbourhood')
    plt.xticks(rotation=90)
    plt.ylabel('Avg Monthly Reviews')
    plt.tight_layout()
    plt.show()
#plot of Monthly Reviews of each Neighourhood by RoomType
    sns.barplot(x='neighbourhood', y='reviews_per_month', data=updated_df, hue='room_type', palette='Set2', errorbar=None)
    plt.title('Avg Monthly Reviews of each Neighbourhood by Room Type')
    plt.xlabel('Neighbourhood')
    plt.xticks(rotation=90)
    plt.ylabel('Avg Monthly Reviews')
    plt.legend(title='Room Type')
    plt.tight_layout()
    plt.show()

# Number of reviews ltm (Last 12 Months)
    reviews_ltm_stats = updated_df.groupby('neighbourhood', observed=True).agg({'number_of_reviews_ltm' : ['sum', 'mean']}).round(2)
    reviews_ltm_stats.columns = ['total_reviews_ltm', 'avg_reviews_ltm']
    reviews_ltm_stats = reviews_ltm_stats.sort_values('total_reviews_ltm', ascending=False)
    print(reviews_ltm_stats)
    top_reviews_ltm = reviews_ltm_stats['total_reviews_ltm'].idxmax()
    least_reviews_ltm = reviews_ltm_stats['total_reviews_ltm'].idxmin()
    print(f"The ward with the least reviews in the last 12 months is: {least_reviews_ltm}, Total reviews: {reviews_ltm_stats['total_reviews_ltm'].min()}")
    print(f"The ward with the most reviews in the last 12 months is: {top_reviews_ltm}, Total review is: {reviews_ltm_stats['total_reviews_ltm'].max()}")
#plot Number of reviews ltm
    reviews_ltm_stats.plot(kind='barh', figsize=(20, 10), color=['skyblue', 'red'])
    plt.title('Reviews in the last 12 months')
    plt.xlabel('Reviews')
    plt.ylabel('Neighbourhood')
    plt.tight_layout()
    plt.show()

#Total Reviews in the last 12 months vs Availability of each ward (Avg)
    reviews_ltm_price = updated_df.groupby('neighbourhood', observed=True).agg({'number_of_reviews_ltm' : 'sum', 'availability_365' : 'mean'}).round(2)
    reviews_ltm_price.columns = ['total_reviews_ltm', 'avg_availability']
    reviews_ltm_price = reviews_ltm_price.sort_values('total_reviews_ltm', ascending=False)
# plot Total Reviews in the last 12 months vs Availability of each ward (Avg)
    reviews_ltm_price.plot(kind='bar', figsize=(20, 10), color=['skyblue', 'green'])
    plt.title('Reviews in the last 12 months')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Reviews')
    plt.tight_layout()
    plt.show()

# Minimun Nights spent (Avg)
    nights_stats = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['minimum_nights'].mean().round(0)
    nights_summary = updated_df.groupby(['neighbourhood', 'room_type'], observed=True)['minimum_nights'].agg(Count='size', Min= 'min', Max= 'max', Average_night_spent= 'mean').round(0)
    print(f"Most Night spent in a Neighbourhood: {nights_stats.idxmax()}, Nights: {nights_stats.max()}")
    print(f"Least Night spent in a Neighbourhood: {nights_stats.idxmin()}, Nights: {nights_stats.min()}")
#plot Minimum Nights by Room Type
    sns.barplot(x='neighbourhood', y='minimum_nights', hue='room_type', data=updated_df, errorbar=None)
    plt.title('Average Minimum Nights by Neighborhood in Albany')
    plt.xlabel('Neighbourhood')
    plt.ylabel('Minimum Nights')
    plt.legend(title='Room Type')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
# Plot of Minimum Nights vs Number of Reviews
    sns.scatterplot(x='minimum_nights', y='number_of_reviews', data=updated_df, alpha=0.5)
    plt.title('Minimum Nights vs Number of Reviews')
    plt.xlabel('minimum Nights')
    plt.ylabel('Number of Reviews')
    plt.show()

if __name__ == "__main__":
    updated_df = load_and_preprocess_data('data/listings.csv')
    analyze_data(updated_df)