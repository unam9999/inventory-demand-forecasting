import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from config import RAW_DATA_FILE

def generate_synthetic_data():
    print("Generating synthetic data...")
    
    # Parameters
    n_stores = 3
    n_products = 10
    start_date = datetime(2023, 1, 1)
    n_days = 730  # 2 years
    
    stores = [1, 2, 3]
    products = list(range(101, 111))
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    data = []
    
    # Base demand for each product
    base_demand = {p: np.random.randint(20, 50) for p in products}
    
    # Seasonality factors (simple weekly pattern)
    # Mon=0, Sun=6. Higher on weekends (Fri, Sat, Sun)
    weekly_seasonality = {0: 0.9, 1: 0.9, 2: 0.9, 3: 1.0, 4: 1.2, 5: 1.3, 6: 1.1}
    
    for date in dates:
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        month = date.month
        
        # Holiday flag (simple: New Year, Christmas, July 4th)
        is_holiday = 0
        if (month == 1 and date.day == 1) or (month == 12 and date.day == 25) or (month == 7 and date.day == 4):
            is_holiday = 1
            
        for store in stores:
            # Store factor (Store 1 is average, 2 is busy, 3 is slow)
            store_factor = {1: 1.0, 2: 1.2, 3: 0.8}[store]
            
            for product in products:
                # Random price fluctuation
                base_price = {p: p/2 for p in products}[product] # Arbitrary price logic
                price = round(base_price * np.random.uniform(0.9, 1.1), 2)
                
                # Promotion (random 5% chance)
                promotion = 1 if np.random.random() < 0.05 else 0
                promo_factor = 1.5 if promotion else 1.0
                
                # Holiday factor
                holiday_factor = 1.3 if is_holiday else 1.0
                
                # Trend (slight increase over time)
                days_passed = (date - start_date).days
                trend_factor = 1 + (days_passed / n_days) * 0.1
                
                # Calculate expected demand (lambda for Poisson)
                expected_demand = (base_demand[product] * 
                                   store_factor * 
                                   weekly_seasonality[day_of_week] * 
                                   promo_factor * 
                                   holiday_factor * 
                                   trend_factor)
                
                # Add random noise
                units_sold = np.random.poisson(expected_demand)
                
                # Inventory logic
                # Assume we usually have enough, but sometimes stock out
                # If demand is very high, we might cap at inventory
                # Let's simulate on_hand_inventory as usually higher than demand
                on_hand_inventory = int(expected_demand * np.random.uniform(1.0, 2.0))
                
                # Occasional stockout (inventory < demand)
                if np.random.random() < 0.02: # 2% chance of stockout issue
                    on_hand_inventory = np.random.randint(0, int(expected_demand * 0.8))
                
                stockout_flag = 0
                if on_hand_inventory < units_sold:
                    units_sold = on_hand_inventory # Can't sell more than we have
                    stockout_flag = 1
                    if on_hand_inventory == 0:
                        stockout_flag = 1 # Definitely stockout
                
                lead_time = np.random.choice([3, 5, 7])
                
                data.append([
                    date.strftime('%Y-%m-%d'),
                    store,
                    product,
                    on_hand_inventory,
                    units_sold,
                    price,
                    promotion,
                    is_holiday,
                    day_of_week,
                    lead_time,
                    stockout_flag
                ])
                
    columns = [
        'date', 'store_id', 'product_id', 'on_hand_inventory', 'units_sold',
        'price', 'promotion', 'holiday', 'day_of_week', 'lead_time_days', 'stockout_flag'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(RAW_DATA_FILE), exist_ok=True)
    
    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"Data generated and saved to {RAW_DATA_FILE}")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_data()
