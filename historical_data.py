#!/usr/bin/env python3
# Historical Data Processing and Analysis for Dark Store Order Processing
# This module provides functionality for analyzing historical order data
# and using it to predict future order patterns and optimize resource allocation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# For interactive plots (if available)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    plotly_available = True
except ImportError:
    plotly_available = False
    print("Plotly not available. Using matplotlib for visualizations.")

# Default order patterns based on real-world data sample
DEFAULT_ORDER_PATTERNS = {
    'orders_by_day': {
        'Monday': 25,
        'Tuesday': 25,
        'Wednesday': 25,
        'Thursday': 25,
        'Friday': 25,
        'Saturday': 25,
        'Sunday': 20
    },
    'orders_by_hour': {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 
        6: 0, 7: 0, 8: 0, 9: 1, 10: 5, 11: 5,
        12: 4, 13: 5, 14: 4, 15: 1, 16: 0, 17: 1,
        18: 0, 19: 0, 20: 0, 21: 1, 22: 0, 23: 0
    },
    'avg_distance_by_day': {
        'Monday': 6.5,
        'Tuesday': 6.5,
        'Wednesday': 6.5,
        'Thursday': 6.5,
        'Friday': 6.5,
        'Saturday': 6.5,
        'Sunday': 6.5
    },
    'weekend_vs_weekday': {
        'weekend_orders': 45,
        'weekday_orders': 130,
        'avg_orders_weekend': 22.5,
        'avg_orders_weekday': 26.0
    },
    'morning_vs_afternoon': {
        'morning_orders': 11,
        'afternoon_orders': 14,
        'morning_percentage': 44.0
    },
    'total_orders': 175,
    'avg_daily_orders': 25,
    'peak_hour_orders': 5,
    'avg_distance': 6.5
}

def get_default_historical_data():
    """
    Generate a DataFrame with synthesized historical data based on real-world sample
    
    Returns:
    DataFrame: Synthesized historical data with realistic order patterns
    """
    print("Generating default historical data based on real-world sample...")
    
    # Create date range for a sample month (8/7/25) - matches the provided data
    base_date = datetime(2025, 8, 7).date()
    
    # Generate orders in the format from the sample
    data = []
    
    # Create sample data directly matching the provided real-world format
    sample_data = [
        {'order_no': 1, 'distance': 6.0, 'time': '10:29 AM'},
        {'order_no': 2, 'distance': 7.5, 'time': '10:39 AM'},
        {'order_no': 3, 'distance': 5.7, 'time': '11:18 AM'},
        {'order_no': 4, 'distance': 5.2, 'time': '11:23 AM'},
        {'order_no': 5, 'distance': 10.0, 'time': '11:37 AM'},
        {'order_no': 6, 'distance': 5.8, 'time': '11:53 AM'},
        {'order_no': 7, 'distance': 12.1, 'time': '12:03 PM'},
        {'order_no': 8, 'distance': 6.8, 'time': '12:05 PM'},
        {'order_no': 9, 'distance': 2.5, 'time': '12:15 PM'},
        {'order_no': 10, 'distance': 8.4, 'time': '12:17 PM'},
        {'order_no': 11, 'distance': 7.0, 'time': '12:45 PM'},
        {'order_no': 12, 'distance': 9.4, 'time': '1:25 PM'},
        {'order_no': 13, 'distance': 2.5, 'time': '1:29 PM'},
        {'order_no': 14, 'distance': 7.2, 'time': '1:32 PM'},
        {'order_no': 15, 'distance': 2.5, 'time': '1:33 PM'},
        {'order_no': 16, 'distance': 9.4, 'time': '1:40 PM'},
        {'order_no': 17, 'distance': 9.4, 'time': '1:47 PM'},
        {'order_no': 18, 'distance': 7.2, 'time': '1:58 PM'},
        {'order_no': 19, 'distance': 7.2, 'time': '2:01 PM'},
        {'order_no': 20, 'distance': 5.2, 'time': '2:37 PM'},
        {'order_no': 21, 'distance': 5.2, 'time': '3:35 PM'},
        {'order_no': 22, 'distance': 6.0, 'time': '3:56 PM'},
        {'order_no': 23, 'distance': 6.0, 'time': '10:30 AM'},
        {'order_no': 24, 'distance': 4.0, 'time': '9:45 PM'},
        {'order_no': 25, 'distance': 5.0, 'time': '5:45 PM'},
    ]
    
    # Process the sample data to create a DataFrame
    for item in sample_data:
        # Parse the time string
        time_str = item['time']
        time_obj = datetime.strptime(time_str, '%I:%M %p').time()
        
        # Combine with the base date
        order_time = datetime.combine(base_date, time_obj)
        
        # Extract hour
        hour = time_obj.hour
        
        data.append({
            'Order No': item['order_no'],
            'Last Mile Distance From Branch': item['distance'],
            'Order Placed Date Time': order_time,
            'Order Date': base_date,
            'Order Day': base_date.strftime('%A'),
            'Order Hour': hour,
            'Is Weekend': base_date.weekday() >= 5,
            'Is Morning': hour < 12
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add additional days with similar patterns for better historical data
    additional_days = 6  # Add 6 more days
    order_no_offset = len(data)
    
    for day_offset in range(1, additional_days + 1):
        new_date = base_date - timedelta(days=day_offset)
        day_factor = 0.9 + (np.random.random() * 0.3)  # Vary by ±15%
        
        for i, item in enumerate(sample_data):
            # Add some randomness to time (±30 minutes)
            time_str = item['time']
            time_obj = datetime.strptime(time_str, '%I:%M %p')
            random_minutes = np.random.randint(-30, 31)
            new_time = (time_obj + timedelta(minutes=random_minutes)).time()
            
            # Vary the distance slightly
            dist_factor = 0.85 + (np.random.random() * 0.3)  # Vary by ±15%
            new_distance = round(item['distance'] * dist_factor, 1)
            
            # Combine with the new date
            order_time = datetime.combine(new_date, new_time)
            
            # Only include this order based on day factor (to vary order count by day)
            if np.random.random() < day_factor:
                data.append({
                    'Order No': order_no_offset + i + 1,
                    'Last Mile Distance From Branch': new_distance,
                    'Order Placed Date Time': order_time,
                    'Order Date': new_date,
                    'Order Day': new_date.strftime('%A'),
                    'Order Hour': new_time.hour,
                    'Is Weekend': new_date.weekday() >= 5,
                    'Is Morning': new_time.hour < 12
                })
                order_no_offset += 1
                
    # Create final DataFrame and sort by date/time
    final_df = pd.DataFrame(data)
    final_df.sort_values('Order Placed Date Time', inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    
    print(f"Generated default historical data with {len(final_df)} orders")
    return final_df

def load_historical_data(file_path):
    """
    Load historical data from Excel file with error handling
    
    Parameters:
    file_path (str): Path to the Excel file containing historical order data
    
    Returns:
    DataFrame: Preprocessed historical data
    """
    # Check if we're getting a default request
    if file_path is None or file_path == "default":
        print("Using default historical data patterns")
        return get_default_historical_data()
        
    print(f"Loading historical data from {file_path}...")
    
    try:
        # Get sheet names first
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        print(f"Available sheets in the Excel file: {sheet_names}")
        
        # Read the first sheet by default
        if len(sheet_names) >= 1:
            df = pd.read_excel(file_path, sheet_name=sheet_names[0])
            print(f"Loaded data from sheet: {sheet_names[0]}")
        else:
            raise ValueError("No sheets found in the Excel file")
            
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        # Try with different options
        try:
            print("Attempting to load with engine='openpyxl'...")
            df = pd.read_excel(file_path, engine='openpyxl')
            print("Successfully loaded with openpyxl engine")
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            raise
    
    print("\nPreprocessing historical data...")
    # Convert the distance column to numeric if it's not already
    if 'Last Mile Distance From Branch' in df.columns and df['Last Mile Distance From Branch'].dtype == 'object':
        # Extract numeric part and convert to float
        df['Last Mile Distance From Branch'] = df['Last Mile Distance From Branch'].str.extract(r'(\d+\.?\d*)')[0].astype(float)
        print("Distance column converted to numeric successfully")
        
    # Ensure datetime column is properly formatted
    if 'Order Placed Date Time' in df.columns and not pd.api.types.is_datetime64_ns_dtype(df['Order Placed Date Time']):
        df['Order Placed Date Time'] = pd.to_datetime(df['Order Placed Date Time'])
    
    # Sort orders by placement time
    df.sort_values('Order Placed Date Time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Extract additional time-based features
    df['Order Date'] = df['Order Placed Date Time'].dt.date
    df['Order Day'] = df['Order Placed Date Time'].dt.day_name()
    df['Order Hour'] = df['Order Placed Date Time'].dt.hour
    df['Is Weekend'] = df['Order Placed Date Time'].dt.dayofweek >= 5
    df['Is Morning'] = df['Order Placed Date Time'].dt.hour < 12
    
    print(f"Data preprocessing complete. Total historical orders: {len(df)}")
    return df

def analyze_order_patterns(historical_df, show_plots=True):
    """
    Analyze historical order patterns
    
    Parameters:
    historical_df (DataFrame): Preprocessed historical order data
    show_plots (bool): Whether to display visualizations
    
    Returns:
    dict: Order pattern analysis results
    """
    print("Analyzing historical order patterns...")
    
    # Initialize results dictionary
    results = {}
    
    # Count orders by day of week
    day_order_counts = historical_df.groupby('Order Day')['Order No'].count()
    results['orders_by_day'] = day_order_counts.to_dict()
    
    # Sort by actual day order (Monday to Sunday)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_order_counts = day_order_counts.reindex(day_names)
    
    # Order distribution by hour
    hour_order_counts = historical_df.groupby('Order Hour')['Order No'].count()
    results['orders_by_hour'] = hour_order_counts.to_dict()
    
    # Average distance by day
    avg_distance_by_day = historical_df.groupby('Order Day')['Last Mile Distance From Branch'].mean()
    results['avg_distance_by_day'] = avg_distance_by_day.to_dict()
    
    # Weekend vs. Weekday comparison
    weekend_orders = historical_df[historical_df['Is Weekend']].shape[0]
    weekday_orders = historical_df[~historical_df['Is Weekend']].shape[0]
    total_days_weekend = historical_df[historical_df['Is Weekend']]['Order Date'].nunique() 
    total_days_weekday = historical_df[~historical_df['Is Weekend']]['Order Date'].nunique()
    
    # Calculate daily average
    avg_orders_weekend = weekend_orders / total_days_weekend if total_days_weekend > 0 else 0
    avg_orders_weekday = weekday_orders / total_days_weekday if total_days_weekday > 0 else 0
    
    results['weekend_vs_weekday'] = {
        'weekend_orders': weekend_orders,
        'weekday_orders': weekday_orders,
        'avg_orders_weekend': avg_orders_weekend,
        'avg_orders_weekday': avg_orders_weekday
    }
    
    # Morning vs. Afternoon comparison
    morning_orders = historical_df[historical_df['Is Morning']].shape[0]
    afternoon_orders = historical_df[~historical_df['Is Morning']].shape[0]
    
    results['morning_vs_afternoon'] = {
        'morning_orders': morning_orders,
        'afternoon_orders': afternoon_orders,
        'morning_percentage': (morning_orders / len(historical_df)) * 100 if len(historical_df) > 0 else 0
    }
    
    # Visualize if requested
    if show_plots:
        # Plot orders by day of week
        plt.figure(figsize=(12, 6))
        sns.barplot(x=day_order_counts.index, y=day_order_counts.values)
        plt.title('Order Volume by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Orders')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Plot orders by hour
        plt.figure(figsize=(14, 6))
        sns.barplot(x=hour_order_counts.index, y=hour_order_counts.values)
        plt.title('Order Volume by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Orders')
        plt.xticks(range(24))
        plt.tight_layout()
        plt.show()
        
        # Plot average distance by day
        plt.figure(figsize=(12, 6))
        sns.barplot(x=avg_distance_by_day.reindex(day_names).index, y=avg_distance_by_day.reindex(day_names).values)
        plt.title('Average Delivery Distance by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Distance (km)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Weekend vs. Weekday comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Weekday', 'Weekend'], y=[avg_orders_weekday, avg_orders_weekend])
        plt.title('Average Daily Order Volume: Weekday vs. Weekend')
        plt.ylabel('Average Orders per Day')
        plt.tight_layout()
        plt.show()
        
        # Morning vs. Afternoon comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Morning', 'Afternoon'], y=[morning_orders, afternoon_orders])
        plt.title('Order Distribution: Morning vs. Afternoon')
        plt.ylabel('Number of Orders')
        plt.tight_layout()
        plt.show()
    
    print("Order pattern analysis complete.")
    return results

def generate_forecast(historical_df, forecast_date, seasonality_factor=1.0, growth_factor=1.0):
    """
    Generate forecasted orders for a specific date based on historical patterns
    
    Parameters:
    historical_df (DataFrame): Historical order data
    forecast_date (datetime.date): Date for which to generate forecast
    seasonality_factor (float): Adjust for seasonal variation (>1 for high season, <1 for low season)
    growth_factor (float): Adjust for business growth (>1 for growth, <1 for contraction)
    
    Returns:
    DataFrame: Forecasted orders for the specified date
    """
    print(f"Generating order forecast for {forecast_date}...")
    
    # Convert forecast_date to proper datetime.date if it's a string
    if isinstance(forecast_date, str):
        forecast_date = pd.to_datetime(forecast_date).date()
    
    # Get day of week for the forecast date
    forecast_day = forecast_date.strftime('%A')
    is_weekend = forecast_date.weekday() >= 5
    
    # Find similar days in historical data
    similar_days = historical_df[historical_df['Order Day'] == forecast_day]
    
    if similar_days.empty:
        print(f"No historical data found for {forecast_day}. Using overall average patterns.")
        similar_days = historical_df  # Use all historical data if no specific day data exists
    
    # Calculate average number of orders for this day of week
    avg_orders = len(similar_days) / similar_days['Order Date'].nunique() if similar_days['Order Date'].nunique() > 0 else 0
    
    # For default data, if no matching day is found, use the default pattern
    if avg_orders == 0 and 'orders_by_day' in DEFAULT_ORDER_PATTERNS:
        print(f"Using default order patterns for {forecast_day}")
        avg_orders = DEFAULT_ORDER_PATTERNS['orders_by_day'].get(forecast_day, DEFAULT_ORDER_PATTERNS['avg_daily_orders'])
    
    # Apply seasonality and growth factors
    forecasted_order_count = int(avg_orders * seasonality_factor * growth_factor)
    
    print(f"Forecasted order count for {forecast_date} ({forecast_day}): {forecasted_order_count}")
    
    # If no orders forecasted, return empty DataFrame with right structure
    if forecasted_order_count == 0:
        columns = historical_df.columns
        return pd.DataFrame(columns=columns)
    
    # Create forecasted orders based on historical patterns
    try:
        # 1. Sample from historical data for this day of week
        if len(similar_days) > forecasted_order_count:
            forecast_sample = similar_days.sample(forecasted_order_count, replace=False)
        else:
            # If we don't have enough historical orders, sample with replacement
            forecast_sample = similar_days.sample(forecasted_order_count, replace=True)
        
        # 2. Create a new DataFrame with sampled orders but adjusted timestamps
        forecasted_orders = forecast_sample.copy()
        
        # 3. Adjust the order dates to the forecast date while keeping the time component
        for idx in forecasted_orders.index:
            try:
                original_dt = forecasted_orders.loc[idx, 'Order Placed Date Time']
                if isinstance(original_dt, pd.Series):
                    # Handle case when original_dt is a Series
                    original_dt = original_dt.iloc[0]
                new_dt = datetime.combine(forecast_date, original_dt.time())
                forecasted_orders.loc[idx, 'Order Placed Date Time'] = new_dt
                forecasted_orders.loc[idx, 'Order Date'] = forecast_date
            except Exception as e:
                print(f"Warning: Error processing date for order {idx}: {e}")
                # Generate a new random time for this order
                hour = np.random.choice(range(24), p=[x/sum(DEFAULT_ORDER_PATTERNS['orders_by_hour'].values()) 
                                                    for x in DEFAULT_ORDER_PATTERNS['orders_by_hour'].values()])
                minute = np.random.randint(0, 60)
                new_dt = datetime.combine(forecast_date, datetime.min.time().replace(hour=hour, minute=minute))
                forecasted_orders.loc[idx, 'Order Placed Date Time'] = new_dt
                forecasted_orders.loc[idx, 'Order Date'] = forecast_date
    except Exception as e:
        print(f"Error in forecast generation: {e}. Generating synthetic orders instead.")
        
        # If sampling fails, create completely new synthetic orders
        data = []
        
        # Distribution of orders by hour
        hourly_weights = [DEFAULT_ORDER_PATTERNS['orders_by_hour'].get(h, 0) for h in range(24)]
        total_weight = sum(hourly_weights)
        hourly_probs = [w/total_weight for w in hourly_weights]
        
        # Generate orders
        for i in range(forecasted_order_count):
            # Random hour based on distribution
            hour = np.random.choice(range(24), p=hourly_probs)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            order_time = datetime.combine(forecast_date, 
                                          datetime.min.time().replace(hour=hour, minute=minute, second=second))
            
            # Random distance based on day average
            avg_dist = DEFAULT_ORDER_PATTERNS['avg_distance_by_day'].get(forecast_day, 
                                                                         DEFAULT_ORDER_PATTERNS['avg_distance'])
            distance = max(0.5, np.random.normal(avg_dist, 1.0))
            
            data.append({
                'Order No': f"SYN{i+1:05d}",
                'Order Placed Date Time': order_time,
                'Last Mile Distance From Branch': distance,
                'Order Date': forecast_date,
                'Order Day': forecast_day,
                'Order Hour': hour,
                'Is Weekend': is_weekend,
                'Is Morning': hour < 12
            })
        
        # Create DataFrame
        forecasted_orders = pd.DataFrame(data)
    
    # 4. Reset the order numbers to be sequential if not already done
    if 'Order No' in forecasted_orders.columns:
        forecasted_orders['Order No'] = [f"F{i+1:05d}" for i in range(len(forecasted_orders))]
    
    # 5. Reset index and return
    forecasted_orders.reset_index(drop=True, inplace=True)
    
    # 6. Ensure all required columns are present
    required_columns = ['Order No', 'Order Placed Date Time', 'Last Mile Distance From Branch', 
                       'Order Date', 'Order Day', 'Order Hour', 'Is Weekend', 'Is Morning']
    
    for col in required_columns:
        if col not in forecasted_orders.columns:
            if col == 'Order No':
                forecasted_orders[col] = [f"F{i+1:05d}" for i in range(len(forecasted_orders))]
            elif col == 'Order Placed Date Time':
                # Should never happen but just in case
                hour_distribution = [i for i, count in DEFAULT_ORDER_PATTERNS['orders_by_hour'].items() for _ in range(count)]
                forecasted_orders[col] = [datetime.combine(forecast_date, 
                                                         datetime.min.time().replace(hour=np.random.choice(hour_distribution))) 
                                        for _ in range(len(forecasted_orders))]
            elif col == 'Last Mile Distance From Branch':
                avg_dist = DEFAULT_ORDER_PATTERNS['avg_distance']
                forecasted_orders[col] = [max(0.5, np.random.normal(avg_dist, 1.0)) for _ in range(len(forecasted_orders))]
            elif col == 'Order Date':
                forecasted_orders[col] = forecast_date
            elif col == 'Order Day':
                forecasted_orders[col] = forecast_day
            elif col == 'Order Hour':
                forecasted_orders[col] = forecasted_orders['Order Placed Date Time'].dt.hour
            elif col == 'Is Weekend':
                forecasted_orders[col] = is_weekend
            elif col == 'Is Morning':
                forecasted_orders[col] = forecasted_orders['Order Hour'] < 12
    
    print(f"Generated {len(forecasted_orders)} forecasted orders")
    return forecasted_orders

def recommend_resources(historical_df, forecasted_orders, target_sla_percentage=90):
    """
    Recommend optimal resource allocation based on historical performance and forecasted orders
    
    Parameters:
    historical_df (DataFrame): Historical order data
    forecasted_orders (DataFrame): Forecasted orders for the target date
    target_sla_percentage (float): Target SLA percentage to achieve
    
    Returns:
    dict: Recommended resource allocation and configuration
    """
    print("Generating resource recommendations...")
    
    # Ensure forecasted_orders is not empty
    if forecasted_orders is None or forecasted_orders.empty:
        print("No forecasted orders provided, recommending minimal staffing")
        return {
            'recommended_pickers': 1,
            'recommended_bikers': 1,
            'picking_time_mins': 15,
            'enable_batching': False,
            'batch_size': 2,
            'batching_num_bikers': 0,
            'recommended_strategy': 'FCFS',
            'forecasted_orders': 0,
            'morning_orders': 0,
            'afternoon_orders': 0,
            'message': "Minimal staffing recommended as no orders are forecasted."
        }
    
    # Count forecasted orders
    forecasted_count = len(forecasted_orders)
    
    # If no orders are forecasted, recommend minimal staffing
    if forecasted_count == 0:
        return {
            'recommended_pickers': 1,
            'recommended_bikers': 1,
            'picking_time_mins': 15,
            'enable_batching': False,
            'batch_size': 2,
            'batching_num_bikers': 0,
            'recommended_strategy': 'FCFS',
            'forecasted_orders': 0,
            'morning_orders': 0,
            'afternoon_orders': 0,
            'message': "Minimal staffing recommended as no orders are forecasted."
        }
    
    # Ensure 'Is Morning' column exists
    if 'Is Morning' not in forecasted_orders.columns:
        # Create the column based on hour of day
        if 'Order Placed Date Time' in forecasted_orders.columns:
            forecasted_orders = forecasted_orders.copy()
            forecasted_orders['Is Morning'] = forecasted_orders['Order Placed Date Time'].dt.hour < 12
        else:
            # If Order Placed Date Time doesn't exist, assume even distribution
            forecasted_orders = forecasted_orders.copy()
            forecasted_orders['Is Morning'] = [i % 2 == 0 for i in range(len(forecasted_orders))]
    
    # Analyze order timing patterns
    morning_orders = forecasted_orders[forecasted_orders['Is Morning']].shape[0]
    afternoon_orders = forecasted_orders[~forecasted_orders['Is Morning']].shape[0]
    
    # Calculate resource needs based on historical performance
    # Assuming each picker can handle approximately 3-4 orders per hour (15-20 min per order)
    picking_time_mins = 15  # Default picking time
    
    # Calculate pickers needed (assuming 8-hour shift with 30 min lunch)
    effective_picker_hours = 7.5  # 8 hours - 30 min lunch
    orders_per_picker_per_day = (60 / picking_time_mins) * effective_picker_hours
    
    # Calculate base picker needs and add 20% buffer for peak times
    base_pickers_needed = forecasted_count / orders_per_picker_per_day
    recommended_pickers = max(1, int(np.ceil(base_pickers_needed * 1.2)))
    
    # Calculate bikers needed based on order distribution and distances
    avg_delivery_time_mins = 20  # Assume average delivery cycle is 20 minutes
    
    # Account for morning order concentration if applicable
    enable_batching = False
    batch_size = 2
    batching_num_bikers = 0
    
    # If there are significant morning orders, recommend batching
    if morning_orders > 5:
        enable_batching = True
        
        # For larger morning order volumes, increase batch size
        if morning_orders > 10:
            batch_size = 3
        
        # For very high morning volumes, dedicate specific bikers
        if morning_orders > 15:
            batching_num_bikers = max(1, int(morning_orders / 10))
    
    # Calculate delivery capacity per biker (accounting for batching efficiency)
    if enable_batching:
        morning_effective_capacity = 60 / (avg_delivery_time_mins * 0.75) * batch_size  # 25% time saving with batching
        afternoon_capacity = 60 / avg_delivery_time_mins
    else:
        morning_effective_capacity = 60 / avg_delivery_time_mins
        afternoon_capacity = 60 / avg_delivery_time_mins
    
    # Effective hours for morning (10am-1pm) and afternoon (3pm-6pm)
    morning_hours = 3
    afternoon_hours = 3
    
    # Bikers needed for each shift (account for lunch hour)
    morning_bikers = morning_orders / (morning_effective_capacity * morning_hours)
    afternoon_bikers = afternoon_orders / (afternoon_capacity * afternoon_hours)
    
    # Take the maximum as the recommended number (add 20% buffer)
    recommended_bikers = max(1, int(np.ceil(max(morning_bikers, afternoon_bikers) * 1.2)))
    
    # Additional adjustments based on order density
    if forecasted_count > 20:
        # For high order volumes, ensure adequate staffing
        recommended_pickers = max(recommended_pickers, 3)
        recommended_bikers = max(recommended_bikers, 3)
    
    recommendation = {
        'recommended_pickers': recommended_pickers,
        'recommended_bikers': recommended_bikers,
        'picking_time_mins': picking_time_mins,
        'enable_batching': enable_batching,
        'batch_size': batch_size,
        'batching_num_bikers': batching_num_bikers,
        'forecasted_orders': forecasted_count,
        'morning_orders': morning_orders,
        'afternoon_orders': afternoon_orders,
        'message': f"Based on the forecast of {forecasted_count} orders ({morning_orders} morning, {afternoon_orders} afternoon)."
    }
    
    # Add scheduling strategy recommendation
    if enable_batching and morning_orders > 0.7 * forecasted_count:
        # If majority of orders are in the morning, prioritize SLA
        recommendation['recommended_strategy'] = "MAXIMIZE_SLA"
        recommendation['message'] += " High morning order concentration suggests prioritizing SLA."
    elif forecasted_count > 15:
        # For high order volumes, maximize throughput
        recommendation['recommended_strategy'] = "MAXIMIZE_ORDERS"
        recommendation['message'] += " High order volume suggests maximizing order throughput."
    else:
        # Default to FCFS for low volumes
        recommendation['recommended_strategy'] = "FCFS"
        recommendation['message'] += " Standard order volume suggests FCFS approach."
    
    print(f"Resource recommendation: {recommendation['recommended_pickers']} pickers, {recommendation['recommended_bikers']} bikers")
    print(f"Strategy recommendation: {recommendation['recommended_strategy']}")
    if recommendation['enable_batching']:
        print(f"Batching recommended: {recommendation['batch_size']} orders per batch")
    
    return recommendation

def visualize_forecast(historical_df, forecasted_orders, forecast_date):
    """
    Visualize forecast comparison with historical data
    
    Parameters:
    historical_df (DataFrame): Historical order data
    forecasted_orders (DataFrame): Forecasted orders
    forecast_date (datetime.date): Date of forecast
    
    Returns:
    None (displays visualizations)
    """
    if forecasted_orders.empty:
        print("No forecasted orders to visualize.")
        return
        
    # Convert forecast_date to proper datetime.date if it's a string
    if isinstance(forecast_date, str):
        forecast_date = pd.to_datetime(forecast_date).date()
    
    # Get day of week for the forecast date
    forecast_day = forecast_date.strftime('%A')
    
    # Find similar historical days
    similar_days = historical_df[historical_df['Order Day'] == forecast_day]
    
    if similar_days.empty:
        print(f"No historical data for {forecast_day} to compare with.")
        return
    
    # Aggregate historical orders by hour for comparison
    historical_hourly = similar_days.groupby('Order Hour')['Order No'].count()
    
    # Normalize historical data (average per day)
    days_count = similar_days['Order Date'].nunique()
    historical_hourly = historical_hourly / days_count if days_count > 0 else historical_hourly
    
    # Aggregate forecasted orders by hour
    forecast_hourly = forecasted_orders.groupby(forecasted_orders['Order Placed Date Time'].dt.hour)['Order No'].count()
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    # Plot historical average
    plt.bar(historical_hourly.index, historical_hourly.values, alpha=0.7, label=f'Historical Avg ({forecast_day}s)', color='steelblue')
    
    # Plot forecast
    plt.bar(forecast_hourly.index, forecast_hourly.values, alpha=0.7, label=f'Forecast ({forecast_date})', color='coral')
    
    plt.title(f'Order Forecast Comparison for {forecast_date} ({forecast_day})')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Orders')
    plt.legend()
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # If Plotly is available, create interactive visualization
    if plotly_available:
        # Prepare data
        historical_df_plot = pd.DataFrame({
            'Hour': historical_hourly.index,
            'Orders': historical_hourly.values,
            'Type': f'Historical Avg ({forecast_day}s)'
        })
        
        forecast_df_plot = pd.DataFrame({
            'Hour': forecast_hourly.index,
            'Orders': forecast_hourly.values,
            'Type': f'Forecast ({forecast_date})'
        })
        
        combined_df = pd.concat([historical_df_plot, forecast_df_plot])
        
        # Create interactive bar chart
        fig = px.bar(combined_df, x='Hour', y='Orders', color='Type', barmode='group',
                    title=f'Order Forecast Comparison for {forecast_date} ({forecast_day})',
                    labels={'Orders': 'Number of Orders', 'Hour': 'Hour of Day'},
                    color_discrete_sequence=['steelblue', 'coral'])
        
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                        legend_title_text='')
        fig.show()

def save_forecast_to_excel(forecasted_orders, file_path):
    """
    Save forecasted orders to Excel file
    
    Parameters:
    forecasted_orders (DataFrame): Forecasted orders
    file_path (str): Path to save the Excel file
    
    Returns:
    bool: Whether the save was successful
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save to Excel
        forecasted_orders.to_excel(file_path, index=False)
        print(f"Forecast successfully saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving forecast to Excel: {e}")
        return False

if __name__ == "__main__":
    # Example usage when run as a script
    print("Historical Data Analysis and Forecasting Utility")
    print("================================================")
    
    # Get historical data file
    default_file = 'PARTNR Nagpur.xlsx'
    if os.path.exists(default_file):
        file_path = input(f"Enter historical data file path (default: {default_file}): ") or default_file
    else:
        file_path = input("Enter historical data file path: ")
    
    # Load historical data
    historical_data = load_historical_data(file_path)
    
    # Analyze patterns
    analyze_patterns = input("Analyze historical order patterns? (y/n, default: y): ").lower() != 'n'
    if analyze_patterns:
        patterns = analyze_order_patterns(historical_data)
    
    # Generate forecast
    forecast_default_date = (datetime.now() + timedelta(days=1)).date()
    forecast_date_str = input(f"Enter forecast date (YYYY-MM-DD, default: {forecast_default_date}): ") or str(forecast_default_date)
    forecast_date = pd.to_datetime(forecast_date_str).date()
    
    seasonality = float(input("Enter seasonality factor (>1 for high season, <1 for low season, default: 1.0): ") or "1.0")
    growth = float(input("Enter growth factor (>1 for growth, <1 for contraction, default: 1.0): ") or "1.0")
    
    # Generate the forecast
    forecasted_orders = generate_forecast(historical_data, forecast_date, seasonality, growth)
    
    # Visualize forecast
    visualize = input("Visualize forecast? (y/n, default: y): ").lower() != 'n'
    if visualize:
        visualize_forecast(historical_data, forecasted_orders, forecast_date)
    
    # Get resource recommendations
    target_sla = float(input("Enter target SLA percentage (default: 90): ") or "90")
    recommendations = recommend_resources(historical_data, forecasted_orders, target_sla)
    
    print("\nResource Recommendations:")
    print(f"- Pickers: {recommendations['recommended_pickers']}")
    print(f"- Bikers: {recommendations['recommended_bikers']}")
    print(f"- Scheduling Strategy: {recommendations['recommended_strategy']}")
    print(f"- Enable Batching: {recommendations['enable_batching']}")
    if recommendations['enable_batching']:
        print(f"- Batch Size: {recommendations['batch_size']}")
        print(f"- Dedicated Batching Bikers: {recommendations['batching_num_bikers']}")
    print(f"- Note: {recommendations['message']}")
    
    # Save forecast to Excel
    save_to_excel = input("Save forecast to Excel? (y/n, default: y): ").lower() != 'n'
    if save_to_excel:
        default_output = f"forecast_{forecast_date}.xlsx"
        output_path = input(f"Enter output file path (default: {default_output}): ") or default_output
        save_forecast_to_excel(forecasted_orders, output_path)
