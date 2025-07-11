#!/usr/bin/env python3
# Order Processing Simulation and Prediction for Dark Store
# Processes orders for August 7th, 2025
#
# This simulation supports three different scheduling strategies:
# 1. FCFS (First Come First Served) - Orders are processed in the order they were received
# 2. MAXIMIZE_ORDERS - Prioritizes shorter delivery trips to maximize the total number of orders delivered
# 3. MAXIMIZE_SLA - Prioritizes orders that are at risk of breaching SLA, based on remaining time until breach
#
# IMPORTANT: All strategies ensure that resources (pickers and bikers) are NEVER left idle when orders are available.
# Resources are always utilized when possible, regardless of prioritization strategy.
# The scheduling strategy is ONLY applied when there are MULTIPLE orders to choose from and a resource needs
# to decide which order to process first. When there's only one order available, it's processed immediately.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import datetime
from pathlib import Path
import warnings

# Import historical data analysis functions
try:
    from historical_data import (
        load_historical_data,
        analyze_order_patterns,
        generate_forecast,
        recommend_resources,
        visualize_forecast,
        save_forecast_to_excel
    )
    historical_data_available = True
except ImportError:
    historical_data_available = False
    print("Warning: historical_data module not available. Predictive features will be limited.")

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# For interactive plots (if available)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Check for nbformat which is required for Plotly to show plots
    try:
        import nbformat
        # Test if we can render a simple plot
        test_fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
        # Just create it, don't show it yet
        plotly_available = True
    except (ImportError, ValueError):
        print("Warning: Plotly found but nbformat is missing. Install with: pip install nbformat>=4.2.0")
        print("Using matplotlib for visualizations.")
        plotly_available = False
except ImportError:
    plotly_available = False
    print("Plotly not available. Using matplotlib for visualizations.")


def preprocess_data(file_path):
    """
    Load and preprocess the Excel data for simulation
    
    Parameters:
    file_path (str): Path to the Excel file
    
    Returns:
    DataFrame: Processed data ready for simulation
    """
    print("Loading data from Excel file...")
    
    # Load the Excel file
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
    
    print("\nPreprocessing data...")
    # Convert the distance column to numeric
    if df['Last Mile Distance From Branch'].dtype == 'object':
        # Extract numeric part and convert to float
        df['Last Mile Distance From Branch'] = df['Last Mile Distance From Branch'].str.extract(r'(\d+\.?\d*)')[0].astype(float)
        print("Distance column converted to numeric successfully")
        
    # Ensure datetime column is properly formatted
    if not pd.api.types.is_datetime64_ns_dtype(df['Order Placed Date Time']):
        df['Order Placed Date Time'] = pd.to_datetime(df['Order Placed Date Time'])
    
    # Sort orders by placement time
    df.sort_values('Order Placed Date Time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Data preprocessing complete. Total orders: {len(df)}")
    return df


def filter_current_day_orders(orders_df, processing_date):
    """
    Filter orders for the current processing date only
    
    Parameters:
    orders_df (DataFrame): DataFrame containing the order data
    processing_date (datetime.date): Date for which orders should be processed
    
    Returns:
    DataFrame: Filtered orders for the processing date
    """
    print(f"Filtering orders for {processing_date}...")
    
    # Filter only orders from the processing date
    filtered_df = orders_df[orders_df['Order Placed Date Time'].dt.date == processing_date].copy()
    
    print(f"Found {len(filtered_df)} orders for {processing_date}")
    return filtered_df


def simulate_order_processing(orders_df, num_pickers, num_bikers, processing_date, picking_time_mins=15, 
                      scheduling_strategy="FCFS", enable_batching=False, batch_size=2,
                      batching_num_bikers=0, verbose=False):
    """
    Simulate order processing based on given constraints
    
    Parameters:
    orders_df (DataFrame): DataFrame containing order data
    num_pickers (int): Number of pickers available
    num_bikers (int): Number of bikers available
    processing_date (datetime.date): The date orders are being processed
    picking_time_mins (int): Time taken for picking/packing an order in minutes
    scheduling_strategy (str): Strategy for order scheduling:
                              "FCFS" - First Come First Served (default)
                              "MAXIMIZE_ORDERS" - Maximize total number of orders delivered
                              "MAXIMIZE_SLA" - Prioritize orders to maximize SLA compliance
    enable_batching (bool): Whether to enable batching of morning orders
    batch_size (int): Maximum number of orders that can be batched together
    batching_num_bikers (int): Number of bikers assigned for batched delivery (0 means all bikers can do batching)
    verbose (bool): Whether to print detailed logs
    
    Returns:
    dict: Results of simulation including order statuses and biker schedules
    """
    if verbose:
        print(f"\nSimulating order processing for {processing_date} with {num_pickers} pickers and {num_bikers} bikers")
        print(f"Using scheduling strategy: {scheduling_strategy}")
    
    # Make a working copy of the dataframe
    df = orders_df.copy()
    
    # Initialize status columns
    df['Order Status'] = 'Undelivered'
    df['Picking Start Time'] = pd.NaT
    df['Picking End Time'] = pd.NaT
    df['Delivery Start Time'] = pd.NaT
    df['Delivery End Time'] = pd.NaT
    df['Assigned Picker'] = None
    df['Assigned Biker'] = None
    df['SLA Met'] = False
    df['Processing Date'] = processing_date
    
    # Define time cutoffs
    store_open_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(9, 0)))
    picker_start_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(9, 30)))
    biker_start_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(10, 0)))
    early_order_cutoff_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(10, 0)))
    early_order_sla_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(12, 0)))
    lunch_start_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(13, 0)))
    lunch_end_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(15, 0)))
    end_of_day = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(18, 0)))
    
    if verbose:
        print(f"Store opens: {store_open_time}")
        print(f"Pickers start at: {picker_start_time}")
        print(f"Bikers start at: {biker_start_time}")
        print(f"Early order cutoff (10 AM): {early_order_cutoff_time}")
        print(f"Early order SLA (12 PM): {early_order_sla_time}")
        print(f"Lunch break window: {lunch_start_time} to {lunch_end_time}")
        print(f"End of day: {end_of_day}")
    
    # Create picker and biker availability tracking
    pickers = [picker_start_time] * num_pickers  # All pickers start at 9:30 AM
    bikers = [biker_start_time] * num_bikers    # All bikers start at 10 AM
    
    # Lunch break tracking
    picker_lunch_taken = [False] * num_pickers
    biker_lunch_taken = [False] * num_bikers
    
    # Store detailed schedule of bikers
    biker_schedules = {i: [] for i in range(num_bikers)}
    
    # Batching configuration
    if enable_batching:
        if verbose:
            print(f"Order batching enabled. Batch size: {batch_size}")
            if batching_num_bikers > 0:
                print(f"Assigned {batching_num_bikers} bikers for batched deliveries (biker IDs 0-{batching_num_bikers-1})")
            else:
                print(f"All bikers can handle batched deliveries")
        
        # Track bikers that can handle batched orders
        if batching_num_bikers > 0:
            batching_bikers = list(range(min(batching_num_bikers, num_bikers)))
        else:
            batching_bikers = list(range(num_bikers))
            
        # Orders waiting to be batched (for morning orders)
        batched_orders = {biker_id: [] for biker_id in batching_bikers}
        # Current batch count for each biker
        current_batch_counts = {biker_id: 0 for biker_id in batching_bikers}
    else:
        batching_bikers = []
        batched_orders = {}
    
    # All orders are from the current processing date
    df['Order Date'] = df['Order Placed Date Time'].dt.date
    df['Is Previous Day'] = False  # No previous day orders
    
    # Initialize order processing time
    for idx in df.index:
        df.at[idx, 'Order Placed Time For Processing'] = df.at[idx, 'Order Placed Date Time']
        # For orders placed before store opening, set their processing time to store opening time
        if df.at[idx, 'Order Placed Date Time'] < store_open_time:
            df.at[idx, 'Order Placed Time For Processing'] = store_open_time
    
    # Organize orders into two lists:
    # 1. Orders that need picking (haven't started picking yet)
    # 2. Orders that need delivery (picking completed but not yet delivered)
    orders_to_pick = df.copy()
    orders_to_deliver = pd.DataFrame(columns=df.columns)  # Empty DataFrame with the same structure
    
    # Calculate SLA deadline for each order
    orders_to_pick['SLA_Deadline'] = orders_to_pick.apply(
        lambda order: early_order_sla_time if order['Order Placed Date Time'] < early_order_cutoff_time
        else order['Order Placed Date Time'] + pd.Timedelta(hours=2), axis=1
    )
    
    # Calculate estimated delivery time (based on distance)
    orders_to_pick['Est_Delivery_Duration'] = orders_to_pick.apply(
        lambda order: pd.Timedelta(minutes=int(4 * order['Last Mile Distance From Branch'])) * 2 + pd.Timedelta(minutes=5),
        axis=1
    )
    
    # Sort orders initially based on FCFS
    orders_to_pick = orders_to_pick.sort_values('Order Placed Time For Processing')
    
    # Current simulation time
    current_time = store_open_time
    
    # Main simulation loop
    while True:
        # Check if we're done (no more orders to process and all resources are idle)
        if orders_to_pick.empty and orders_to_deliver.empty:
            if verbose:
                print(f"All orders processed. Simulation ended at {current_time}")
            break
            
        # Check if we've reached end of day
        if current_time >= end_of_day:
            if verbose:
                print(f"Reached end of day ({end_of_day}). Stopping simulation.")
            break
            
        # Identify resources available at the current time
        available_pickers = [i for i, time in enumerate(pickers) if time <= current_time]
        available_bikers = [i for i, time in enumerate(bikers) if time <= current_time]
        
        # Process lunch breaks for available resources
        for i in available_pickers[:]:  # Use a copy of the list since we might modify it
            if not picker_lunch_taken[i] and lunch_start_time <= current_time < lunch_end_time:
                # Take lunch break
                pickers[i] = current_time + pd.Timedelta(minutes=30)
                picker_lunch_taken[i] = True
                available_pickers.remove(i)  # Remove from available pickers
                if verbose:
                    print(f"Picker {i} taking lunch break from {current_time} to {pickers[i]}")
                    
        for i in available_bikers[:]:  # Use a copy of the list since we might modify it
            if not biker_lunch_taken[i] and lunch_start_time <= current_time < lunch_end_time:
                # Take lunch break
                bikers[i] = current_time + pd.Timedelta(minutes=30)
                biker_lunch_taken[i] = True
                available_bikers.remove(i)  # Remove from available bikers
                if verbose:
                    print(f"Biker {i} taking lunch break from {current_time} to {bikers[i]}")
        
        # Process orders waiting for delivery (if bikers are available)
        if not orders_to_deliver.empty and available_bikers:
            # Filter orders that are ready for delivery (picking has finished)
            ready_for_delivery = orders_to_deliver[orders_to_deliver['Picking End Time'] <= current_time].copy()
            
            if not ready_for_delivery.empty:
                # Identify morning orders for potential batching
                if enable_batching:
                    morning_orders = ready_for_delivery[ready_for_delivery['Order Placed Date Time'] < early_order_cutoff_time].copy()
                    non_morning_orders = ready_for_delivery[ready_for_delivery['Order Placed Date Time'] >= early_order_cutoff_time].copy()
                    
                    # Process morning orders for batching if any are available
                    if not morning_orders.empty:
                        # Check available batching bikers
                        available_batching_bikers = [b for b in batching_bikers if b in available_bikers]
                        
                        if available_batching_bikers:
                            if verbose:
                                print(f"Processing {len(morning_orders)} morning orders for potential batching")
                            
                            # Prioritize morning orders just like regular orders
                            if len(morning_orders) > 1:
                                if scheduling_strategy == "FCFS":
                                    morning_orders = morning_orders.sort_values('Picking End Time')
                                elif scheduling_strategy == "MAXIMIZE_ORDERS":
                                    morning_orders = morning_orders.sort_values('Est_Delivery_Duration')
                                elif scheduling_strategy == "MAXIMIZE_SLA":
                                    morning_orders['Time_To_SLA_Breach'] = morning_orders['SLA_Deadline'] - current_time
                                    morning_orders['Can_Meet_SLA'] = morning_orders.apply(
                                        lambda x: current_time + x['Est_Delivery_Duration'] <= x['SLA_Deadline'], axis=1
                                    )
                                    morning_orders = morning_orders.sort_values(
                                        by=['Can_Meet_SLA', 'Time_To_SLA_Breach', 'Est_Delivery_Duration'],
                                        ascending=[False, True, True]
                                    )
                            
                            # Add orders to batches for available batching bikers
                            for order_idx in morning_orders.index:
                                order = morning_orders.loc[order_idx]
                                
                                # Find the batching biker with the fewest orders in their current batch
                                biker_id = min(
                                    available_batching_bikers, 
                                    key=lambda b: current_batch_counts[b] if b in current_batch_counts else 0
                                )
                                
                                # If this biker's batch is full, skip unless they have no orders yet
                                if biker_id in current_batch_counts and current_batch_counts[biker_id] >= batch_size and len(batched_orders[biker_id]) > 0:
                                    continue
                                    
                                # Add the order to this biker's batch
                                batched_orders[biker_id].append((order_idx, order))
                                current_batch_counts[biker_id] = len(batched_orders[biker_id])
                                
                                if verbose:
                                    print(f"Added morning Order {order['Order No']} to biker {biker_id}'s batch (now: {current_batch_counts[biker_id]}/{batch_size})")
                                
                                # Remove from ready for delivery list to avoid double processing
                                orders_to_deliver = orders_to_deliver.drop(order_idx)
                            
                            # Remove processed morning orders from the ready list
                            ready_for_delivery = non_morning_orders
                            
                            # Process batches that are ready (full or no more morning orders)
                            for biker_id in available_batching_bikers[:]:
                                if biker_id in batched_orders and batched_orders[biker_id]:
                                    # If batch is full or there are no more morning orders to add
                                    if len(batched_orders[biker_id]) >= batch_size or len(batched_orders[biker_id]) > 0 and morning_orders.empty:
                                        if verbose:
                                            print(f"Processing batch for biker {biker_id} with {len(batched_orders[biker_id])} orders")
                                            
                                        # Calculate total distance for this batch (assume orders are delivered in sequence)
                                        total_distance = sum(order['Last Mile Distance From Branch'] for _, order in batched_orders[biker_id])
                                        
                                        # Calculate batch delivery time (assume 50% efficiency gain from batching)
                                        # For batched orders, we assume the biker travels to all delivery points before returning
                                        # This optimizes the route compared to returning to store after each delivery
                                        efficiency_factor = 0.75  # 25% time saving from not returning after each delivery
                                        one_way_time = pd.Timedelta(minutes=int(4 * total_distance * efficiency_factor))
                                        customer_wait_total = pd.Timedelta(minutes=5 * len(batched_orders[biker_id]))  # 5 min per customer
                                        return_journey = pd.Timedelta(minutes=int(4 * total_distance / len(batched_orders[biker_id])))  # Only count one return trip
                                        
                                        total_batch_time = one_way_time + customer_wait_total + return_journey
                                        
                                        # Find the latest picking end time among batched orders
                                        latest_picking_end = max(order['Picking End Time'] for _, order in batched_orders[biker_id])
                                        
                                        # Delivery times
                                        delivery_start = max(latest_picking_end, current_time)
                                        delivery_end = delivery_start + total_batch_time
                                        
                                        # Calculate individual customer handover times (estimate spread throughout the journey)
                                        journey_interval = one_way_time / len(batched_orders[biker_id])
                                        
                                        # Check if delivery would end after 6 PM
                                        if delivery_end <= end_of_day:
                                            # Process each order in the batch
                                            batch_order_ids = []
                                            batch_distances = []
                                            
                                            for i, (order_idx, order) in enumerate(batched_orders[biker_id]):
                                                # Estimate this customer's handover time (spread throughout the journey)
                                                customer_handover = delivery_start + journey_interval * (i + 1)
                                                
                                                # Update the order data
                                                df.loc[order_idx, 'Assigned Biker'] = biker_id
                                                df.loc[order_idx, 'Delivery Start Time'] = delivery_start
                                                df.loc[order_idx, 'Customer Handover Time'] = customer_handover
                                                df.loc[order_idx, 'Delivery End Time'] = delivery_end
                                                df.loc[order_idx, 'Order Status'] = 'Delivered'
                                                df.loc[order_idx, 'Batch ID'] = f"Batch-{biker_id}-{delivery_start.strftime('%H%M')}"
                                                df.loc[order_idx, 'Is Batched'] = True
                                                
                                                # Check SLA
                                                original_order_time = order['Order Placed Date Time']
                                                if original_order_time < early_order_cutoff_time:
                                                    sla_time = early_order_sla_time
                                                else:
                                                    sla_time = original_order_time + pd.Timedelta(hours=2)
                                                    
                                                if customer_handover <= sla_time:
                                                    df.loc[order_idx, 'SLA Met'] = True
                                                    if verbose:
                                                        print(f"Batched Order {order['Order No']} delivered by biker {biker_id}. SLA MET!")
                                                else:
                                                    if verbose:
                                                        print(f"Batched Order {order['Order No']} delivered by biker {biker_id}. SLA MISSED!")
                                                
                                                batch_order_ids.append(str(order['Order No']))
                                                batch_distances.append(order['Last Mile Distance From Branch'])
                                            
                                            # Add to biker schedule as a single batched activity
                                            biker_schedules[biker_id].append({
                                                'Order No': ','.join(batch_order_ids),
                                                'Start': delivery_start,
                                                'End': delivery_end,
                                                'Activity': f"Batch Delivery: {len(batch_order_ids)} orders ({sum(batch_distances):.1f} km)",
                                                'Distance': sum(batch_distances),
                                                'Is Batched': True,
                                                'Batch Size': len(batch_order_ids)
                                            })
                                            
                                            # Update biker availability and remove from available list
                                            bikers[biker_id] = delivery_end
                                            available_bikers.remove(biker_id)
                                            
                                        else:
                                            if verbose:
                                                print(f"Batched orders for biker {biker_id} cannot be delivered before end of day. Skipping.")
                                                
                                        # Clear this biker's batch
                                        batched_orders[biker_id] = []
                                        current_batch_counts[biker_id] = 0
                
                # Apply prioritization strategy for regular delivery ONLY when multiple orders are available
                if not ready_for_delivery.empty and len(ready_for_delivery) > 1:
                    if scheduling_strategy == "FCFS":
                        # Sort by picking end time (first come first served for delivery)
                        ready_for_delivery = ready_for_delivery.sort_values('Picking End Time')
                    elif scheduling_strategy == "MAXIMIZE_ORDERS":
                        # Sort by estimated delivery duration (shortest first)
                        ready_for_delivery = ready_for_delivery.sort_values('Est_Delivery_Duration')
                    elif scheduling_strategy == "MAXIMIZE_SLA":
                        # Sort by urgency based on SLA deadline
                        ready_for_delivery['Time_To_SLA_Breach'] = ready_for_delivery['SLA_Deadline'] - current_time
                        # First prioritize orders that can meet SLA
                        ready_for_delivery['Can_Meet_SLA'] = ready_for_delivery.apply(
                            lambda x: current_time + x['Est_Delivery_Duration'] <= x['SLA_Deadline'], axis=1
                        )
                        ready_for_delivery = ready_for_delivery.sort_values(
                            by=['Can_Meet_SLA', 'Time_To_SLA_Breach', 'Est_Delivery_Duration'],
                            ascending=[False, True, True]
                        )
                
                # Process each regular delivery order until we run out of bikers or orders
                while not ready_for_delivery.empty and available_bikers:
                    # Get the next order for delivery
                    order_idx = ready_for_delivery.index[0]
                    order = ready_for_delivery.iloc[0]
                    picking_end = order['Picking End Time']
                    
                    # Calculate delivery times
                    distance = order['Last Mile Distance From Branch']
                    delivery_one_way = pd.Timedelta(minutes=int(4*distance))
                    customer_wait = pd.Timedelta(minutes=5)
                    return_journey = pd.Timedelta(minutes=int(4*distance))
                    total_delivery_time = delivery_one_way + customer_wait + return_journey
                    
                    # Assign to the first available biker
                    biker_id = available_bikers[0]
                    available_bikers.pop(0)  # Remove the assigned biker from available list
                    
                    delivery_start = max(picking_end, current_time)  # Start delivery as soon as picking is done
                    customer_handover_time = delivery_start + delivery_one_way
                    delivery_end = delivery_start + total_delivery_time
                    
                    # Check if delivery would end after 6 PM
                    if delivery_end <= end_of_day:
                        # Update the order data
                        df.loc[order_idx, 'Assigned Biker'] = biker_id
                        df.loc[order_idx, 'Delivery Start Time'] = delivery_start
                        df.loc[order_idx, 'Customer Handover Time'] = customer_handover_time
                        df.loc[order_idx, 'Delivery End Time'] = delivery_end
                        df.loc[order_idx, 'Order Status'] = 'Delivered'
                        df.loc[order_idx, 'Is Batched'] = False
                        
                        # Update biker availability
                        bikers[biker_id] = delivery_end
                        
                        # Add to biker schedule
                        biker_schedules[biker_id].append({
                            'Order No': order['Order No'],
                            'Start': delivery_start,
                            'End': delivery_end,
                            'Activity': f"Order {order['Order No']} ({distance:.1f} km)",
                            'Distance': distance,
                            'Is Batched': False
                        })
                        
                        # Check SLA
                        original_order_time = order['Order Placed Date Time']
                        if original_order_time < early_order_cutoff_time:
                            sla_time = early_order_sla_time
                        else:
                            sla_time = original_order_time + pd.Timedelta(hours=2)
                            
                        if delivery_end <= sla_time:
                            df.loc[order_idx, 'SLA Met'] = True
                            if verbose:
                                print(f"Order {order['Order No']} assigned to biker {biker_id}. Delivery from {delivery_start} to {delivery_end}. SLA MET!")
                        else:
                            if verbose:
                                print(f"Order {order['Order No']} assigned to biker {biker_id}. Delivery from {delivery_start} to {delivery_end}. SLA MISSED!")
                    else:
                        if verbose:
                            print(f"Order {order['Order No']} cannot be delivered before end of day. Skipping.")
                    
                    # Remove from delivery orders
                    orders_to_deliver = orders_to_deliver.drop(order_idx)
                    ready_for_delivery = ready_for_delivery.drop(order_idx)
        
        # Process orders waiting for picking (if pickers are available)
        if not orders_to_pick.empty and available_pickers:
            # Filter orders that have arrived by the current time
            available_orders = orders_to_pick[orders_to_pick['Order Placed Time For Processing'] <= current_time]
            
            if not available_orders.empty:
                # Only apply prioritization strategy if there are multiple orders to choose from
                # Otherwise, just pick the single available order - no need to leave pickers idle
                if len(available_orders) > 1:
                    if scheduling_strategy == "FCFS":
                        # First Come First Served - sort by order time
                        prioritized_orders = available_orders.sort_values('Order Placed Time For Processing')
                    elif scheduling_strategy == "MAXIMIZE_ORDERS":
                        # Sort by estimated delivery duration (shortest first to maximize throughput)
                        prioritized_orders = available_orders.sort_values(['Est_Delivery_Duration', 'Order Placed Time For Processing'])
                    elif scheduling_strategy == "MAXIMIZE_SLA":
                        # Calculate time remaining to meet SLA
                        available_orders['Time_To_SLA_Breach'] = available_orders['SLA_Deadline'] - current_time
                        
                        # Estimate if order can still meet SLA
                        picking_duration = pd.Timedelta(minutes=picking_time_mins)
                        available_orders['Can_Meet_SLA'] = available_orders.apply(
                            lambda x: current_time + picking_duration + x['Est_Delivery_Duration'] <= x['SLA_Deadline'], 
                            axis=1
                        )
                        
                        # First prioritize orders that can meet SLA, then by urgency (least time to SLA breach)
                        prioritized_orders = available_orders.sort_values(
                            by=['Can_Meet_SLA', 'Time_To_SLA_Breach', 'Order Placed Time For Processing'], 
                            ascending=[False, True, True]
                        )
                    else:
                        # Default to FCFS
                        prioritized_orders = available_orders.sort_values('Order Placed Time For Processing')
                else:
                    # If there's only one order, no prioritization needed
                    prioritized_orders = available_orders
                
                # Process each order until we run out of pickers or orders
                while not prioritized_orders.empty and available_pickers:
                    # Get the next order for picking
                    order_idx = prioritized_orders.index[0]
                    order = prioritized_orders.iloc[0]
                    
                    # Assign to an available picker
                    picker_id = available_pickers[0]
                    available_pickers.pop(0)  # Remove the assigned picker from available list
                    
                    # Set picking times
                    picking_start = current_time
                    picking_duration = pd.Timedelta(minutes=picking_time_mins)
                    picking_end = picking_start + picking_duration
                    
                    # Update the order data
                    df.loc[order_idx, 'Assigned Picker'] = picker_id
                    df.loc[order_idx, 'Picking Start Time'] = picking_start
                    df.loc[order_idx, 'Picking End Time'] = picking_end
                    
                    # Update picker availability
                    pickers[picker_id] = picking_end
                    
                    if verbose:
                        print(f"Order {order['Order No']} assigned to picker {picker_id}. Picking from {picking_start} to {picking_end}")
                    
                    # Move the order to the delivery queue
                    order_copy = order.copy()
                    order_copy['Picking End Time'] = picking_end
                    orders_to_deliver = pd.concat([orders_to_deliver, pd.DataFrame([order_copy])], ignore_index=False)
                    
                    # Remove from orders to pick
                    orders_to_pick = orders_to_pick.drop(order_idx)
                    prioritized_orders = prioritized_orders.drop(order_idx)
        
        # If no actions were performed in this iteration, we need to advance time
        next_time_points = []
        
        # Next order arrival
        if not orders_to_pick.empty:
            unarrived_orders = orders_to_pick[orders_to_pick['Order Placed Time For Processing'] > current_time]
            if not unarrived_orders.empty:
                next_time_points.append(unarrived_orders['Order Placed Time For Processing'].min())
        
        # Next picker becoming available
        busy_pickers = [t for t in pickers if t > current_time]
        if busy_pickers:
            next_time_points.append(min(busy_pickers))
        
        # Next biker becoming available
        busy_bikers = [t for t in bikers if t > current_time]
        if busy_bikers:
            next_time_points.append(min(busy_bikers))
        
        # If there are no next time points, we're done
        if not next_time_points:
            if verbose:
                print(f"No more events to process. Ending simulation at {current_time}")
            break
        
        # Advance time to the next event
        next_time = min(next_time_points)
        if verbose and next_time > current_time:
            print(f"Advancing time from {current_time} to {next_time}")
        current_time = next_time
        
        # Continue to the next iteration after advancing time
        # This will re-evaluate available pickers, bikers, and orders
        continue
    
    # Prepare results
    total_orders = len(df)
    delivered_orders = df[df['Order Status'] == 'Delivered'].shape[0]
    undelivered_orders = total_orders - delivered_orders
    sla_met_orders = df[df['SLA Met'] == True].shape[0]
    sla_percentage = (sla_met_orders / delivered_orders * 100) if delivered_orders > 0 else 0
    
    if verbose:
        print(f"\nSimulation completed for scheduling strategy: {scheduling_strategy}")
        print(f"Total orders: {total_orders}, Delivered: {delivered_orders}, SLA met: {sla_met_orders}")
        print(f"SLA percentage: {sla_percentage:.2f}%")
    
    results = {
        'orders_df': df,
        'total_orders': total_orders,
        'delivered_orders': delivered_orders,
        'undelivered_orders': undelivered_orders,
        'sla_met_orders': sla_met_orders,
        'sla_percentage': sla_percentage,
        'biker_schedules': biker_schedules
    }
    
    return results


# The prioritize_orders function is no longer needed as the prioritization logic
# has been integrated directly into the simulate_order_processing function


def visualize_biker_schedules(biker_schedules, processing_date):
    """Visualize the schedules of all bikers"""
    
    # Skip if there are no biker schedules
    if not any(biker_schedules.values()):
        print("No biker schedules to visualize.")
        return
    
    # Get the day's date for the plot
    day_start = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(9, 0)))  # Store opening time
    picker_start = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(9, 30)))  # Picker start time
    biker_start = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(10, 0)))  # Biker start time
    day_end = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(18, 0)))
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(15, len(biker_schedules) * 1.5))
    
    # Set up the plot
    ax.set_yticks(range(len(biker_schedules)))
    ax.set_yticklabels([f'Biker {i}' for i in biker_schedules.keys()])
    ax.set_xlim(day_start, day_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Add a grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Color map for distance
    cmap = plt.cm.get_cmap('YlOrRd')
    max_distance = max([activity.get('Distance', 0) for biker_id, schedule in biker_schedules.items() 
                        for activity in schedule] or [10])
    
    # Plot each biker's schedule
    for biker_id, schedule in biker_schedules.items():
        for activity in schedule:
            # Extract info
            start = activity['Start']
            end = activity['End']
            order_no = activity.get('Order No', 'N/A')
            distance = activity.get('Distance', 0)
            is_batched = activity.get('Is Batched', False)
            batch_size = activity.get('Batch Size', 1)
            
            # Calculate color based on distance
            color = 'orangered' if is_batched else cmap(distance / max_distance)
            
            # Create rectangle for the activity
            rect = Rectangle((mdates.date2num(start), biker_id - 0.4), 
                           mdates.date2num(end) - mdates.date2num(start), 0.8, 
                           facecolor=color, 
                           edgecolor='darkred' if is_batched else 'navy',
                           alpha=0.8,
                           hatch='///' if is_batched else None)
            ax.add_patch(rect)
            
            # Add order number as text
            middle_time = start + (end - start) / 2
            label = f"Batch ({batch_size})" if is_batched else f"#{order_no}"
            ax.text(mdates.date2num(middle_time), biker_id, label, 
                    ha='center', va='center', fontsize=9, 
                    color='white' if is_batched or distance/max_distance > 0.5 else 'black')
    
    # Add color bar for distance
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_distance))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Distance (km)')
    
    # Add a legend for batched orders
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orangered', hatch='///', label='Batched Delivery', alpha=0.8),
        Patch(facecolor=cmap(0.5), label='Regular Delivery', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title and labels
    ax.set_title(f'Biker Schedule for {processing_date}', fontsize=14)
    ax.set_xlabel('Time of Day', fontsize=12)
    
    plt.tight_layout()
    plt.show()


def display_simulation_results(simulation_results, show_all_orders=False):
    """Display the results of the simulation in a structured format"""
    result_df = simulation_results['orders_df']
    
    print("\n============= SIMULATION RESULTS SUMMARY =============")
    print(f"Total Orders: {simulation_results['total_orders']}")
    print(f"Delivered Orders: {simulation_results['delivered_orders']} " + 
          f"({simulation_results['delivered_orders'] / simulation_results['total_orders'] * 100:.2f}%)")
    print(f"Undelivered Orders: {simulation_results['undelivered_orders']} " + 
          f"({simulation_results['undelivered_orders'] / simulation_results['total_orders'] * 100:.2f}%)")
    
    # SLA metrics only for delivered orders
    if simulation_results['delivered_orders'] > 0:
        print(f"Orders Meeting SLA (delivered within 2 hours): {simulation_results['sla_met_orders']} " + 
              f"({simulation_results['sla_percentage']:.2f}% of delivered orders)")
    else:
        print("No orders delivered, cannot calculate SLA metrics.")
        
    # Order timing details table - save to Excel file instead of printing
    print("\nGenerating detailed order timing Excel report...")
    
    # Prepare timing information in a readable format
    timing_df = result_df[['Order No', 'Order Placed Date Time', 'Picking Start Time', 
                          'Picking End Time', 'Delivery Start Time', 'Customer Handover Time', 'Delivery End Time',
                          'Last Mile Distance From Branch', 'Assigned Picker', 'Assigned Biker',
                          'Order Status', 'SLA Met', 'Is Previous Day']]
    
    # Add batch information if it exists
    if 'Is Batched' in result_df.columns:
        # Count batched orders
        batched_orders = result_df['Is Batched'].fillna(False).sum()
        if batched_orders > 0:
            batch_ids = result_df.loc[result_df['Is Batched'] == True, 'Batch ID'].dropna().unique()
            print(f"\nBatched Orders: {batched_orders} (in {len(batch_ids)} batches)")
            
            # Add batch info to timing_df
            timing_df['Is Batched'] = result_df['Is Batched']
            timing_df['Batch ID'] = result_df['Batch ID']
    
    # Rename columns for clarity
    timing_df = timing_df.rename(columns={
        'Order Placed Date Time': 'Order Placed Time',
        'Picking Start Time': 'Pick Start Time',
        'Picking End Time': 'Pick End Time', 
        'Delivery Start Time': 'Rider Out Time',
        'Customer Handover Time': 'Customer Handover Time',
        'Delivery End Time': 'Rider In Time',
        'Last Mile Distance From Branch': 'Distance (km)'
    })
    
    # Calculate time differences (durations)
    # Only for delivered orders
    delivered_mask = timing_df['Order Status'] == 'Delivered'
    if delivered_mask.any():
        # Pick duration
        timing_df.loc[delivered_mask, 'Pick Duration (mins)'] = (
            (timing_df.loc[delivered_mask, 'Pick End Time'] - 
             timing_df.loc[delivered_mask, 'Pick Start Time']).dt.total_seconds() / 60
        ).round(1)
        
        # Delivery duration
        timing_df.loc[delivered_mask, 'Delivery Duration (mins)'] = (
            (timing_df.loc[delivered_mask, 'Rider In Time'] - 
             timing_df.loc[delivered_mask, 'Rider Out Time']).dt.total_seconds() / 60
        ).round(1)
        
        # Total order fulfillment time
        timing_df.loc[delivered_mask, 'Total Time (mins)'] = (
            (timing_df.loc[delivered_mask, 'Rider In Time'] - 
             timing_df.loc[delivered_mask, 'Pick Start Time']).dt.total_seconds() / 60
        ).round(1)
    
    # Save to Excel file
    processing_date_str = simulation_results['orders_df']['Processing Date'].iloc[0].strftime('%Y-%m-%d')
    excel_filename = f'order_timing_report_{processing_date_str}.xlsx'
    
    # Create Excel writer with xlsxwriter engine
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        timing_df.to_excel(writer, sheet_name='Order Timing Details', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Order Timing Details']
        
        # Create formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'align': 'center',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'})
        number_format = workbook.add_format({'num_format': '0.0'})
        center_format = workbook.add_format({'align': 'center'})
        
        # Apply formats
        # Set column widths
        worksheet.set_column('A:A', 10)   # Order No
        worksheet.set_column('B:G', 20)   # DateTime columns (including Customer Handover Time)
        worksheet.set_column('H:H', 12)   # Distance
        worksheet.set_column('I:J', 10)   # Assigned resources
        worksheet.set_column('K:K', 15)   # Status
        worksheet.set_column('L:L', 10)   # SLA Met
        worksheet.set_column('M:P', 15)   # Durations
        
        # Write header row with format
        for col_num, value in enumerate(timing_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply date formatting to all datetime columns
        for col_idx, col_name in enumerate(timing_df.columns):
            if col_name in ['Order Placed Time', 'Pick Start Time', 'Pick End Time', 
                           'Rider Out Time', 'Customer Handover Time', 'Rider In Time']:
                # Apply date format to the entire column
                col_letter = chr(65 + col_idx)  # Convert to column letter (A, B, C, etc.)
                worksheet.set_column(f'{col_letter}:{col_letter}', 20, date_format)
        
        # Apply conditional formatting for SLA Met column
        sla_col = timing_df.columns.get_loc('SLA Met')
        worksheet.conditional_format(1, sla_col, len(timing_df), sla_col, 
                                    {'type': '2_color_scale',
                                     'min_color': '#FFC7CE',  # Light red for False
                                     'max_color': '#C6EFCE'}) # Light green for True
                                     
        # Alternative direct approach with cell formula
        worksheet.conditional_format(1, sla_col, len(timing_df), sla_col, 
                                    {'type': 'cell',
                                     'criteria': '==',
                                     'value': 'TRUE',
                                     'format': workbook.add_format({'bg_color': '#C6EFCE'})})
                                     
        worksheet.conditional_format(1, sla_col, len(timing_df), sla_col, 
                                    {'type': 'cell',
                                     'criteria': '==',
                                     'value': 'FALSE',
                                     'format': workbook.add_format({'bg_color': '#FFC7CE'})})
    
    print(f"Excel report saved as: {excel_filename}")
    
    # Previous day vs. current day orders
    prev_day_orders = result_df[result_df['Is Previous Day']].shape[0]
    curr_day_orders = result_df[~result_df['Is Previous Day']].shape[0]
    print(f"\nPrevious Day Orders: {prev_day_orders}")
    print(f"Current Day Orders: {curr_day_orders}")
    
    # Detailed order status
    print("\n------------- ORDER STATUS BREAKDOWN -------------")
    status_counts = result_df.groupby('Order Status').size()
    for status, count in status_counts.items():
        print(f"{status}: {count} orders ({count/len(result_df)*100:.2f}%)")
    
    # SLA breakdown for delivered orders
    delivered_df = result_df[result_df['Order Status'] == 'Delivered']
    if not delivered_df.empty:
        print("\n------------- SLA BREAKDOWN (DELIVERED ORDERS) -------------")
        sla_counts = delivered_df.groupby('SLA Met').size()
        for sla, count in sla_counts.items():
            print(f"SLA {'Met' if sla else 'Missed'}: {count} orders ({count/len(delivered_df)*100:.2f}%)")
    
    # Previous day orders performance
    prev_day_delivered = result_df[(result_df['Is Previous Day']) & (result_df['Order Status'] == 'Delivered')].shape[0]
    if prev_day_orders > 0:
        print(f"\nPrevious Day Orders Delivered: {prev_day_delivered}/{prev_day_orders} " +
              f"({prev_day_delivered/prev_day_orders*100:.2f}%)")
    
    # Display all orders if requested
    if show_all_orders:
        print("\n------------- ALL ORDERS DETAILS -------------")
        display_cols = ['Order No', 'Order Placed Date Time', 'Last Mile Distance From Branch', 
                        'Order Status', 'Picking Start Time', 'Delivery End Time', 'SLA Met', 'Is Previous Day']
        print(result_df[display_cols].to_string())
    
    # Resource utilization
    print("\n------------- RESOURCE UTILIZATION -------------")
    # Calculate picker utilization
    picking_times = result_df.dropna(subset=['Picking Start Time', 'Picking End Time'])
    total_picking_time = sum((row['Picking End Time'] - row['Picking Start Time']).total_seconds() / 3600 
                           for _, row in picking_times.iterrows())
    
    # Calculate biker utilization
    delivery_times = result_df.dropna(subset=['Delivery Start Time', 'Delivery End Time'])
    total_delivery_time = sum((row['Delivery End Time'] - row['Delivery Start Time']).total_seconds() / 3600 
                            for _, row in delivery_times.iterrows())
    
    # Working hours (9 AM to 6 PM = 9 hours)
    working_hours = 9
    
    picker_count = len(set(result_df['Assigned Picker'].dropna()))
    biker_count = len(set(result_df['Assigned Biker'].dropna()))
    
    if picker_count > 0:
        picker_utilization = (total_picking_time / (picker_count * working_hours)) * 100
        print(f"Picker Utilization: {picker_utilization:.2f}%")
    
    if biker_count > 0:
        biker_utilization = (total_delivery_time / (biker_count * working_hours)) * 100
        print(f"Biker Utilization: {biker_utilization:.2f}%")


def find_optimal_configuration(orders_df, processing_date, picking_time_mins=15, max_pickers=10, max_bikers=10, 
                          target_sla=90, scheduling_strategy="FCFS", 
                          enable_batching=False, batch_size=2, batching_num_bikers=0):
    """
    Find the optimal configuration of pickers and bikers to meet a target SLA percentage
    
    Parameters:
    orders_df (DataFrame): DataFrame containing order data
    processing_date (datetime.date): The date orders are being processed
    picking_time_mins (int): Time taken for picking/packing an order in minutes
    max_pickers (int): Maximum number of pickers to consider
    max_bikers (int): Maximum number of bikers to consider
    target_sla (float): Target SLA percentage to achieve
    scheduling_strategy (str): Strategy for order scheduling:
                               "FCFS" - First Come First Served (default)
                               "MAXIMIZE_ORDERS" - Maximize total number of orders delivered
                               "MAXIMIZE_SLA" - Prioritize orders to maximize SLA compliance
    enable_batching (bool): Whether to enable batching of morning orders
    batch_size (int): Maximum number of orders that can be batched together
    batching_num_bikers (int): Number of bikers assigned for batched delivery (0 means all bikers can do batching)
    
    Returns:
    DataFrame: Results of all simulations
    """
    results = []
    found_optimal = False
    
    print(f"\nRunning optimization with picking time: {picking_time_mins} minutes")
    print(f"Using scheduling strategy: {scheduling_strategy}")
    
    if enable_batching:
        print(f"Order batching enabled. Batch size: {batch_size}")
        if batching_num_bikers > 0:
            print(f"Using {batching_num_bikers} dedicated bikers for batching")
    
    for pickers in range(1, max_pickers + 1):
        inner_loop_break = False
        for bikers in range(1, max_bikers + 1):
            print(f"Testing configuration: {pickers} pickers, {bikers} bikers...")
            
            # If using dedicated batching bikers, ensure we have enough total bikers
            actual_batching_bikers = min(batching_num_bikers, bikers) if batching_num_bikers > 0 else 0
            
            # Run the simulation
            sim_results = simulate_order_processing(
                orders_df, pickers, bikers, processing_date, 
                picking_time_mins=picking_time_mins,
                scheduling_strategy=scheduling_strategy,
                enable_batching=enable_batching,
                batch_size=batch_size,
                batching_num_bikers=actual_batching_bikers
            )
            
            total_orders = sim_results['total_orders']
            delivered_orders = sim_results['delivered_orders'] 
            sla_met_orders = sim_results['sla_met_orders']
            
            # Calculate actual percentage of SLA met (out of delivered orders)
            if delivered_orders > 0:
                sla_percentage = (sla_met_orders / delivered_orders) * 100
            else:
                sla_percentage = 0
                
            # Calculate delivery rate
            delivery_rate = (delivered_orders / total_orders) * 100 if total_orders > 0 else 0
            
            # Store the results
            results.append({
                'Pickers': pickers,
                'Bikers': bikers,
                'Delivered_Orders': delivered_orders,
                'Total_Orders': total_orders,
                'Delivery_Rate': delivery_rate,
                'SLA_Met_Orders': sla_met_orders,
                'SLA_Percentage': sla_percentage
            })
            
            # We consider a configuration optimal if:
            # 1. It meets or exceeds the target SLA percentage
            # 2. It delivers all orders
            if (sla_percentage >= target_sla and delivered_orders == total_orders):
                print(f"Found optimal configuration: {pickers} pickers, {bikers} bikers")
                print(f"  - Delivers all {delivered_orders}/{total_orders} orders")
                print(f"  - SLA Met: {sla_percentage:.2f}% (target: {target_sla}%)")
                inner_loop_break = True
                found_optimal = True
                break
        
        if inner_loop_break:
            break
            
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # If we didn't find an optimal configuration, show the best one
    if not found_optimal and len(results) > 0:
        # Find configuration with highest SLA percentage that delivers all orders
        full_delivery = results_df[results_df['Delivered_Orders'] == results_df['Total_Orders']]
        
        if not full_delivery.empty:
            best_config = full_delivery.loc[full_delivery['SLA_Percentage'].idxmax()]
            print(f"\nBest configuration (delivers all orders):")
            print(f"  - {int(best_config['Pickers'])} pickers, {int(best_config['Bikers'])} bikers")
            print(f"  - SLA Met: {best_config['SLA_Percentage']:.2f}% (target: {target_sla}%)")
        else:
            # Find configuration with highest delivery rate
            best_config = results_df.loc[results_df['Delivery_Rate'].idxmax()]
            print(f"\nBest configuration (highest delivery rate):")
            print(f"  - {int(best_config['Pickers'])} pickers, {int(best_config['Bikers'])} bikers")
            print(f"  - Delivers {int(best_config['Delivered_Orders'])}/{int(best_config['Total_Orders'])} orders ({best_config['Delivery_Rate']:.2f}%)")
            print(f"  - SLA Met: {best_config['SLA_Percentage']:.2f}% (target: {target_sla}%)")
    
    return results_df


def plot_optimization_results(results_df):
    """Plot the optimization results"""
    if results_df.empty:
        print("No results to plot.")
        return
        
    # Add a new column to identify configurations that deliver all orders
    if 'Total_Orders' in results_df.columns:
        results_df['All_Delivered'] = results_df['Delivered_Orders'] == results_df['Total_Orders']
    
    if plotly_available:
        try:
            # Create a more informative scatter plot with Plotly
            fig = px.scatter(
                results_df, 
                x='Pickers', 
                y='Bikers', 
                size='Delivery_Rate', 
                color='SLA_Percentage',
                hover_data=['Delivered_Orders', 'SLA_Met_Orders', 'Delivery_Rate', 'SLA_Percentage'],
                labels={
                    'SLA_Percentage': 'SLA %',
                    'Delivery_Rate': 'Delivery %'
                },
                title='Delivery Performance by Resource Configuration',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            # Add markers to highlight configurations that deliver all orders
            if 'All_Delivered' in results_df.columns:
                full_delivery = results_df[results_df['All_Delivered']]
                if not full_delivery.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=full_delivery['Pickers'],
                            y=full_delivery['Bikers'],
                            mode='markers',
                            marker=dict(
                                symbol='circle-open',
                                size=15,
                                color='red',
                                line=dict(width=2)
                            ),
                            name='Delivers All Orders'
                        )
                    )
            
            # Try to show the plot, but fall back to matplotlib if it fails
            try:
                fig.show()
            except (ValueError, ImportError) as e:
                print(f"Warning: Could not display Plotly visualization: {e}")
                print("Falling back to matplotlib visualization...")
                plotly_available = False
        except Exception as e:
            print(f"Warning: Error creating Plotly visualization: {e}")
            print("Falling back to matplotlib visualization...")
            plotly_available = False
    else:
        # Create heatmaps with matplotlib
        # First plot - SLA percentage
        pivot_sla = results_df.pivot(index='Bikers', columns='Pickers', values='SLA_Percentage')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_sla, annot=True, cmap='viridis', fmt='.1f')
        plt.title('SLA Percentage by Resource Configuration')
        plt.xlabel('Number of Pickers')
        plt.ylabel('Number of Bikers')
        plt.tight_layout()
        plt.show()
        
        # Second plot - Delivery rate
        pivot_delivery = results_df.pivot(index='Bikers', columns='Pickers', values='Delivery_Rate')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_delivery, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Delivery Rate by Resource Configuration')
        plt.xlabel('Number of Pickers')
        plt.ylabel('Number of Bikers')
        plt.tight_layout()
        plt.show()
        
        # If available, highlight configurations that deliver all orders
        if 'All_Delivered' in results_df.columns:
            full_delivery = results_df[results_df['All_Delivered']]
            if not full_delivery.empty:
                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    data=full_delivery, 
                    x='Pickers', 
                    y='Bikers', 
                    size='SLA_Percentage', 
                    hue='SLA_Percentage',
                    palette='viridis',
                    sizes=(100, 500),
                    legend='brief'
                )
                plt.title('Configurations That Deliver All Orders')
                plt.xlabel('Number of Pickers')
                plt.ylabel('Number of Bikers')
                plt.tight_layout()
                plt.show()


def main():
    print("\n============= DARK STORE ORDER PREDICTION AND SIMULATION =============")
    print("Running prediction and simulation using default historical data patterns...")
    
    # Initialize variables with defaults
    num_pickers = 3
    num_bikers = 5
    picking_time_mins = 15
    scheduling_strategy = "FCFS"
    enable_batching = False
    batch_size = 2
    batching_num_bikers = 0
    
    if not historical_data_available:
        print("Error: Historical data module is not available.")
        print("Please ensure 'historical_data.py' is in the same directory.")
        return
        
    # Use default historical data patterns
    print("\nLoading historical data patterns...")
    historical_data = load_historical_data("default")
    print(f"Successfully loaded {len(historical_data)} default order patterns.")
    print("Using built-in patterns for order distribution by day and hour.")
    
    # Get forecast date
    forecast_default_date = datetime.datetime.now().date()  # Use today's date as default
    forecast_date_str = input(f"Enter forecast date (YYYY-MM-DD, default: {forecast_default_date}): ") or str(forecast_default_date)
    forecast_date = pd.to_datetime(forecast_date_str).date()
    processing_date = forecast_date
    
    # Use fixed seasonality factor
    seasonality = 1.0
    
    # Generate forecast
    print(f"\nGenerating order forecast for {processing_date}...")
    forecasted_orders = generate_forecast(historical_data, processing_date, seasonality, 1.0)  # Fixed growth factor of 1.0
    
    # Save forecast as current orders
    orders_df = forecasted_orders
    all_orders_df = forecasted_orders
    
    print("\n============= SIMULATION SETUP =============")
    print(f"Processing date: {processing_date}")
    print(f"Total forecasted orders: {len(orders_df)}")
    
    # Get simulation parameters
    while True:
        try:
            num_pickers = int(input("\nEnter number of pickers (default: 3): ") or "3")
            num_bikers = int(input("Enter number of bikers (default: 5): ") or "5")
            picking_time_mins = int(input("Enter picking/packing time in minutes (default: 15): ") or "15")
            
            # Ask for scheduling strategy
            print("\nAvailable scheduling strategies:")
            print("1. FCFS - First Come First Served")
            print("2. MAXIMIZE_ORDERS - Maximize number of orders delivered in a day")
            print("3. MAXIMIZE_SLA - Maximize orders delivered within SLA")
            strategy_choice = input("Select scheduling strategy (1-3, default: 1): ") or "1"
            
            strategy_mapping = {
                "1": "FCFS",
                "2": "MAXIMIZE_ORDERS",
                "3": "MAXIMIZE_SLA"
            }
            
            scheduling_strategy = strategy_mapping.get(strategy_choice, "FCFS")
            
            # Ask about order batching
            enable_batching = input("\nEnable order batching for morning orders? (y/n, default: n): ").lower() == 'y'
            batch_size = 2  # Default batch size
            batching_num_bikers = 0  # Default: all bikers can do batching
            
            if enable_batching:
                batch_size = int(input("Maximum number of orders per batch (default: 2): ") or "2")
                
                # Ask if all bikers should handle batching or just a subset
                dedicated_batching = input("Use dedicated bikers for batching? (y/n, default: n): ").lower() == 'y'
                if dedicated_batching:
                    batching_num_bikers = int(input(f"Number of bikers for batching (max {num_bikers}, default: 2): ") or "2")
                    batching_num_bikers = min(batching_num_bikers, num_bikers)  # Ensure we don't exceed total bikers
                    
                print(f"\nBatching configuration:")
                print(f"- Max orders per batch: {batch_size}")
                if dedicated_batching:
                    print(f"- Using {batching_num_bikers} dedicated bikers for batching (bikers 0-{batching_num_bikers-1})")
                else:
                    print(f"- All bikers can handle batched orders")
            
            verbose = input("\nEnable verbose logging? (y/n, default: n): ").lower() == 'y'
            break
        except ValueError:
            print("Please enter valid numbers for pickers, bikers, and picking time.")
        # Get manual simulation parameters
        while True:
            try:
                num_pickers = int(input("\nEnter number of pickers (default: 3): ") or "3")
                num_bikers = int(input("Enter number of bikers (default: 5): ") or "5")
                picking_time_mins = int(input("Enter picking/packing time in minutes (default: 15): ") or "15")
                
                # Ask for scheduling strategy
                print("\nAvailable scheduling strategies:")
                print("1. FCFS - First Come First Served")
                print("2. MAXIMIZE_ORDERS - Maximize number of orders delivered in a day")
                print("3. MAXIMIZE_SLA - Maximize orders delivered within SLA")
                strategy_choice = input("Select scheduling strategy (1-3, default: 1): ") or "1"
                
                strategy_mapping = {
                    "1": "FCFS",
                    "2": "MAXIMIZE_ORDERS",
                    "3": "MAXIMIZE_SLA"
                }
                
                scheduling_strategy = strategy_mapping.get(strategy_choice, "FCFS")
                
                # Ask about order batching
                enable_batching = input("\nEnable order batching for morning orders? (y/n, default: n): ").lower() == 'y'
                batch_size = 2  # Default batch size
                batching_num_bikers = 0  # Default: all bikers can do batching
                
                if enable_batching:
                    batch_size = int(input("Maximum number of orders per batch (default: 2): ") or "2")
                    
                    # Ask if all bikers should handle batching or just a subset
                    dedicated_batching = input("Use dedicated bikers for batching? (y/n, default: n): ").lower() == 'y'
                    if dedicated_batching:
                        batching_num_bikers = int(input(f"Number of bikers for batching (max {num_bikers}, default: 2): ") or "2")
                        batching_num_bikers = min(batching_num_bikers, num_bikers)  # Ensure we don't exceed total bikers
                        
                    print(f"\nBatching configuration:")
                    print(f"- Max orders per batch: {batch_size}")
                    if dedicated_batching:
                        print(f"- Using {batching_num_bikers} dedicated bikers for batching (bikers 0-{batching_num_bikers-1})")
                    else:
                        print(f"- All bikers can handle batched orders")
                
                verbose = input("\nEnable verbose logging? (y/n, default: n): ").lower() == 'y'
                break
            except ValueError:
                print("Please enter valid numbers for pickers, bikers, and picking time.")
    
    print(f"\nRunning simulation with {num_pickers} pickers and {num_bikers} bikers, {picking_time_mins} min picking time...")
    print(f"Scheduling strategy: {scheduling_strategy}")
    if enable_batching:
        print(f"Order batching enabled for morning orders (max {batch_size} orders per batch)")
    
    # Run the simulation
    simulation_results = simulate_order_processing(
        orders_df, num_pickers, num_bikers, processing_date, 
        picking_time_mins=picking_time_mins, 
        scheduling_strategy=scheduling_strategy,
        enable_batching=enable_batching,
        batch_size=batch_size,
        batching_num_bikers=batching_num_bikers,
        verbose=verbose
    )
    
    # Display the results
    display_simulation_results(simulation_results)
    
    # Visualize biker schedules
    visualize_biker_schedules(simulation_results['biker_schedules'], processing_date)


if __name__ == "__main__":
    main()