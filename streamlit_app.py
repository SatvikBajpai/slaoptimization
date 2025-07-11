#!/usr/bin/env python3
# Order Processing Simulation - Streamlit Dashboard with Scheduling Strategies
# Interactive dashboard for Dark Store Order Processing Simulation

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import datetime
from pathlib import Path
import io
import base64
import traceback
import sys

# Configure the page first (must be the first Streamlit command)
st.set_page_config(
    page_title="Dark Store Order Processing Simulator",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import functions from prediction.py which has the order batching feature
try:
    import sys
    import os
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # First try with the .py extension
    try:
        from prediction import (
            preprocess_data, 
            filter_current_day_orders,
            simulate_order_processing, 
            visualize_biker_schedules,
            find_optimal_configuration,
            plot_optimization_results
        )
        # Also import historical data functions if available
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
            st.success("Successfully imported simulation functions and historical data analysis from prediction.py")
        except ImportError:
            historical_data_available = False
            st.warning("Historical data module not available. Predictive features will be limited.")
    except ImportError:
        # Try to import the module directly without relying on file extension
        import importlib.util
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("prediction", os.path.join(current_dir, "prediction.py"))
        prediction_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prediction_module)
        
        # Get the required functions from the module
        preprocess_data = prediction_module.preprocess_data
        filter_current_day_orders = prediction_module.filter_current_day_orders
        simulate_order_processing = prediction_module.simulate_order_processing
        visualize_biker_schedules = prediction_module.visualize_biker_schedules
        find_optimal_configuration = prediction_module.find_optimal_configuration
        plot_optimization_results = prediction_module.plot_optimization_results
        
        st.success("Successfully imported simulation functions from prediction.py using importlib (with batching feature)")
except Exception as e:
    st.error(f"Error importing simulation functions: {e}")
    st.write("Traceback:")
    st.code(traceback.format_exc())
    st.stop()

def render_visualization(fig):
    """Render matplotlib figure in Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf)
    plt.close(fig)

def create_download_link(df, filename="data.xlsx"):
    """Generate a link to download the DataFrame as Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Order Timing Details', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Order Timing Details']
        
        # Create formats
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'align': 'center', 'fg_color': '#D7E4BC', 'border': 1
        })
        
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'})
        
        # Apply formats - Set column widths
        worksheet.set_column('A:A', 10)   # Order No
        worksheet.set_column('B:G', 20)   # DateTime columns (including Customer Handover Time)
        worksheet.set_column('H:H', 12)   # Distance
        worksheet.set_column('I:J', 10)   # Assigned resources
        worksheet.set_column('K:K', 15)   # Status
        worksheet.set_column('L:L', 10)   # SLA Met
        worksheet.set_column('M:P', 15)   # Durations
        
        # Write header row with format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply date formatting to all datetime columns
        for col_idx, col_name in enumerate(df.columns):
            if col_name in ['Order Placed Time', 'Pick Start Time', 'Pick End Time', 
                           'Rider Out Time', 'Customer Handover Time', 'Rider In Time']:
                # Apply date format to the entire column
                col_letter = chr(65 + col_idx)  # Convert to column letter (A, B, C, etc.)
                worksheet.set_column(f'{col_letter}:{col_letter}', 20, date_format)
        
        # Apply conditional formatting for SLA Met column
        if 'SLA Met' in df.columns:
            sla_col = df.columns.get_loc('SLA Met')
            worksheet.conditional_format(1, sla_col, len(df), sla_col, 
                                        {'type': 'cell',
                                        'criteria': '==',
                                        'value': 'TRUE',
                                        'format': workbook.add_format({'bg_color': '#C6EFCE'})})
                                        
            worksheet.conditional_format(1, sla_col, len(df), sla_col, 
                                        {'type': 'cell',
                                        'criteria': '==',
                                        'value': 'FALSE',
                                        'format': workbook.add_format({'bg_color': '#FFC7CE'})})
    
    val = output.getvalue()
    b64 = base64.b64encode(val).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Excel File</a>'
    return href

def display_metrics(simulation_results):
    """Display key metrics in a structured format"""
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", f"{simulation_results['total_orders']}")
    
    with col2:
        delivery_rate = (simulation_results['delivered_orders'] / simulation_results['total_orders'] * 100) if simulation_results['total_orders'] > 0 else 0
        st.metric("Delivered Orders", f"{simulation_results['delivered_orders']} ({delivery_rate:.1f}%)")
    
    with col3:
        undelivery_rate = (simulation_results['undelivered_orders'] / simulation_results['total_orders'] * 100) if simulation_results['total_orders'] > 0 else 0
        st.metric("Undelivered Orders", f"{simulation_results['undelivered_orders']} ({undelivery_rate:.1f}%)")
    
    with col4:
        if simulation_results['delivered_orders'] > 0:
            st.metric("SLA Met", f"{simulation_results['sla_met_orders']} ({simulation_results['sla_percentage']:.1f}%)")
        else:
            st.metric("SLA Met", "N/A")

def display_simulation_results(simulation_results, show_charts=True):
    """Display the results of the simulation in a structured format with improved visuals"""
    # Create a copy of the result dataframe to avoid SettingWithCopyWarning
    result_df = simulation_results['orders_df'].copy()
    
    # Display metrics in a visually appealing card layout
    display_metrics(simulation_results)
    
    # Check if this simulation used batching
    batching_used = 'Is Batched' in result_df.columns and result_df['Is Batched'].any()
    
    # If batching was used, display batch metrics in a clean layout
    if batching_used:
        # Create a fresh copy to avoid warnings
        batched_orders = result_df[result_df['Is Batched'] == True].copy()
        num_batched_orders = len(batched_orders)
        batch_ids = batched_orders['Batch ID'].dropna().unique()
        
        # Create card-like container for batch info
        st.markdown("### 📦 Batching Performance")
        
        # Create metrics in 3 columns
        batch_cols = st.columns(3)
        with batch_cols[0]:
            st.metric(
                "Total Batched Orders", 
                num_batched_orders, 
                delta=f"{(num_batched_orders/len(result_df)*100):.1f}%" if len(result_df) > 0 else None,
                delta_color="off"
            )
        
        with batch_cols[1]:
            st.metric("Number of Batches", len(batch_ids))
            
        with batch_cols[2]:
            avg_batch_size = num_batched_orders / len(batch_ids) if len(batch_ids) > 0 else 0
            st.metric("Average Batch Size", f"{avg_batch_size:.1f}")
        
        # Display SLA performance comparison
        if num_batched_orders > 0:
            batched_sla_met = batched_orders[batched_orders['SLA Met'] == True].shape[0]
            batched_sla_rate = (batched_sla_met / num_batched_orders) * 100 if num_batched_orders > 0 else 0
            
            # Create a copy of the filtered dataframe to avoid warnings
            non_batched_delivered = result_df[
                (result_df['Is Batched'] != True) & 
                (result_df['Order Status'] == 'Delivered')
            ].copy()
            
            non_batched_sla_met = non_batched_delivered[non_batched_delivered['SLA Met'] == True].shape[0]
            non_batched_sla_rate = (non_batched_sla_met / len(non_batched_delivered)) * 100 if len(non_batched_delivered) > 0 else 0
            
            # Calculate the delta for comparison
            sla_delta = batched_sla_rate - non_batched_sla_rate
            
            st.markdown("#### SLA Performance Comparison")
            sla_cols = st.columns(2)
            
            with sla_cols[0]:
                st.metric(
                    "Batched Orders SLA Rate", 
                    f"{batched_sla_rate:.1f}%",
                    delta=f"{sla_delta:.1f}%" if sla_delta != 0 else None,
                    delta_color="normal" if sla_delta >= 0 else "inverse"
                )
                
            with sla_cols[1]:
                st.metric("Non-Batched Orders SLA Rate", f"{non_batched_sla_rate:.1f}%")
    
    # Prepare timing information in a readable format
    # Create a fresh copy to avoid pandas warnings
    timing_df = result_df[['Order No', 'Order Placed Date Time', 'Picking Start Time', 
                          'Picking End Time', 'Delivery Start Time', 'Customer Handover Time', 'Delivery End Time',
                          'Last Mile Distance From Branch', 'Assigned Picker', 'Assigned Biker',
                          'Order Status', 'SLA Met', 'Is Previous Day']].copy()
                          
    # Add batching columns if available
    if batching_used:
        timing_df.loc[:, 'Is Batched'] = result_df['Is Batched']
        timing_df.loc[:, 'Batch ID'] = result_df['Batch ID']
    
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
    
    # Calculate time differences (durations) only for delivered orders
    delivered_mask = timing_df['Order Status'] == 'Delivered'
    if delivered_mask.any():
        # Initialize the columns first to avoid SettingWithCopyWarning
        timing_df.loc[:, 'Pick Duration (mins)'] = np.nan
        timing_df.loc[:, 'Delivery Duration (mins)'] = np.nan
        timing_df.loc[:, 'Total Time (mins)'] = np.nan
        
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
    
    # Create tabs with modern icons for different sections of results
    results_tab1, results_tab2, results_tab3 = st.tabs([
        "📊 Performance Summary", 
        "📋 Order Details", 
        "📈 Visualizations"
    ])
    
    with results_tab1:
        st.markdown("### 🏆 Overall Performance")
        
        # Order status breakdown with modern visualization
        status_counts = result_df.groupby('Order Status').size().reset_index(name='count')
        status_counts['percentage'] = (status_counts['count'] / status_counts['count'].sum() * 100).round(1)
        
        # Process delivered orders data
        delivered_df = result_df[result_df['Order Status'] == 'Delivered'].copy()
        
        # Show clean warning if no orders delivered
        if delivered_df.empty:
            st.warning("⚠️ No orders were delivered in this simulation. Check your resource configuration.")
        else:
            # SLA breakdown with better visuals
            sla_counts = delivered_df.groupby('SLA Met').size().reset_index(name='count')
            sla_counts['SLA Met'] = sla_counts['SLA Met'].map({True: 'Met', False: 'Missed'})
            sla_counts['percentage'] = (sla_counts['count'] / sla_counts['count'].sum() * 100).round(1)
            
            # Display two columns with modern pie charts
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                # Order status pie chart with improved styling
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Better color palette
                colors = ['#38b000', '#d90429']
                
                # Create pie chart with modern styling
                ax.pie(
                    status_counts['count'], 
                    labels=status_counts['Order Status'], 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=colors, 
                    shadow=False,  # Remove shadow for cleaner look
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},  # Add white border
                    textprops={'fontsize': 12}
                )
                ax.axis('equal')
                plt.title('Order Status Breakdown', fontsize=14, pad=20)
                
                render_visualization(fig)
                
            with summary_col2:
                # SLA pie chart with improved styling
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Better color palette for SLA chart
                colors = ['#38b000', '#d90429']  # Green for met, red for missed
                
                # Highlight the SLA met portion
                explode = (0.1, 0) if sla_counts.iloc[0]['SLA Met'] == 'Met' else (0, 0.1)
                
                # Create pie chart with modern styling
                ax.pie(
                    sla_counts['count'], 
                    explode=explode, 
                    labels=sla_counts['SLA Met'], 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=colors, 
                    shadow=False,  # Remove shadow for cleaner look
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},  # Add white border
                    textprops={'fontsize': 12}
                )
                ax.axis('equal')
                plt.title('SLA Performance (Delivered Orders)', fontsize=14, pad=20)
                
                render_visualization(fig)
                
                # Add informative caption
                st.caption("SLA Met: Orders delivered within the promised time window")
        
        # Resource utilization section with modern styling
        st.markdown("### 📈 Resource Utilization")
        
        # Calculate picker utilization with improved calculation
        picking_times = result_df.dropna(subset=['Picking Start Time', 'Picking End Time'])
        total_picking_time = sum((row['Picking End Time'] - row['Picking Start Time']).total_seconds() / 3600 
                              for _, row in picking_times.iterrows())
        
        # Calculate biker utilization with improved calculation
        delivery_times = result_df.dropna(subset=['Delivery Start Time', 'Delivery End Time'])
        total_delivery_time = sum((row['Delivery End Time'] - row['Delivery Start Time']).total_seconds() / 3600 
                                for _, row in delivery_times.iterrows())
        
        # Working hours (9 AM to 6 PM = 9 hours)
        working_hours = 9
        
        # Get unique resource counts
        picker_count = len(set(result_df['Assigned Picker'].dropna()))
        biker_count = len(set(result_df['Assigned Biker'].dropna()))
        
        # Display utilization metrics and modern charts in columns
        util_col1, util_col2 = st.columns(2)
        
        with util_col1:
            if picker_count > 0:
                # Calculate picker utilization
                picker_utilization = (total_picking_time / (picker_count * working_hours)) * 100
                
                # Show metric with improved styling
                st.metric("Picker Utilization", 
                          f"{picker_utilization:.1f}%",
                          delta=f"{picker_utilization-50:.1f}%" if picker_utilization != 50 else None,
                          delta_color="normal" if picker_utilization >= 50 else "inverse")
                
                # Create a modern gauge chart for picker utilization
                fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
                
                # Use color based on utilization level
                if picker_utilization < 30:
                    color = '#ef476f'  # Red for low utilization
                elif picker_utilization < 60:
                    color = '#ffd166'  # Yellow for medium utilization
                else:
                    color = '#06d6a0'  # Green for high utilization
                
                # Create gradient background
                ax.barh(0, 100, height=0.8, color='#e9ecef', alpha=0.8, edgecolor='white', zorder=1)
                
                # Create utilization bar with clean styling
                bar = ax.barh(0, picker_utilization, height=0.8, color=color, alpha=0.9, 
                             edgecolor='white', linewidth=1, zorder=2)
                
                # Add percentage text on the bar
                ax.text(picker_utilization / 2, 0, f"{picker_utilization:.1f}%", 
                       ha='center', va='center', color='black', 
                       fontweight='bold', fontsize=14, zorder=3)
                
                # Add target markers
                ax.axvline(50, color='#495057', linestyle='-', alpha=0.3, linewidth=1, zorder=0)
                ax.axvline(75, color='#495057', linestyle='-', alpha=0.3, linewidth=1, zorder=0)
                ax.text(50, 1.1, "Target (50%)", ha='center', va='center', color='#495057', fontsize=8)
                ax.text(75, 1.1, "Optimal (75%)", ha='center', va='center', color='#495057', fontsize=8)
                
                # Clean up the chart
                ax.set_xlim(0, 100)
                ax.set_yticks([])
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xlabel("Utilization (%)", fontweight='bold')
                ax.set_title("Picker Utilization", fontweight='bold', fontsize=14, pad=10)
                ax.grid(axis='x', alpha=0.2)
                
                # Add explanatory note
                ax.text(0, -1.2, f"Total picking time: {total_picking_time:.1f} hrs ÷ ({picker_count} pickers × {working_hours} hrs)",
                       ha='left', va='center', color='#495057', fontsize=9)
                
                # Render the chart
                render_visualization(fig)
            else:
                st.metric("Picker Utilization", "N/A")
        
        with util_col2:
            if biker_count > 0:
                # Calculate biker utilization
                biker_utilization = (total_delivery_time / (biker_count * working_hours)) * 100
                
                # Show metric with improved styling
                st.metric("Biker Utilization", 
                          f"{biker_utilization:.1f}%",
                          delta=f"{biker_utilization-50:.1f}%" if biker_utilization != 50 else None,
                          delta_color="normal" if biker_utilization >= 50 else "inverse")
                
                # Create a modern gauge chart for biker utilization
                fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
                
                # Use color based on utilization level
                if biker_utilization < 30:
                    color = '#ef476f'  # Red for low utilization
                elif biker_utilization < 60:
                    color = '#ffd166'  # Yellow for medium utilization
                else:
                    color = '#06d6a0'  # Green for high utilization
                
                # Create gradient background
                ax.barh(0, 100, height=0.8, color='#e9ecef', alpha=0.8, edgecolor='white', zorder=1)
                
                # Create utilization bar with clean styling
                bar = ax.barh(0, biker_utilization, height=0.8, color=color, alpha=0.9, 
                             edgecolor='white', linewidth=1, zorder=2)
                
                # Add percentage text on the bar
                ax.text(biker_utilization / 2, 0, f"{biker_utilization:.1f}%", 
                       ha='center', va='center', color='black', 
                       fontweight='bold', fontsize=14, zorder=3)
                
                # Add target markers
                ax.axvline(50, color='#495057', linestyle='-', alpha=0.3, linewidth=1, zorder=0)
                ax.axvline(75, color='#495057', linestyle='-', alpha=0.3, linewidth=1, zorder=0)
                ax.text(50, 1.1, "Target (50%)", ha='center', va='center', color='#495057', fontsize=8)
                ax.text(75, 1.1, "Optimal (75%)", ha='center', va='center', color='#495057', fontsize=8)
                
                # Clean up the chart
                ax.set_xlim(0, 100)
                ax.set_yticks([])
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xlabel("Utilization (%)", fontweight='bold')
                ax.set_title("Biker Utilization", fontweight='bold', fontsize=14, pad=10)
                ax.grid(axis='x', alpha=0.2)
                
                # Add explanatory note
                ax.text(0, -1.2, f"Total delivery time: {total_delivery_time:.1f} hrs ÷ ({biker_count} bikers × {working_hours} hrs)",
                       ha='left', va='center', color='#495057', fontsize=9)
                
                # Render the chart
                render_visualization(fig)
            else:
                st.metric("Biker Utilization", "N/A")

    with results_tab2:
        st.markdown("### 📋 Order Details")
        
        # Add Excel download link with better styling
        processing_date_str = result_df['Processing Date'].iloc[0].strftime('%Y-%m-%d')
        excel_filename = f'order_timing_report_{processing_date_str}.xlsx'
        
        # Create download button with improved visibility
        download_col1, download_col2 = st.columns([1, 2])
        with download_col1:
            st.markdown(create_download_link(timing_df, excel_filename), unsafe_allow_html=True)
        
        # Add search/filter capability
        filter_container = st.container()
        with filter_container:
            search_col1, search_col2, search_col3 = st.columns(3)
            
            with search_col1:
                # Filter by order status
                status_options = ['All'] + list(timing_df['Order Status'].unique())
                selected_status = st.selectbox('Filter by Status', status_options)
            
            with search_col2:
                # Filter by SLA status for delivered orders
                sla_options = ['All', 'Met', 'Not Met']
                selected_sla = st.selectbox('Filter by SLA', sla_options)
            
            with search_col3:
                # Filter by batch status if batching was used
                if batching_used:
                    batch_options = ['All', 'Batched', 'Not Batched']
                    selected_batch = st.selectbox('Filter by Batch', batch_options)
        
        # Apply filters to create filtered_df
        filtered_df = timing_df.copy()
        
        # Apply status filter
        if selected_status != 'All':
            filtered_df = filtered_df[filtered_df['Order Status'] == selected_status]
        
        # Apply SLA filter (only for delivered orders)
        if selected_sla != 'All':
            # Only filter delivered orders
            delivered_mask = filtered_df['Order Status'] == 'Delivered'
            if selected_sla == 'Met':
                filtered_df = filtered_df[~delivered_mask | (delivered_mask & (filtered_df['SLA Met'] == True))]
            elif selected_sla == 'Not Met':
                filtered_df = filtered_df[~delivered_mask | (delivered_mask & (filtered_df['SLA Met'] == False))]
        
        # Apply batch filter if batching was used
        if batching_used and selected_batch != 'All':
            if selected_batch == 'Batched':
                filtered_df = filtered_df[filtered_df['Is Batched'] == True]
            elif selected_batch == 'Not Batched':
                filtered_df = filtered_df[filtered_df['Is Batched'] != True]
        
        # Show the filtered table with better formatting and full width
        if filtered_df.empty:
            st.warning("No orders match the selected filters.")
        else:
            st.dataframe(filtered_df, use_container_width=True, height=400)
            st.caption(f"Showing {len(filtered_df)} of {len(timing_df)} orders")
    
    with results_tab3:
        st.markdown("### 📊 Visualizations")
        
        if simulation_results['delivered_orders'] > 0:
            # Create tabs for different visualizations with better labels
            viz_tab1, viz_tab2 = st.tabs(["📈 Order Processing Timeline", "🚲 Biker Schedules"])
            
            with viz_tab1:
                st.markdown("#### Order Processing Timeline")
                
                # Filter for delivered orders only and ensure we have a copy
                delivered_orders = result_df[result_df['Order Status'] == 'Delivered'].copy().sort_values('Order Placed Date Time')
                
                if not delivered_orders.empty:
                    # Calculate proper figure height based on number of orders (min 5 inches)
                    num_orders = len(delivered_orders)
                    fig_height = max(5, min(15, num_orders * 0.4))  # Cap height at 15 inches
                    
                    # Create timeline plot with better proportions and modern styling
                    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
                    
                    # Define modern color palette
                    pick_color = '#3a86ff'      # Blue
                    delivery_color = '#ff9f1c'  # Orange
                    placed_color = '#1a535c'    # Dark teal
                    handover_color = '#4cc9f0'  # Light blue
                    
                    # Plot each order's timeline with modern styling
                    for i, (idx, order) in enumerate(delivered_orders.iterrows()):
                        # Picking phase with rounded corners
                        pick_duration = (order['Picking End Time'] - order['Picking Start Time']).total_seconds() / 60
                        ax.barh(i, pick_duration, 
                               left=mdates.date2num(order['Picking Start Time']), 
                               color=pick_color, label='Picking', 
                               height=0.6, alpha=0.8, edgecolor='white', linewidth=1)
                        
                        # Delivery phase with rounded corners
                        delivery_duration = (order['Delivery End Time'] - order['Delivery Start Time']).total_seconds() / 60
                        ax.barh(i, delivery_duration,
                               left=mdates.date2num(order['Delivery Start Time']), 
                               color=delivery_color, label='Delivery',
                               height=0.6, alpha=0.8, edgecolor='white', linewidth=1)
                        
                        # Order placed marker with modern styling
                        ax.scatter(mdates.date2num(order['Order Placed Date Time']), i, 
                                  marker='o', color=placed_color, s=50, 
                                  edgecolor='white', linewidth=1,
                                  label='Order Placed', zorder=5)
                        
                        # Customer handover marker with modern styling
                        if pd.notna(order['Customer Handover Time']):
                            ax.scatter(mdates.date2num(order['Customer Handover Time']), i, 
                                      marker='*', color=handover_color, s=120, 
                                      edgecolor='white', linewidth=1,
                                      label='Customer Handover', zorder=5)
                    
                    # Set up the plot with clean modern styling
                    ax.set_yticks(range(len(delivered_orders)))
                    
                    # Truncate order numbers if there are too many orders
                    y_labels = [f"Order {order['Order No']}" for _, order in delivered_orders.iterrows()]
                    if len(y_labels) > 20:
                        # Shorten some labels for better readability
                        for i in range(1, len(y_labels)-1):
                            if i % 3 != 0:
                                y_labels[i] = f"{order['Order No']}"
                    
                    ax.set_yticklabels(y_labels, fontsize=10)
                    
                    # Set better y-axis limits
                    ax.set_ylim(-0.5, len(delivered_orders) - 0.5)
                    
                    # Get the day's date for the plot
                    processing_date = result_df['Processing Date'].iloc[0]
                    day_start = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(9, 0)))
                    day_end = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(18, 0)))
                    ax.set_xlim(day_start, day_end)
                    
                    # Format x-axis as times with clean styling
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    ax.tick_params(axis='x', labelsize=10)
                    
                    # Add a subtle grid for better readability
                    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
                    
                    # Add time markers for important hours with subtle styling
                    for hour in [9, 12, 15, 18]:
                        hour_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(hour, 0)))
                        ax.axvline(x=mdates.date2num(hour_time), color='#495057', linestyle='--', alpha=0.3, zorder=0)
                    
                    # Add legend with clean styling and better positioning
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), 
                             loc='upper right', fontsize=10, framealpha=0.9,
                             ncol=2, edgecolor='lightgray')
                    
                    # Set title and labels with modern styling
                    ax.set_title(f'Order Processing Timeline for {processing_date_str}', 
                                fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Orders', fontsize=12, fontweight='bold')
                    
                    # Set background color for cleaner look
                    ax.set_facecolor('#f8f9fa')
                    
                    render_visualization(fig)
                    
                    # Add an elegant legend explaining the timeline elements
                    st.info("� **Picking** → 🔶 **Delivery** → ⚪ Order Placed → ⭐ Customer Handover")
                else:
                    st.warning("No delivered orders to visualize.")
            
            with viz_tab2:
                st.markdown("#### Biker Schedules")
                
                if any(simulation_results['biker_schedules'].values()):
                    # Calculate proper figure height based on number of bikers with reasonable limits
                    num_bikers = len(simulation_results['biker_schedules'])
                    fig_height = max(5, min(12, num_bikers * 1.2))  # Cap at 12 inches
                    
                    # Create a matplotlib figure with modern styling
                    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
                    
                    # Get the day's date for the plot
                    processing_date = result_df['Processing Date'].iloc[0]
                    day_start = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(9, 0)))
                    day_end = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(18, 0)))
                    
                    # Set up the plot with clean styling
                    ax.set_yticks(range(len(simulation_results['biker_schedules'])))
                    ax.set_yticklabels([f'Biker {i}' for i in simulation_results['biker_schedules'].keys()])
                    ax.set_xlim(day_start, day_end)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    
                    # Add a subtle grid for better readability
                    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
                    
                    # Set better padding at the top of the plot
                    ax.set_ylim(-0.5, len(simulation_results['biker_schedules']) - 0.5)
                    
                    # Use a modern color map for distance
                    cmap = plt.cm.get_cmap('viridis')
                    max_distance = max([activity.get('Distance', 0) for biker_id, schedule in simulation_results['biker_schedules'].items() 
                                      for activity in schedule] or [10])
                    
                    # Plot each biker's schedule with modern styling
                    for biker_id, schedule in simulation_results['biker_schedules'].items():
                        for activity in schedule:
                            # Extract info
                            start = activity['Start']
                            end = activity['End']
                            order_no = activity.get('Order No', 'N/A')
                            distance = activity.get('Distance', 0)
                            
                            # Calculate color based on distance
                            color = cmap(distance / max_distance)
                            
                            # Create rectangle for the activity with modern styling
                            rect = Rectangle((mdates.date2num(start), biker_id - 0.35), 
                                           mdates.date2num(end) - mdates.date2num(start), 0.7, 
                                           facecolor=color, edgecolor='white', alpha=0.85,
                                           linewidth=1)
                            ax.add_patch(rect)
                            
                            # Determine text color based on background brightness for better readability
                            # Use a proper luminance calculation
                            r, g, b = color[0], color[1], color[2]
                            luminance = 0.299 * r + 0.587 * g + 0.114 * b
                            text_color = 'white' if luminance < 0.6 else 'black'
                            
                            # Add order number as text
                            middle_time = start + (end - start) / 2
                            
                            # Only show text if the rectangle is wide enough
                            width_in_hours = (end - start).total_seconds() / 3600
                            if width_in_hours >= 0.25:  # If at least 15 minutes
                                ax.text(mdates.date2num(middle_time), biker_id, f"#{order_no}", 
                                        ha='center', va='center', fontsize=9, 
                                        color=text_color, fontweight='bold')
                    
                    # Add color bar with modern styling
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_distance))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
                    cbar.set_label('Distance (km)', fontsize=10, fontweight='bold')
                    
                    # Set title and labels with modern styling
                    ax.set_title(f'Biker Schedule for {processing_date_str}', fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
                    
                    # Add time markers for important hours with subtle styling
                    for hour in [9, 12, 15, 18]:
                        hour_time = pd.Timestamp(datetime.datetime.combine(processing_date, datetime.time(hour, 0)))
                        ax.axvline(x=mdates.date2num(hour_time), color='#495057', linestyle='--', alpha=0.3)
                    
                    # Set background color for cleaner look
                    ax.set_facecolor('#f8f9fa')
                    
                    render_visualization(fig)
                    
                    # Add an elegant legend explaining the color scale
                    st.info("� **Color intensity** indicates delivery distance: darker colors represent longer distances.")
                    
                    # Add stats about biker workload
                    st.markdown("#### 🚲 Biker Workload Distribution")
                    
                    # Calculate workload per biker
                    biker_stats = {}
                    for biker_id, schedule in simulation_results['biker_schedules'].items():
                        total_time = sum((activity['End'] - activity['Start']).total_seconds() / 60 for activity in schedule)
                        delivery_count = len(schedule)
                        total_distance = sum(activity.get('Distance', 0) for activity in schedule)
                        
                        biker_stats[biker_id] = {
                            'Total Time (mins)': round(total_time, 1),
                            'Deliveries': delivery_count,
                            'Total Distance (km)': round(total_distance, 1)
                        }
                    
                    # Convert to dataframe for display
                    biker_stats_df = pd.DataFrame.from_dict(biker_stats, orient='index')
                    biker_stats_df.index.name = 'Biker ID'
                    biker_stats_df = biker_stats_df.reset_index()
                    
                    st.dataframe(biker_stats_df, use_container_width=True)
                else:
                    st.warning("No biker schedules to visualize.")
        else:
            st.warning("⚠️ No delivered orders to visualize. Try running the simulation with different parameters.")

def run_optimization_analysis(orders_df, processing_date, picking_time_mins, max_pickers, max_bikers, target_sla, 
                           scheduling_strategy="FCFS", enable_batching=False, batch_size=2, batching_num_bikers=0):
    """Run optimization analysis and display results"""
    with st.spinner("Running optimization analysis..."):
        try:
            # Show batching configuration if enabled
            if enable_batching:
                st.info(f"Optimization with batching: Max {batch_size} orders per batch, " + 
                        (f"{batching_num_bikers} dedicated bikers" if batching_num_bikers > 0 else "all bikers for batching"))
            
            optimal_config = find_optimal_configuration(
                orders_df, processing_date, picking_time_mins=picking_time_mins, 
                max_pickers=max_pickers, max_bikers=max_bikers, target_sla=target_sla,
                scheduling_strategy=scheduling_strategy,
                enable_batching=enable_batching,
                batch_size=batch_size,
                batching_num_bikers=batching_num_bikers
            )
            
            if not optimal_config.empty:
                st.subheader("Optimization Results")
                
                # Add a new column to identify configurations that deliver all orders
                optimal_config['All_Delivered'] = optimal_config['Delivered_Orders'] == optimal_config['Total_Orders']
                
                # Create tabs for different optimization views
                opt_tab1, opt_tab2 = st.tabs(["📊 Summary", "🔍 Detailed Results"])
                
                with opt_tab1:
                    # Display a summary of the top configurations
                    st.write("Top Configurations:")
                    summary = optimal_config.sort_values(['Delivery_Rate', 'SLA_Percentage'], ascending=False).head(5)
                    st.dataframe(summary[['Pickers', 'Bikers', 'Delivered_Orders', 'Total_Orders', 'Delivery_Rate', 'SLA_Percentage']], use_container_width=True)
                    
                    # Highlight optimal configuration that meets SLA target
                    full_delivery = optimal_config[optimal_config['All_Delivered']]
                    if not full_delivery.empty:
                        sla_met = full_delivery[full_delivery['SLA_Percentage'] >= target_sla]
                        if not sla_met.empty:
                            # Get the configuration with the minimum resources
                            best_config = sla_met.sort_values(['Pickers', 'Bikers']).iloc[0]
                            
                            st.success(f"✅ Optimal Configuration Found: {int(best_config['Pickers'])} pickers, {int(best_config['Bikers'])} bikers")
                            st.write(f"This configuration delivers all orders with an SLA performance of {best_config['SLA_Percentage']:.1f}%")
                        else:
                            st.warning(f"⚠️ Found configurations that deliver all orders, but none meet the target SLA of {target_sla}%")
                            best_config = full_delivery.sort_values('SLA_Percentage', ascending=False).iloc[0]
                            st.write(f"Best available: {int(best_config['Pickers'])} pickers, {int(best_config['Bikers'])} bikers with SLA performance of {best_config['SLA_Percentage']:.1f}%")
                    else:
                        st.warning("⚠️ No configuration found that can deliver all orders")
                        best_config = optimal_config.sort_values(['Delivery_Rate', 'SLA_Percentage'], ascending=False).iloc[0]
                        st.write(f"Best available: {int(best_config['Pickers'])} pickers, {int(best_config['Bikers'])} bikers")
                        st.write(f"Delivers {int(best_config['Delivered_Orders'])}/{int(best_config['Total_Orders'])} orders ({best_config['Delivery_Rate']:.1f}%) with SLA performance of {best_config['SLA_Percentage']:.1f}%")
                
                with opt_tab2:
                    # Create optimization visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # SLA percentage heatmap
                        st.write("SLA Performance by Configuration")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        pivot_sla = optimal_config.pivot(index='Bikers', columns='Pickers', values='SLA_Percentage')
                        sns.heatmap(pivot_sla, annot=True, cmap='viridis', fmt='.1f', ax=ax)
                        plt.title('SLA Percentage by Resource Configuration')
                        plt.xlabel('Number of Pickers')
                        plt.ylabel('Number of Bikers')
                        plt.tight_layout()
                        render_visualization(fig)
                    
                    with viz_col2:
                        # Delivery rate heatmap
                        st.write("Delivery Rate by Configuration")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        pivot_delivery = optimal_config.pivot(index='Bikers', columns='Pickers', values='Delivery_Rate')
                        sns.heatmap(pivot_delivery, annot=True, cmap='YlGnBu', fmt='.1f', ax=ax)
                        plt.title('Delivery Rate by Resource Configuration')
                        plt.xlabel('Number of Pickers')
                        plt.ylabel('Number of Bikers')
                        plt.tight_layout()
                        render_visualization(fig)
                    
                    # If there are configurations that deliver all orders, highlight them
                    full_delivery = optimal_config[optimal_config['All_Delivered']]
                    if not full_delivery.empty:
                        st.subheader("Configurations That Deliver All Orders")
                        st.dataframe(full_delivery[['Pickers', 'Bikers', 'SLA_Percentage']], use_container_width=True)
                    
                    # Allow downloading the optimization results
                    processing_date_str = processing_date.strftime('%Y-%m-%d')
                    st.download_button(
                        label="Download Optimization Results",
                        data=optimal_config.to_csv(index=False).encode('utf-8'),
                        file_name=f'optimization_results_{processing_date_str}.csv',
                        mime='text/csv',
                    )
            else:
                st.error("No optimization results to display.")
        except Exception as e:
            st.error(f"Error during optimization: {e}")
            st.write("Traceback:")
            st.code(traceback.format_exc())

def load_data_and_generate_recommendations(selected_forecast_date=None):
    """
    Helper function to load data and generate recommendations only when needed
    
    Parameters:
    selected_forecast_date (datetime, optional): If provided, use this date for forecasting
                                               instead of tomorrow's date
    """
    # If forecast date is provided and different from current, reset forecast and recommendations
    if selected_forecast_date is not None and ('forecast_date' not in st.session_state or 
                                             selected_forecast_date != st.session_state.forecast_date):
        if 'forecasted_orders' in st.session_state:
            del st.session_state.forecasted_orders
        if 'resource_recommendations' in st.session_state:
            del st.session_state.resource_recommendations
    
    if 'historical_data' not in st.session_state:
        with st.spinner("Loading default historical patterns..."):
            try:
                historical_data = load_historical_data("default")
                st.session_state.historical_data = historical_data
                st.session_state.data_loading_complete = True
            except Exception as e:
                st.error(f"Error loading default patterns: {e}")
                st.exception(e)
                return False
    
    if 'forecasted_orders' not in st.session_state:
        with st.spinner("Generating order forecast..."):
            try:
                if selected_forecast_date is None:
                    forecast_date = pd.to_datetime("today") + pd.Timedelta(days=1)
                else:
                    forecast_date = selected_forecast_date
                
                forecasted_orders = generate_forecast(
                    st.session_state.historical_data,
                    forecast_date,
                    seasonality_factor=1.0,
                    growth_factor=1.0
                )
                st.session_state.forecasted_orders = forecasted_orders
                st.session_state.forecast_date = forecast_date.date()
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                st.exception(e)
                return False
    
    if 'resource_recommendations' not in st.session_state:
        with st.spinner("Generating resource recommendations..."):
            try:
                # Ensure forecasted_orders has 'Is Morning' column for resource recommendations
                forecasted_df = st.session_state.forecasted_orders.copy()
                if 'Is Morning' not in forecasted_df.columns:
                    forecasted_df.loc[:, 'Is Morning'] = forecasted_df['Order Placed Date Time'].dt.hour < 12
                
                target_sla = 90  # Fixed target SLA percentage
                recommendations = recommend_resources(
                    st.session_state.historical_data,
                    forecasted_df,
                    target_sla
                )
                st.session_state.resource_recommendations = recommendations
                
                # Store orders in session state for simulation
                st.session_state.orders_df = st.session_state.forecasted_orders.copy()
                st.session_state.last_date = st.session_state.forecast_date
            except Exception as e:
                st.error(f"Error generating resource recommendations: {e}")
                st.exception(e)
                return False
                
    return True  # Data is already loaded

def main():
    # Clear layout and attractive header
    st.title("🚚 Dark Store Order Processing Simulator")
    
    # Progress indicator to show status during data loading
    progress_container = st.empty()
    
    # Sidebar configuration (clean and minimal)
    with st.sidebar:
        st.title("Simulation Dashboard")
        
        st.markdown("""
        This simulator uses built-in historical order patterns to:
        - 📊 Forecast orders for selected date
        - 🚲 Simulate order processing with your chosen parameters
        - 🔄 Get recommendations based on simulation results
        """)
        
        # Allow user to select a forecast date (default to tomorrow)
        default_forecast_date = pd.to_datetime("today") + pd.Timedelta(days=1)
        
        # Create a date selector with min date as today and max date as 14 days from today
        min_date = pd.to_datetime("today").date()
        max_date = (pd.to_datetime("today") + pd.Timedelta(days=14)).date()
        
        selected_date = st.date_input(
            "Select Forecast Date:",
            value=default_forecast_date.date(),
            min_value=min_date,
            max_value=max_date,
            help="Select a date to forecast orders for (up to 14 days in the future)"
        )
        
        # Convert to pandas datetime for consistency
        selected_forecast_date = pd.to_datetime(selected_date)
        
        st.success(f"**Forecast Date:** {selected_forecast_date.strftime('%B %d, %Y')}")
        
        # Add visual separation
        st.divider()
    
    # Only load historical data and generate forecasts (without recommendations yet)
    with progress_container:
        # Just load historical data and generate forecasted orders
        if 'historical_data' not in st.session_state:
            with st.spinner("Loading default historical patterns..."):
                try:
                    historical_data = load_historical_data("default")
                    st.session_state.historical_data = historical_data
                except Exception as e:
                    st.error(f"Error loading default patterns: {e}")
                    st.exception(e)
                    st.stop()
        
        # Generate forecasted orders
        if 'forecast_date' not in st.session_state or selected_forecast_date.date() != st.session_state.forecast_date:
            with st.spinner("Generating order forecast..."):
                try:
                    forecasted_orders = generate_forecast(
                        st.session_state.historical_data,
                        selected_forecast_date,
                        seasonality_factor=1.0,
                        growth_factor=1.0
                    )
                    st.session_state.forecasted_orders = forecasted_orders
                    st.session_state.forecast_date = selected_forecast_date.date()
                    # Store orders in session state for simulation
                    st.session_state.orders_df = forecasted_orders.copy()
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
                    st.exception(e)
                    st.stop()
    
    # Display forecast info
    if 'forecasted_orders' in st.session_state:
        order_count = len(st.session_state.forecasted_orders)
        
        # Add metrics to sidebar
        with st.sidebar:
            st.subheader("📊 Forecast Overview")
            st.metric("Forecasted Orders", order_count)
            
        # Create two columns for the main content
        left_col, right_col = st.columns([2, 1])
        
        with right_col:
            # Display forecast summary in a card-like container
            st.subheader("📈 Order Forecast")
            
            if not st.session_state.forecasted_orders.empty:
                forecasted_orders = st.session_state.forecasted_orders.copy()
                
                # Show morning vs afternoon split
                forecasted_orders.loc[:, 'Is Morning'] = forecasted_orders['Order Placed Date Time'].dt.hour < 12
                morning_count = forecasted_orders[forecasted_orders['Is Morning'] == True].shape[0]
                afternoon_count = forecasted_orders[forecasted_orders['Is Morning'] == False].shape[0]
                
                st.metric("Morning Orders (Before 12 PM)", morning_count)
                st.metric("Afternoon Orders (After 12 PM)", afternoon_count)
                
                # Create a distribution by hour chart
                hourly_forecast = forecasted_orders.groupby(forecasted_orders['Order Placed Date Time'].dt.hour)['Order No'].count()
                hour_df = pd.DataFrame({
                    'Hour': hourly_forecast.index,
                    'Orders': hourly_forecast.values
                })
                
                # Format the chart
                st.markdown("#### Hourly Distribution")
                st.bar_chart(hour_df.set_index('Hour'))
        
        with left_col:
            # Display simulation parameters in a clean layout
            st.subheader("⚙️ Simulation Parameters")
            
            # Set default values (conservative)
            default_pickers = 3
            default_bikers = 3
            default_picking_time = 15
            
            # Create columns for parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_pickers = st.slider("Number of Pickers", 1, 10, default_pickers)
            
            with col2:
                num_bikers = st.slider("Number of Bikers", 1, 10, default_bikers)
            
            with col3:
                picking_time_mins = st.slider("Picking Time (mins)", 5, 30, default_picking_time)
            
            # Allow user to select strategy
            strategy_options = {
                "FCFS": "First Come First Served - Process orders in sequence received",
                "MAXIMIZE_ORDERS": "Maximize Throughput - Prioritize shorter deliveries",
                "MAXIMIZE_SLA": "Maximize SLA - Prioritize orders at risk of SLA breach"
            }
            
            scheduling_strategy = st.selectbox(
                "Scheduling Strategy",
                options=list(strategy_options.keys()),
                format_func=lambda x: x,
                index=0  # Default to FCFS
            )
            
            # Show strategy description
            st.caption(strategy_options.get(scheduling_strategy, "Custom strategy"))
            
            # Order Batching section - user configurable
            st.markdown("#### 📦 Order Batching Configuration")
            
            enable_batching = st.checkbox("Enable Order Batching", value=False, 
                                        help="When enabled, multiple orders in the same direction can be assigned to a single biker")
            
            if enable_batching:
                batch_col1, batch_col2 = st.columns(2)
                
                with batch_col1:
                    batch_size = st.slider("Max Orders per Batch", 2, 5, 2)
                
                with batch_col2:
                    use_dedicated_bikers = st.checkbox("Use Dedicated Bikers", value=False)
                    
                if use_dedicated_bikers:
                    batching_num_bikers = st.slider("Dedicated Bikers for Batching", 1, num_bikers-1 if num_bikers > 1 else 1, 1)
                else:
                    batching_num_bikers = 0
            else:
                batch_size = 2
                batching_num_bikers = 0
                
            # Create two columns for single strategy vs all strategies
            sim_col1, sim_col2 = st.columns(2)
            
            # Define function to run all strategies and compare results
            def run_all_strategies(forecast_date, num_pickers, num_bikers, picking_time_mins, enable_batching, batch_size, batching_num_bikers):
                """Run simulations for all available strategies and compare results"""
                strategies = ["FCFS", "MAXIMIZE_ORDERS", "MAXIMIZE_SLA"]
                all_results = {}
                
                # Ensure orders_df is properly set
                if 'orders_df' not in st.session_state or st.session_state.orders_df.empty:
                    st.session_state.orders_df = st.session_state.forecasted_orders.copy()
                
                for strategy in strategies:
                    # Run the simulation for this strategy
                    results = simulate_order_processing(
                        st.session_state.orders_df.copy(),  # Use a copy to avoid any issues
                        num_pickers, 
                        num_bikers, 
                        forecast_date, 
                        picking_time_mins=picking_time_mins, 
                        scheduling_strategy=strategy,
                        enable_batching=enable_batching,
                        batch_size=batch_size,
                        batching_num_bikers=batching_num_bikers,
                        verbose=False
                    )
                    all_results[strategy] = results
                
                # Determine the best strategy
                best_strategy = strategies[0]  # Default to FCFS
                best_sla = all_results[best_strategy]['sla_percentage']
                best_delivered = all_results[best_strategy]['delivered_orders']
                
                for strategy in strategies[1:]:
                    current_sla = all_results[strategy]['sla_percentage']
                    current_delivered = all_results[strategy]['delivered_orders']
                    
                    # Prioritize SLA percentage, then delivered orders
                    if current_sla > best_sla or (current_sla == best_sla and current_delivered > best_delivered):
                        best_strategy = strategy
                        best_sla = current_sla
                        best_delivered = current_delivered
                
                return {
                    'all_results': all_results,
                    'best_strategy': best_strategy
                }
            
            # Run single strategy simulation button
            with sim_col1:
                if st.button("▶️ Run Selected Strategy", use_container_width=True):
                    with st.spinner(f"Running simulation with {scheduling_strategy} strategy..."):
                        try:
                            # Use the selected forecast date for simulation
                            forecast_date = st.session_state.forecast_date
                            
                            # Ensure orders_df is properly set
                            if 'orders_df' not in st.session_state or st.session_state.orders_df.empty:
                                st.session_state.orders_df = st.session_state.forecasted_orders.copy()
                            
                            # Run the simulation with selected strategy
                            simulation_results = simulate_order_processing(
                                st.session_state.orders_df, 
                                num_pickers, 
                                num_bikers, 
                                forecast_date, 
                                picking_time_mins=picking_time_mins, 
                                scheduling_strategy=scheduling_strategy,
                                enable_batching=enable_batching,
                                batch_size=batch_size,
                                batching_num_bikers=batching_num_bikers,
                                verbose=False
                            )
                            
                            # Store results in session state
                            st.session_state.simulation_results = simulation_results
                            st.session_state.all_strategy_results = None  # Clear any previous multi-strategy results
                            
                            # Generate recommendations based on simulation results if SLA is below target
                            if simulation_results['sla_percentage'] < 90:
                                # Generate resource recommendations based on simulation results
                                try:
                                    st.session_state.resource_recommendations = recommend_resources(
                                        forecast_date,
                                        st.session_state.orders_df.copy(),
                                        simulation_results['sla_percentage']
                                    )
                                except Exception as rec_error:
                                    st.warning(f"Could not generate recommendations: {rec_error}")
                                    st.session_state.resource_recommendations = None
                            else:
                                # Store standard recommendations
                                st.session_state.resource_recommendations = {
                                    'recommended_pickers': num_pickers,
                                    'recommended_bikers': num_bikers,
                                    'picking_time_mins': picking_time_mins,
                                    'recommended_strategy': scheduling_strategy,
                                    'enable_batching': enable_batching,
                                    'batch_size': batch_size,
                                    'batching_num_bikers': batching_num_bikers,
                                    'forecasted_orders': len(st.session_state.orders_df),
                                    'message': "Your configuration achieved target SLA. No changes needed."
                                }
                            st.session_state.has_results = True
                            st.session_state.simulation_params = {
                                "pickers": num_pickers,
                                "bikers": num_bikers,
                                "picking_time": picking_time_mins,
                                "strategy": scheduling_strategy,
                                "batching": enable_batching,
                                "batch_size": batch_size if enable_batching else "N/A"
                            }
                            
                            # Show success message
                            success_message = f"✅ Simulation completed with {scheduling_strategy} strategy"
                            if enable_batching:
                                success_message += f" and order batching enabled!"
                            else:
                                success_message += "!"
                            
                            st.success(success_message)
                        except Exception as e:
                            st.error(f"Error running simulation: {e}")
                            st.write("Traceback:")
                            st.code(traceback.format_exc())
            
            # Run all strategies simulation button
            with sim_col2:
                if st.button("🔄 Run All Strategies", use_container_width=True, type="primary"):
                    with st.spinner("Running simulations for all strategies..."):
                        try:
                            # Use the selected forecast date for simulation
                            forecast_date = st.session_state.forecast_date
                            
                            # Run simulations for all strategies
                            all_strategies_result = run_all_strategies(
                                forecast_date,
                                num_pickers,
                                num_bikers,
                                picking_time_mins,
                                enable_batching,
                                batch_size,
                                batching_num_bikers
                            )
                            
                            # Store results in session state
                            st.session_state.all_strategy_results = all_strategies_result
                            best_strategy = all_strategies_result['best_strategy']
                            
                            # Set the best strategy as the main result
                            st.session_state.simulation_results = all_strategies_result['all_results'][best_strategy]
                            st.session_state.has_results = True
                            
                            # Store the simulation parameters
                            st.session_state.simulation_params = {
                                "pickers": num_pickers,
                                "bikers": num_bikers,
                                "picking_time": picking_time_mins,
                                "strategy": best_strategy,
                                "batching": enable_batching,
                                "batch_size": batch_size if enable_batching else "N/A",
                                "is_multi_strategy": True
                            }
                            
                            # Generate recommendations based on best simulation results if SLA is below target
                            if st.session_state.simulation_results['sla_percentage'] < 90:
                                try:
                                    st.session_state.resource_recommendations = recommend_resources(
                                        forecast_date,
                                        st.session_state.orders_df.copy(),
                                        st.session_state.simulation_results['sla_percentage']
                                    )
                                except Exception as rec_error:
                                    st.warning(f"Could not generate recommendations: {rec_error}")
                                    st.session_state.resource_recommendations = None
                            else:
                                # Store standard recommendations
                                st.session_state.resource_recommendations = {
                                    'recommended_pickers': num_pickers,
                                    'recommended_bikers': num_bikers,
                                    'picking_time_mins': picking_time_mins,
                                    'recommended_strategy': best_strategy,
                                    'enable_batching': enable_batching,
                                    'batch_size': batch_size,
                                    'batching_num_bikers': batching_num_bikers,
                                    'forecasted_orders': len(st.session_state.orders_df),
                                    'message': f"Your configuration with {best_strategy} strategy achieved target SLA. No changes needed."
                                }
                            
                            # Show success message with best strategy highlighted
                            st.success(f"✅ Compared all strategies! **{best_strategy}** performed best with {st.session_state.simulation_results['sla_percentage']:.1f}% SLA and {st.session_state.simulation_results['delivered_orders']} orders delivered.")
                        except Exception as e:
                            st.error(f"Error running all strategies simulation: {e}")
                            st.write("Traceback:")
                            st.code(traceback.format_exc())
                        
        # Show results (only after simulation is run and button is clicked)
        if 'has_results' in st.session_state and st.session_state.has_results:
            st.markdown("---")
            st.header("📋 Simulation Results")
            
            # Check if we have results for all strategies
            has_multi_strategy = 'all_strategy_results' in st.session_state and st.session_state.all_strategy_results is not None
            
            if has_multi_strategy:
                # Create a container for strategy comparison
                st.markdown("### 🔍 Strategy Comparison")
                
                # Get all strategy results
                all_results = st.session_state.all_strategy_results['all_results']
                best_strategy = st.session_state.all_strategy_results['best_strategy']
                
                # Create a comparison table with more metrics
                comparison_data = []
                for strategy, results in all_results.items():
                    comparison_data.append({
                        "Strategy": strategy,
                        "SLA Met (%)": results['sla_percentage'],
                        "Delivered Orders": results['delivered_orders'],
                        "Total Orders": results['total_orders'],
                        "Delivery Rate (%)": (results['delivered_orders']/results['total_orders']*100),
                        "Is Best": "✅" if strategy == best_strategy else ""
                    })
                
                # Display comparison table
                comparison_df = pd.DataFrame(comparison_data)
                
                # Format the dataframe for better display
                formatted_df = comparison_df.copy()
                formatted_df["SLA Met (%)"] = formatted_df["SLA Met (%)"].apply(lambda x: f"{x:.1f}%")
                formatted_df["Delivery Rate (%)"] = formatted_df["Delivery Rate (%)"].apply(lambda x: f"{x:.1f}%")
                
                # Show a summary box highlighting the best strategy
                best_results = all_results[best_strategy]
                st.success(f"""
                ### 🏆 Best Strategy: {best_strategy}
                
                This strategy achieved:
                - **SLA Met**: {best_results['sla_percentage']:.1f}% ({best_results['sla_met_orders']} of {best_results['delivered_orders']} delivered orders)
                - **Delivery Rate**: {(best_results['delivered_orders']/best_results['total_orders']*100):.1f}% ({best_results['delivered_orders']} of {best_results['total_orders']} total orders)
                
                Compared to the other strategies, {best_strategy} performed better in {"SLA compliance" if best_results['sla_percentage'] > all_results['FCFS']['sla_percentage'] else "total delivered orders"}.
                """)
                
                # Add visual metric comparison using columns
                st.write("#### Strategy Performance Comparison")
                metric_cols = st.columns(3)
                
                # Find the best values for each metric
                best_sla = comparison_df["SLA Met (%)"].max()
                best_delivered = comparison_df["Delivered Orders"].max()
                best_rate = comparison_df["Delivery Rate (%)"].max()
                
                # Display metrics
                with metric_cols[0]:
                    for strategy, results in all_results.items():
                        delta = results['sla_percentage'] - list(all_results.values())[0]['sla_percentage'] if strategy != list(all_results.keys())[0] else None
                        st.metric(
                            f"{strategy} - SLA Met", 
                            f"{results['sla_percentage']:.1f}%",
                            delta=f"{delta:.1f}%" if delta is not None else None,
                            delta_color="normal" if delta is not None and delta >= 0 else "inverse"
                        )
                
                with metric_cols[1]:
                    for strategy, results in all_results.items():
                        delta = results['delivered_orders'] - list(all_results.values())[0]['delivered_orders'] if strategy != list(all_results.keys())[0] else None
                        st.metric(
                            f"{strategy} - Delivered",
                            results['delivered_orders'],
                            delta=delta,
                            delta_color="normal" if delta is not None and delta >= 0 else "inverse"
                        )
                
                with metric_cols[2]:
                    for strategy, results in all_results.items():
                        delivery_rate = (results['delivered_orders']/results['total_orders']*100)
                        first_strategy_rate = (list(all_results.values())[0]['delivered_orders']/list(all_results.values())[0]['total_orders']*100)
                        delta = delivery_rate - first_strategy_rate if strategy != list(all_results.keys())[0] else None
                        st.metric(
                            f"{strategy} - Delivery Rate",
                            f"{delivery_rate:.1f}%",
                            delta=f"{delta:.1f}%" if delta is not None else None,
                            delta_color="normal" if delta is not None and delta >= 0 else "inverse"
                        )
                
                # Create visual comparison charts
                st.write("#### Visual Comparison")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Create bar chart for SLA
                    sla_data = []
                    for strategy, results in all_results.items():
                        sla_data.append({
                            "Strategy": strategy,
                            "SLA Met (%)": results['sla_percentage'],
                            "Is Best": strategy == best_strategy
                        })
                    
                    sla_df = pd.DataFrame(sla_data)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(
                        sla_df['Strategy'], 
                        sla_df['SLA Met (%)'],
                        color=['#3a86ff' if not is_best else '#38b000' for is_best in sla_df['Is Best']]
                    )
                    
                    # Add value labels on top of bars
                    for i, v in enumerate(sla_df['SLA Met (%)']):
                        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
                    
                    # Add styling
                    ax.set_title("SLA Met by Strategy", fontweight='bold', fontsize=12)
                    ax.set_ylim(0, max(sla_df['SLA Met (%)']) * 1.15)  # Add some space for labels
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add target line at 90%
                    ax.axhline(y=90, color='red', linestyle='--', alpha=0.7)
                    ax.text(len(sla_df) - 0.5, 90 + 2, "Target (90%)", ha='right', color='red')
                    
                    st.pyplot(fig)
                
                with chart_col2:
                    # Create bar chart for Delivered Orders
                    delivery_data = []
                    for strategy, results in all_results.items():
                        delivery_data.append({
                            "Strategy": strategy,
                            "Delivered": results['delivered_orders'],
                            "Undelivered": results['undelivered_orders'],
                            "Is Best": strategy == best_strategy
                        })
                    
                    delivery_df = pd.DataFrame(delivery_data)
                    
                    # Create stacked bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bottom_bars = ax.bar(
                        delivery_df['Strategy'], 
                        delivery_df['Delivered'],
                        label='Delivered',
                        color=['#3a86ff' if not is_best else '#38b000' for is_best in delivery_df['Is Best']]
                    )
                    
                    top_bars = ax.bar(
                        delivery_df['Strategy'], 
                        delivery_df['Undelivered'],
                        bottom=delivery_df['Delivered'],
                        label='Undelivered',
                        color='#ef476f',
                        alpha=0.7
                    )
                    
                    # Add value labels
                    for i, (d, u) in enumerate(zip(delivery_df['Delivered'], delivery_df['Undelivered'])):
                        ax.text(i, d/2, f"{d}", ha='center', va='center', color='white', fontweight='bold')
                        if u > 0:
                            ax.text(i, d + u/2, f"{u}", ha='center', va='center', color='white', fontweight='bold')
                    
                    # Add styling
                    ax.set_title("Order Processing Results by Strategy", fontweight='bold', fontsize=12)
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    
                    st.pyplot(fig)
                
                # Display detailed comparison table
                st.write("#### Detailed Comparison")
                st.dataframe(formatted_df, hide_index=True, use_container_width=True)
                
                # Show explanation of each strategy
                with st.expander("📖 Strategy Explanations"):
                    st.markdown("""
                    **FCFS (First Come First Served)**: Orders are processed in the order they were received. This is a fair approach but may not optimize for SLA or order count.
                    
                    **MAXIMIZE_ORDERS**: Prioritizes shorter delivery trips to maximize the total number of orders delivered. Good when you want to handle as many orders as possible.
                    
                    **MAXIMIZE_SLA**: Prioritizes orders that are at risk of breaching SLA, based on remaining time until breach. Best when meeting delivery time commitments is critical.
                    """)
                
                # Allow user to select which strategy results to view
                selected_strategy = st.selectbox(
                    "Select strategy to view detailed results:", 
                    list(all_results.keys()),
                    index=list(all_results.keys()).index(best_strategy)
                )
                
                # Visual indicator of selected strategy
                st.info(f"Showing detailed results for **{selected_strategy}** strategy")
                
                # Update the simulation results based on selected strategy
                st.session_state.simulation_results = all_results[selected_strategy]
                
                # Update params for display
                params = st.session_state.simulation_params.copy()
                params['strategy'] = selected_strategy
            else:
                # Display simulation parameters used
                params = st.session_state.simulation_params
                
            # Show selected strategy parameters  
            st.caption(f"Results using {params['pickers']} pickers, {params['bikers']} bikers, {params['picking_time']} mins picking time, {params['strategy']} strategy{'with batching' if params['batching'] else ''}")
            
            try:
                display_simulation_results(st.session_state.simulation_results)
                
                # Show recommendations after simulation
                if 'resource_recommendations' in st.session_state:
                    with st.expander("💡 Resource Recommendations", expanded=True):
                        rec = st.session_state.resource_recommendations
                        st.info(rec['message'])
                        
                        # Create columns for recommendations
                        rec_col1, rec_col2, rec_col3 = st.columns(3)
                        with rec_col1:
                            st.metric("Recommended Pickers", rec['recommended_pickers'], 
                                    delta=rec['recommended_pickers'] - st.session_state.simulation_params['pickers'] 
                                    if rec['recommended_pickers'] != st.session_state.simulation_params['pickers'] else None)
                        
                        with rec_col2:
                            st.metric("Recommended Bikers", rec['recommended_bikers'],
                                    delta=rec['recommended_bikers'] - st.session_state.simulation_params['bikers'] 
                                    if rec['recommended_bikers'] != st.session_state.simulation_params['bikers'] else None)
                        
                        with rec_col3:
                            st.metric("Recommended Picking Time", f"{rec['picking_time_mins']} mins",
                                    delta=rec['picking_time_mins'] - st.session_state.simulation_params['picking_time'] 
                                    if rec['picking_time_mins'] != st.session_state.simulation_params['picking_time'] else None,
                                    delta_color="inverse" if rec['picking_time_mins'] < st.session_state.simulation_params['picking_time'] else "normal")
                        
                        # Show strategy recommendation if different and if not already compared all strategies
                        if 'recommended_strategy' in rec and not ('is_multi_strategy' in st.session_state.simulation_params and st.session_state.simulation_params['is_multi_strategy']):
                            if rec['recommended_strategy'] != params['strategy']:  # Use the local params which reflects the currently displayed strategy
                                st.warning(f"Consider changing strategy to **{rec['recommended_strategy']}** for better results")
                                if 'all_strategy_results' not in st.session_state or st.session_state.all_strategy_results is None:
                                    st.info("💡 Try clicking 'Run All Strategies' to compare different scheduling approaches automatically")
                        
                        # Show batching recommendation
                        if rec['enable_batching'] != st.session_state.simulation_params['batching']:
                            if rec['enable_batching']:
                                st.warning(f"Consider enabling batching (with batch size {rec['batch_size']}) for improved efficiency")
                            else:
                                st.success("Consider disabling batching for this scenario")
            except Exception as e:
                st.error(f"Error displaying results: {e}")
                st.write("Traceback:")
                st.code(traceback.format_exc())
    else:
        st.info("Select your simulation parameters and click 'Run Simulation' to see results and get recommendations.")
    
    # Footer
    with st.sidebar:
        st.markdown("---")
        st.caption("Dark Store Order Processing Simulator")
        st.caption("© 2025")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.write("Traceback:")
        st.code(traceback.format_exc())
