# SLA Optimization for Dark Store Order Processing

This Streamlit application simulates and optimizes dark store order processing, helping determine the optimal resource allocation (pickers and bikers) to meet SLA requirements.

## Features

- **Realistic Order Simulation**: Uses built-in historical data that matches real-world patterns
- **Multiple Scheduling Strategies**:
  - **FCFS (First Come First Served)**: Orders are processed in the order received
  - **MAXIMIZE_ORDERS**: Prioritizes shorter delivery trips to maximize total orders delivered
  - **MAXIMIZE_SLA**: Prioritizes orders at risk of breaching SLA deadline
- **Automatic Strategy Comparison**: Compares all strategies and highlights the best performer
- **Interactive Configuration**: Adjust number of pickers, bikers, and picking time
- **Comprehensive Visualizations**: View order timelines, biker schedules, and performance metrics
- **Resource Optimization**: Get recommendations for optimal resource configuration

## Installation

1. Clone this repository:

```bash
git clone https://github.com/SatvikBajpai/slaoptimization.git
cd slaoptimization
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Or use the provided convenience script:

```bash
chmod +x run_app.sh
./run_app.sh
```

## How to Use

1. Set simulation parameters in the sidebar (date, pickers, bikers, picking time, etc.)
2. Click "Run Simulation" to process orders with all strategies
3. View the recommended best strategy and its results
4. Toggle to view results from other strategies
5. Analyze visualizations and metrics to understand performance

## About

This tool helps dark stores and quick-commerce businesses optimize their operations by simulating different resource configurations and scheduling strategies to meet delivery SLAs while maximizing efficiency.

## How to Use

1. **Set Parameters**: Use the sidebar to set the simulation parameters:
   - Processing date
   - Number of pickers and bikers
   - Number of previous day orders
   - Verbosity level

2. **Run Simulation**: Click "Run Simulation" to process the orders based on the given parameters

3. **View Results**: Examine the order statistics, SLA performance, and resource utilization

4. **Optimization**: Optionally run optimization to find the ideal resource configuration

## Data Structure

The application uses built-in historical data with the following structure:
- Order ID
- Order Placed Date Time 
- Last Mile Distance (in km)

## Notes

- The app processes orders from 6 PM on the previous day through 6 PM on the processing day
- The simulation automatically compares all available strategies and recommends the best one
- All key simulation parameters can be adjusted from the sidebar
