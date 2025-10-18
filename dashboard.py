import pandas as pd
import numpy as np
import os
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime


# File Paths
processed_data_dir = 'processed_data'
models_dir = 'trained_models'
anomalies_file = os.path.join(processed_data_dir, 'master_df_with_anomalies.csv')
forecast_file = os.path.join(processed_data_dir, 'final_forecast_df.csv')

try:
    # Load historical data with anomaly flags
    df_history = pd.read_csv(anomalies_file)
    df_history['date'] = pd.to_datetime(df_history['date'])
    df_history['anomaly_points'] = np.where(df_history['ensemble_anomaly'] == -1, df_history['vulnerability_index'], np.nan)

    # Load forecast data
    df_forecast = pd.read_csv(forecast_file)
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
    
    # Load the trained XGBoost models
    xgb_models = {}
    for cp_name in df_history['chokepoint_name'].unique():
        model_filename = os.path.join(models_dir, f'xgb_model_{cp_name.replace(" ", "_")}.joblib')
        xgb_models[cp_name] = joblib.load(model_filename)
    
    print(f" Data and {len(xgb_models)} XGBoost models loaded successfully.")
    
except Exception as e:
    exit()

# Initialize the Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

chokepoint_options = [{'label': name, 'value': name} for name in df_history['chokepoint_name'].unique()]

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
    
    # Header
    html.Div(style={'backgroundColor': '#1f2937', 'padding': '20px'}, children=[
        html.H1("Maritime Chokepoint Vulnerability Dashboard", style={'textAlign': 'center', 'color': 'white'}),
    ]),
    
    # Main Content
    html.Div(style={'padding': '20px'}, children=[
        
        # Historical Explorer
        html.H2("Historical Vulnerability Explorer", style={'borderBottom': '2px solid #ddd', 'paddingBottom': '10px'}),
        
        html.Div([
            html.Label("Select a Chokepoint:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Dropdown(id='chokepoint-dropdown-main', options=chokepoint_options, value=chokepoint_options[0]['value'], clearable=False, style={'width': '400px', 'display': 'inline-block'}),
        ]),
        
        dcc.Graph(id='main-vulnerability-chart'),
        dcc.Graph(id='component-stress-chart'),
        
        # Live Prediction Simulator
        html.H2("Live Prediction Simulator", style={'borderBottom': '2px solid #ddd', 'paddingBottom': '10px', 'marginTop': '40px'}),
        
        html.Div(className='row', style={'display': 'flex', 'alignItems': 'center'}, children=[
            
            # Input Sliders
            html.Div(className='six columns', style={'width': '45%', 'padding': '20px'}, children=[
                html.H4("Set Hypothetical Stress Levels for 'Tomorrow'"),
                
                html.Label("Normalized Traffic Stress (0=High Traffic, 1=Low Traffic)"),
                dcc.Slider(id='traffic-slider', min=0, max=1, step=0.05, value=0.5, marks={i/10: str(i/10) for i in range(11)}),
                
                html.Label("Normalized Weather Stress (0=Calm, 1=Severe)", style={'marginTop': '20px'}),
                dcc.Slider(id='weather-slider', min=0, max=1, step=0.05, value=0.2, marks={i/10: str(i/10) for i in range(11)}),

                html.Label("Normalized Geopolitical Risk (0=Low Risk, 1=High Risk)", style={'marginTop': '20px'}),
                dcc.Slider(id='geo-slider', min=0, max=1, step=0.05, value=0.1, marks={i/10: str(i/10) for i in range(11)}),

                html.Button('Run Prediction', id='predict-button', n_clicks=0, style={'marginTop': '30px', 'padding': '10px 20px', 'fontSize': '16px'}),
            ]),
            
            # Output Display
            html.Div(className='six columns', id='prediction-output-div', style={'width': '45%', 'padding': '20px', 'textAlign': 'center'}),
        ])
    ])
])


# Callback for Historical Explorer Graphs
@app.callback(
    [Output('main-vulnerability-chart', 'figure'),
     Output('component-stress-chart', 'figure')],
    [Input('chokepoint-dropdown-main', 'value')]
)
def update_historical_charts(selected_chokepoint):
    df_hist = df_history[df_history['chokepoint_name'] == selected_chokepoint]
    df_fore = df_forecast[df_forecast['chokepoint_name'] == selected_chokepoint]

    # Main Vulnerability Chart
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['vulnerability_index'], mode='lines', name='Historical Index', line=dict(color='black', width=1.5)))
    fig_main.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['anomaly_points'], mode='markers', name='Detected Anomaly', marker=dict(color='red', size=10, symbol='x')))
    fig_main.add_trace(go.Scatter(x=df_fore['ds'], y=df_fore['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False))
    fig_main.add_trace(go.Scatter(x=df_fore['ds'], y=df_fore['yhat_upper'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,176,246,0.2)', name='Uncertainty'))
    fig_main.add_trace(go.Scatter(x=df_fore['ds'], y=df_fore['yhat'], mode='lines', name='Forecast', line=dict(color='darkblue', width=2, dash='dash')))
    fig_main.update_layout(title=f'Vulnerability Index: History & Forecast for {selected_chokepoint}', yaxis_title="Vulnerability Index")

    # Component Stress Chart
    fig_comp = px.line(df_hist, x='date', y=['norm_traffic_stress', 'norm_weather_stress', 'norm_geopolitical_risk'],
                       title=f'Component Stress Scores for {selected_chokepoint}')
    fig_comp.update_layout(yaxis_title="Normalized Score (0-1)")

    return fig_main, fig_comp

# Callback for Live Prediction Simulator
@app.callback(
    Output('prediction-output-div', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('chokepoint-dropdown-main', 'value'),
     State('traffic-slider', 'value'),
     State('weather-slider', 'value'),
     State('geo-slider', 'value')]
)
def run_live_prediction(n_clicks, selected_chokepoint, traffic_stress, weather_stress, geo_risk):
    if n_clicks == 0:
        return ""

    # We use the same weighting as in the Week 4 notebook
    W_TRAFFIC, W_WEATHER, W_GEOPOLITICAL = 1/3, 1/3, 1/3
    hypothetical_index = (traffic_stress * W_TRAFFIC + weather_stress * W_WEATHER + geo_risk * W_GEOPOLITICAL)

    # We'll predict for "tomorrow"
    tomorrow = datetime.now() + pd.Timedelta(days=1)
    
    # Create the feature vector for the model
    xgb_features = pd.DataFrame([{
        'year': tomorrow.year,
        'month': tomorrow.month,
        'dayofweek': tomorrow.weekday(),
        'dayofyear': tomorrow.timetuple().tm_yday,
        'weekofyear': tomorrow.isocalendar()[1]
    }])
    
    # Load the correct model and make a prediction
    model = xgb_models[selected_chokepoint]
    xgb_prediction = model.predict(xgb_features)[0]

    return html.Div([
        html.H4("Prediction Results"),
        html.P(f"For Chokepoint: {selected_chokepoint}"),
        
        html.Div(style={'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '5px', 'margin': '10px 0'}, children=[
            html.H5("Vulnerability from Component Scores:"),
            html.H3(f"{hypothetical_index:.3f}", style={'color': '#e34234', 'fontSize': '28px'}),
            html.P("(Calculated by weighting your inputs)")
        ]),
        
        html.Div(style={'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '5px', 'margin': '10px 0'}, children=[
            html.H5("Vulnerability from Time-Based Forecast (XGBoost):"),
            html.H3(f"{xgb_prediction:.3f}", style={'color': '#007bff', 'fontSize': '28px'}),
            html.P(f"(Model's prediction for {tomorrow.strftime('%Y-%m-%d')})")
        ]),
    ])

if __name__ == '__main__':
    app.run(debug=True)