import dash
from dash import dcc, html, figure
import pandas as pd
import plotly.express as px
import base64
import subprocess
import io
import nbformat
from nbconvert import HTMLExporter

app = dash.Dash(__name__)

# Define the list of notebook files
notebook_files = ['performance_evaluation.ipynb', 'ddos_train_evaluate_export_models.ipynb', 'cyberattacks_train_evaluate_export_models.ipynb']

# Function to run a notebook and process its output
def run_notebook(notebook_file):
    proc = subprocess.Popen(['jupyter', 'nbconvert', '--execute', '--to', 'notebook', '--inplace', notebook_file], stdout=subprocess.PIPE)
    output, _ = proc.communicate()
    output = output.decode('utf-8')
    # Process the notebook output (e.g., extract relevant data or visuals)
    processed_output = process_output(output, notebook_file)
    return processed_output

# Function to process the notebook output and extract visualizations
def process_output(output, notebook_file):
    if notebook_file == 'performance_evaluation.ipynb':
        # Extract performance evaluation metrics from the output
        performance_metrics = extract_metrics(output, 'accuracy', 'precision', 'recall', 'f1-score')
        figure = create_metrics_figure(performance_metrics, 'Model Performance Metrics')
        return figure
    elif notebook_file == 'ddos_train_evaluate_export_models.ipynb':
        # Extract DDoS model performance metrics from the output
        ddos_metrics = extract_metrics(output, 'accuracy', 'precision', 'recall', 'f1-score')
        figure = create_metrics_figure(ddos_metrics, 'DDoS Attack Detection Performance')
        return figure
    elif notebook_file == 'cyberattacks_train_evaluate_export_models.ipynb':
        # Extract cyberattack model performance metrics from the output
        cyberattack_metrics = extract_metrics(output, 'accuracy', 'precision', 'recall', 'f1-score')
        figure = create_metrics_figure(cyberattack_metrics, 'Cyberattack Detection Performance')
        return figure

# Function to extract performance metrics from the notebook output
def extract_metrics(output, accuracy_metric, precision_metric, recall_metric, f1_metric):
    # Initialize variables to store the extracted metrics
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_scores = []

    # Iterate through each cell in the notebook output
    for cell in output.cells:
        # Check if the cell contains code
        if cell.cell_type == 'code':
            # Extract the relevant lines of code that contain the metrics
            for line in cell.source.split('\n'):
                if accuracy_metric in line and 'accuracy' in line:
                    accuracy_values.append(float(line.split(accuracy_metric)[1].strip()))
                elif precision_metric in line and 'precision' in line:
                    precision_values.append(float(line.split(precision_metric)[1].strip()))
                elif recall_metric in line and 'recall' in line:
                    recall_values.append(float(line.split(recall_metric)[1].strip()))
                elif f1_metric in line and 'f1-score' in line:
                    f1_scores.append(float(line.split(f1_metric)[1].strip()))

    # Create a dictionary to store the extracted metrics
    metrics = {
        'accuracy': accuracy_values,
        'precision': precision_values,
        'recall': recall_values,
        'f1-score': f1_scores
    }

    return metrics

# Function to create a Plotly figure for performance metrics
def create_metrics_figure(metrics_data, title):
    figure = {
        'data': [
            {'x': metrics_data['accuracy'], 'y': range(len(metrics_data['accuracy'])), 'type': 'bar', 'name': 'Accuracy'},
            {'x': metrics_data['precision'], 'y': range(len(metrics_data['precision'])), 'type': 'bar', 'name': 'Precision'},
            {'x': metrics_data['recall'], 'y': range(len(metrics_data['recall'])), 'type': 'bar', 'name': 'Recall'},
            {'x': metrics_data['f1-score'], 'y': range(len(metrics_data['f1-score'])), 'type': 'bar', 'name': 'F1-score'}
        ],
        'layout': {
            'title': title,
            'xaxis': {'title': 'Metric Value'},
            'yaxis': {'title': 'Model'},
            'barmode': 'group'
        }
    }
    return figure

# Define the dashboard layout
app.layout = html.Div([
    # Header
    html.H1('Cybersecurity Performance Dashboard'),

    # Tabs
    dcc.Tabs([
        dcc.Tab(label='Cyberattacks', children=[
            html.Div([
                html.H2('Cyberattack Detection Performance'),
                dcc.Graph(id='cyberattacks-graph')
            ])
        ]),
        dcc.Tab(label='DDoS', children=[
            html.Div([
                html.H2('DDoS Attack Detection Performance'),
                dcc.Graph(id='ddos-graph')
            ])
        ]),
        dcc.Tab(label='Performance Evaluations', children=[
            html.Div([
                html.H2('Model Evaluation Metrics'),
                dcc.Graph(id='performance-evaluations-graph')
            ])
        ])
    ]),

    # Footer
    html.Footer(['Copyright 2023 Gich-M. All rights reserved.']),
])

# Callback to update graphs based on notebook outputs
@app.callback(
    [Output('cyberattacks-graph', 'figure'),
     Output('ddos-graph', 'figure'),
     Output('performance-evaluations-graph', 'figure')],
    [])
def update_graphs():
    cyberattacks_figure = run_notebook('cyberattacks_train_evaluate_export_models.ipynb')
    ddos_figure = run_notebook('ddos_train_evaluate_export_models.ipynb')
    performance_evaluations_figure = run_notebook('performance_evaluation.ipynb')
    return cyberattacks_figure, ddos_figure, performance_evaluations_figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
