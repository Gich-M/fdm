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
notebook_files = ['performance_evaluation.ipynb',
                  'ddos_train_evaluate_export_models.ipynb',
                  'cyberattacks_train_evaluate_export_models.ipynb']

# Function to run a notebook and process its output


def run_notebook(notebook_file):
    proc = subprocess.Popen(['jupyter',
                             'nbconvert',
                             '--execute',
                             '--to',
                             'notebook',
                             '--inplace',
                             notebook_file],
                            stdout=subprocess.PIPE)
    output, _ = proc.communicate()
    output = output.decode('utf-8')
    # Process the notebook output (e.g., extract relevant data or visuals)
    processed_output = process_output(output, notebook_file)
    return processed_output

# Function to process the notebook output and extract visualizations


def process_output(output, notebook_file):
    if notebook_file == 'performance_evaluation.ipynb':
        # Extract performance evaluation metrics from the output
        performance_metrics, figures = extract_performance_metrics(output)
        return performance_metrics, figures
    elif notebook_file == 'ddos_train_evaluate_export_models.ipynb':
        # Extract DDoS model performance metrics from the output
        ddos_metrics, figures = extract_ddos_metrics(output)
        return ddos_metrics, figures
    elif notebook_file == 'cyberattacks_train_evaluate_export_models.ipynb':
        # Extract cyberattack model performance metrics from the output
        cyberattack_metrics, figures = extract_cyberattack_metrics(output)
        return cyberattack_metrics, figures

def extract_performance_metrics(output):
    # Extract performance metrics and figures for dtree, svm, knn, and mlp
    dtree_metrics, dtree_figures = extract_model_metrics(output, 'dtree')
    svm_metrics, svm_figures = extract_model_metrics(output, 'svm')
    knn_metrics, knn_figures = extract_model_metrics(output, 'knn')
    mlp_metrics, mlp_figures = extract_model_metrics(output, 'mlp')

    return {
        'dtree': dtree_metrics,
        'svm': svm_metrics,
        'knn': knn_metrics,
        'mlp': mlp_metrics
    }, dtree_figures, svm_figures, knn_figures, mlp_figures

# Function to extract DDoS model performance metrics and figures from the notebook output
def extract_ddos_metrics(output):
    accuracy, figures = extract_model_metrics(output, 'ddos')
    return {'accuracy': accuracy}, figures

# Function to extract cyberattack model performance metrics and figures from the notebook output
def extract_cyberattack_metrics(output):
    dtree_metrics, dtree_figures = extract_model_metrics(output, 'dtree')
    svm_metrics, svm_figures = extract_model_metrics(output, 'svm')
    knn_metrics, knn_figures = extract_model_metrics(output, 'knn')
    mlp_metrics, mlp_figures = extract_model_metrics(output, 'mlp')

    return {
        'dtree': dtree_metrics,
        'svm': svm_metrics,
        'knn': knn_metrics,
        'mlp': mlp_metrics
    }, dtree_figures, svm_figures, knn_figures, mlp_figures

# Function to extract performance metrics and figures for a specific model
def extract_model_metrics(output, model_name):
    accuracy_values = []
    classification_reports = []
    figures = []

    # Iterate through each cell in the notebook output
    for cell in output.cells:
        # Check if the cell contains code output
        if cell.cell_type == 'code' and cell.outputs:
            # Extract the relevant lines of code output that contain the metrics
            for output in cell.outputs:
                if output.output_type == 'stream':
                    text = output.text
                    for line in text.split('\n'):
                        if 'Accuracy:' in line and model_name in line:
                            accuracy_values.append(float(line.split('Accuracy:')[1].strip()))
                        elif 'precision recall f1-score support' in line and model_name in line:
                            classification_report = [line]
                            for i in range(4):
                                classification_report.append(next(iter(output.text.split('\n'))))
                            classification_reports.append('\n'.join(classification_report))
                # Extract the figures from the code output
                elif output.output_type == 'display_data':
                    figures.append(output.data)

    # Create a dictionary to store the extracted metrics
    metrics = {
        'accuracy': accuracy_values,
        'classification_report': classification_reports
    }

    return metrics, figures

# Callback to update graphs and outputs based on notebook outputs
@app.callback(
    [Output('cyberattacks-graph', 'figure'),
     Output('cyberattacks-output', 'children'),
     Output('ddos-graph', 'figure'),
     Output('ddos-output', 'children'),
     Output('dtree-graph', 'figure'),
     Output('dtree-output', 'children'),
     Output('svm-graph', 'figure'),
     Output('svm-output', 'children'),
     Output('knn-graph', 'figure'),
     Output('knn-output', 'children'),
     Output('mlp-graph', 'figure'),
     Output('mlp-output', 'children')],
    [])

def update_graphs_and_outputs():
    cyberattack_metrics, cyberattack_dtree_figures, cyberattack_svm_figures, cyberattack_knn_figures, cyberattack_mlp_figures = run_notebook('cyberattacks_train_evaluate_export_models.ipynb')
    ddos_metrics, ddos_figures = run_notebook('ddos_train_evaluate_export_models.ipynb')
    performance_metrics, dtree_figures, svm_figures, knn_figures, mlp_figures = run_notebook('performance_evaluation.ipynb')

    cyberattack_output = []
    for model, metrics in cyberattack_metrics.items():
        cyberattack_output.append(html.H3(f'{model.upper()} Model'))
        cyberattack_output.append(html.P(f'Accuracy: {metrics["accuracy"][0]}'))
        cyberattack_output.append(html.Pre('\n'.join(metrics['classification_report'])))

    ddos_output = [
        html.H3('DDoS Model'),
        html.P(f'Accuracy: {ddos_metrics["accuracy"][0]}')
    ]

    performance_output = []
    for model, metrics in performance_metrics.items():
        performance_output.append(html.H3(f'{model.upper()} Model'))
        performance_output.append(html.P(f'Accuracy: {metrics["accuracy"][0]}'))
        performance_output.append(html.Pre('\n'.join(metrics['classification_report'])))

    return [
        cyberattack_dtree_figures + cyberattack_svm_figures + cyberattack_knn_figures + cyberattack_mlp_figures, cyberattack_output,
        ddos_figures, ddos_output,
        dtree_figures, performance_output,
        svm_figures, performance_output,
        knn_figures, performance_output,
        mlp_figures, performance_output
    ]
