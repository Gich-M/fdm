import dash
from dash import dcc, html, figure, Output
import pandas as pd
import plotly.express as px
import base64
import subprocess
import io
import nbformat
from nbconvert import HTMLExporter
import matplotlib.pyplot as plt
import plotly.graph_objs as go

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
        performance_metrics, figures, confusion_matrices = extract_performance_metrics(
            output)
        return performance_metrics, figures, confusion_matrices
    elif notebook_file == 'ddos_train_evaluate_export_models.ipynb':
        # Extract DDoS model performance metrics from the output
        ddos_metrics, figures, confusion_matrices = extract_ddos_metrics(
            output)
        return ddos_metrics, figures, confusion_matrices
    elif notebook_file == 'cyberattacks_train_evaluate_export_models.ipynb':
        # Extract cyberattack model performance metrics from the output
        cyberattack_metrics, figures, confusion_matrices = extract_cyberattack_metrics(
            output)
        return cyberattack_metrics, figures, confusion_matrices

# Function to extract performance metrics and figures from the notebook output


def extract_performance_metrics(output):
    # Extract performance metrics and figures for dtree, svm, knn, and mlp
    dtree_metrics, dtree_figures, dtree_confusion_matrices = extract_model_metrics(
        output, 'dtree')
    svm_metrics, svm_figures, svm_confusion_matrices = extract_model_metrics(
        output, 'svm')
    knn_metrics, knn_figures, knn_confusion_matrices = extract_model_metrics(
        output, 'knn')
    mlp_metrics, mlp_figures, mlp_confusion_matrices = extract_model_metrics(
        output, 'mlp')

    return {'dtree': dtree_metrics, 'svm': svm_metrics, 'knn': knn_metrics, 'mlp': mlp_metrics}, dtree_figures + svm_figures + \
        knn_figures + mlp_figures, dtree_confusion_matrices + svm_confusion_matrices + knn_confusion_matrices + mlp_confusion_matrices

# Function to extract DDoS model performance metrics and figures from the
# notebook output


def extract_ddos_metrics(output):
    accuracy, figures, confusion_matrices = extract_model_metrics(
        output, 'ddos')
    return {'accuracy': accuracy}, figures, confusion_matrices

# Function to extract cyberattack model performance metrics and figures
# from the notebook output


def extract_cyberattack_metrics(output):
    dtree_metrics, dtree_figures, dtree_confusion_matrices = extract_model_metrics(
        output, 'dtree')
    svm_metrics, svm_figures, svm_confusion_matrices = extract_model_metrics(
        output, 'svm')
    knn_metrics, knn_figures, knn_confusion_matrices = extract_model_metrics(
        output, 'knn')
    mlp_metrics, mlp_figures, mlp_confusion_matrices = extract_model_metrics(
        output, 'mlp')

    return {'dtree': dtree_metrics, 'svm': svm_metrics, 'knn': knn_metrics, 'mlp': mlp_metrics}, dtree_figures + svm_figures + \
        knn_figures + mlp_figures, dtree_confusion_matrices + svm_confusion_matrices + knn_confusion_matrices + mlp_confusion_matrices

# Function to extract performance metrics and figures for a specific model


def extract_model_metrics(output, model_name):
    accuracy_values = []
    classification_reports = []
    figures = []
    confusion_matrices = []

    # Iterate through each cell in the notebook output
    for cell in output.cells:
        # Check if the cell contains code output
        if cell.cell_type == 'code' and cell.outputs:
            # Extract the relevant lines of code output that contain the
            # metrics
            for output in cell.outputs:
                if output.output_type == 'stream':
                    text = output.text
                    for line in text.split('\n'):
                        if 'Accuracy:' in line and model_name in line:
                            accuracy_values.append(
                                float(line.split('Accuracy:')[1].strip()))
                        elif 'precision recall f1-score support' in line and model_name in line:
                            classification_report = [line]
                            for i in range(4):
                                classification_report.append(
                                    next(iter(output.text.split('\n'))))
                            classification_reports.append(
                                '\n'.join(classification_report))

                # Extract the figures from the code output
                elif output.output_type == 'display_data':
                    if 'image/png' in output.data:
                        # Convert matplotlib figure to Plotly graph object
                        fig = plt.figure()
                        plt.imshow(output.data['image/png'])
                        plt.axis('off')
                        plt.close(fig)  # Close the figure to release resources
                        plotly_fig = mpl_to_plotly(fig)
                        figures.append(plotly_fig)
                    elif 'image/jpeg' in output.data:
                        # Convert matplotlib figure to Plotly graph object
                        fig = plt.figure()
                        plt.imshow(output.data['image/jpeg'])
                        plt.axis('off')
                        plt.close(fig)  # Close the figure to release resources
                        plotly_fig = mpl_to_plotly(fig)
                        figures.append(plotly_fig)
                    else:
                        # Confusion matrix plot
                        confusion_matrices.append(output.data)

    # Create a dictionary to store the extracted metrics
    metrics = {
        'accuracy': accuracy_values,
        'classification_report': classification_reports
    }

    return metrics, figures, confusion_matrices

# Function to convert matplotlib figure to Plotly graph object


def mpl_to_plotly(fig):
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    # Convert matplotlib figure to a PIL image
    img = fig.canvas.print_to_buffer()
    from PIL import Image
    img = Image.frombytes('RGBA', (width, height), img)
    # Convert PIL image to Plotly image
    return go.layout.Image(
        source=img,
        x=0, y=0,
        xref='x', yref='y',
        sizex=1, sizey=1,
        sizing='stretch',
        layer='below'
    )


# Define the dashboard layout
app.layout = html.Div([
    # Header
    html.H1('Cybersecurity Performance Dashboard'),

    # Dropdown to select notebook
    dcc.Dropdown(
        id='notebook-dropdown',
        options=[
            {'label': 'Performance Evaluation',
             'value': 'performance_evaluation.ipynb'},
            {'label': 'DDoS', 'value': 'ddos_train_evaluate_export_models.ipynb'},
            {'label': 'Cyberattacks',
             'value': 'cyberattacks_train_evaluate_export_models.ipynb'}
        ],
        value='performance_evaluation.ipynb'
    ),

    # Tabs
    dcc.Tabs([
        dcc.Tab(label='Cyberattacks', children=[
            html.Div([
                html.H2('Cyberattack Detection Performance'),
                dcc.Graph(id='cyberattacks-graph'),
                html.Div(id='cyberattacks-output'),
                dcc.Graph(id='cyberattacks-confusion-matrix')
            ])
        ]),
        dcc.Tab(label='DDoS', children=[
            html.Div([
                html.H2('DDoS Attack Detection Performance'),
                dcc.Graph(id='ddos-graph'),
                html.Div(id='ddos-output'),
                dcc.Graph(id='ddos-confusion-matrix')
            ])
        ]),
        dcc.Tab(label='Performance Evaluations', children=[
            html.Div([
                html.H2('Model Evaluation Metrics'),
                dcc.Graph(id='dtree-graph'),
                html.Div(id='dtree-output'),
                dcc.Graph(id='dtree-confusion-matrix'),
                dcc.Graph(id='svm-graph'),
                html.Div(id='svm-output'),
                dcc.Graph(id='svm-confusion-matrix'),
                dcc.Graph(id='knn-graph'),
                html.Div(id='knn-output'),
                dcc.Graph(id='knn-confusion-matrix'),
                dcc.Graph(id='mlp-graph'),
                html.Div(id='mlp-output'),
                dcc.Graph(id='mlp-confusion-matrix')
            ])
        ])
    ]),
])

# Callback to update graphs and outputs based on notebook outputs


@app.callback(
    [Output('cyberattacks-graph', 'figure'),
     Output('cyberattacks-output', 'children'),
     Output('cyberattacks-confusion-matrix', 'figure'),
     Output('ddos-graph', 'figure'),
     Output('ddos-output', 'children'),
     Output('ddos-confusion-matrix', 'figure'),
     Output('dtree-graph', 'figure'),
     Output('dtree-output', 'children'),
     Output('dtree-confusion-matrix', 'figure'),
     Output('svm-graph', 'figure'),
     Output('svm-output', 'children'),
     Output('svm-confusion-matrix', 'figure'),
     Output('knn-graph', 'figure'),
     Output('knn-output', 'children'),
     Output('knn-confusion-matrix', 'figure'),
     Output('mlp-graph', 'figure'),
     Output('mlp-output', 'children'),
     Output('mlp-confusion-matrix', 'figure')
     ],
    [dash.dependencies.Input('notebook-dropdown', 'value')])
def update_graphs_and_outputs(notebook_file):
    if notebook_file == 'cyberattacks_train_evaluate_export_models.ipynb':
        cyberattack_metrics, cyberattack_figures, cyberattack_confusion_matrices = run_notebook(
            notebook_file)
        ddos_metrics, ddos_figures, ddos_confusion_matrices = [], [], []
        performance_metrics, performance_figures, performance_confusion_matrices = [], [], []

        cyberattack_output = []
        for model, metrics in cyberattack_metrics.items():
            cyberattack_output.append(html.H3(f'{model.upper()} Model'))
            cyberattack_output.append(
                html.P(f'Accuracy: {metrics["accuracy"][0]}'))
            cyberattack_output.append(
                html.Pre(
                    '\n'.join(
                        metrics['classification_report'])))

        return (
            cyberattack_figures, cyberattack_confusion_matrices,
            cyberattack_output,
            ddos_figures, ddos_confusion_matrices, [],
            performance_figures, performance_confusion_matrices, []
        )

    elif notebook_file == 'ddos_train_evaluate_export_models.ipynb':
        cyberattack_metrics, cyberattack_figures, cyberattack_confusion_matrices = [], [], []
        ddos_metrics, ddos_figures, ddos_confusion_matrices = run_notebook(
            notebook_file)
        performance_metrics, performance_figures, performance_confusion_matrices = [], [], []

        ddos_output = [
            html.H3('DDoS Model'),
            html.P(f'Accuracy: {ddos_metrics["accuracy"][0]}')
        ]

        return (
            cyberattack_figures, cyberattack_confusion_matrices, [],
            ddos_figures, ddos_confusion_matrices, ddos_output,
            performance_figures, performance_confusion_matrices, []
        )

    else:
        cyberattack_metrics, cyberattack_figures, cyberattack_confusion_matrices = [], [], []
        ddos_metrics, ddos_figures, ddos_confusion_matrices = [], [], []
        performance_metrics, performance_figures, performance_confusion_matrices = run_notebook(
            notebook_file)

        performance_output = []
        for model, metrics in performance_metrics.items():
            performance_output.append(html.H3(f'{model.upper()} Model'))
            performance_output.append(
                html.P(f'Accuracy: {metrics["accuracy"][0]}'))
            performance_output.append(
                html.Pre(
                    '\n'.join(
                        metrics['classification_report'])))

        return (
            cyberattack_figures,
            cyberattack_confusion_matrices,
            [],
            ddos_figures,
            ddos_confusion_matrices,
            [],
            performance_figures,
            performance_confusion_matrices,
            performance_output)


if __name__ == '__main__':
    app.run_server(debug=True)
