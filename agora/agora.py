from pathlib import Path
import logging
import sys
from typing import Tuple, Dict

import click
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

import torch
from torch import Tensor
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
import torch_geometric.transforms as T

from sklearn.metrics import classification_report
from pudb import set_trace as st

__version__ = "Alpha 0.5"
log = None

from node_model import NodeModel
from edge_model import EdgeModel
from graphdataset import GraphDataset

from flask import Flask, request, jsonify, after_this_request, send_file, render_template
import json
import os
import tempfile
import shutil

################################################################################
# GNN code 
################################################################################

def _generate_employee_tensor(df: DataFrame) -> Tuple[Tensor, pd.core.arrays.categorical.CategoricalAccessor]:
    df = df.sort_values('idx')
    df = df.astype({'department': 'category', 'remote': int, 'fulltime': int})
    df = df.assign(dep_codes = df['department'].cat.codes)

    node_features = torch.tensor(df[['dep_codes', 'remote', 'fulltime']].values, dtype=torch.float)

    return node_features, df['department'].cat

def _generate_comm_graph(df: DataFrame, employeesIdx: pd.Series, node_features: Tensor, date: str,
    force_undirected: bool = False, enriched_node_features=False) -> torch_geometric.data.data.Data:
    log.debug(f"Generate graph for date {date} of shape {df.shape}")

    edge_index = torch.tensor([df['sender'].values, df['recver'].values], dtype=torch.long)
    # Flip edge direction
    if force_undirected:
        log.debug('Forcing undirected graph by having a reverse edge for each normal edge.')
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    edge_features = torch.tensor(df[['slack', 'email', 'zoom']].values, dtype=torch.float)

    if enriched_node_features:
        log.info('Enriching node features with aggregated edge features ...')
        degree = df.groupby('sender')['recver'].nunique()
        degree.name = 'degree'
        channel_interactions = df.groupby('sender')[['slack', 'email', 'zoom']].sum()
        total_intactions = channel_interactions.sum(axis=1)
        total_intactions.name = 'total_interactions'
        normalized_channel_interactions = channel_interactions.div(total_intactions, axis=0)
        agg_edge_features = pd.concat([degree, total_intactions, normalized_channel_interactions], axis=1)
        # Fill missing node entries with 0.0 values
        agg_edge_features = pd.DataFrame(employeesIdx).join(agg_edge_features).fillna(0.0).drop(columns='idx')

        # Extend the node tensor with the aggregated edge information
        tef = torch.tensor(agg_edge_features.values, dtype=torch.float)
        node_features = torch.cat([node_features, tef], axis=1)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    return data

def _load_data(comm: DataFrame, employees: DataFrame, enriched_node_features=False) -> Dict[str, Tensor] :
    """
    Load communication and employee data, returning a separate tensor for each day
    """
    log.info(f'Loading employee table {employees.shape}')
    node_features, department_encoding = _generate_employee_tensor(employees)

    log.info(f'Loading communication table {comm.shape}')
    
    # Cleanup input data (remove empty interactions, sum duplicate entries)
    comm = comm.query('interactions != 0')
    comm = comm.groupby(['sender', 'recver', 'channel', 'date'], as_index=False).sum()

    # Make zoom, slack, email its own table, and fill empty interactions with 0
    comm = comm.pivot(columns='channel', values=['interactions'], index=['sender', 'recver', 'date']).fillna(0).reset_index()
    # Fix column names after pivot
    comm.columns = ['sender', 'recver', 'date'] + comm.columns.droplevel()[3:].tolist()

    # Split data by date, generating its own data/graph for every day
    datasets = {}
    for date, df in comm.groupby('date'):
        datasets[date] = _generate_comm_graph(df, employees['idx'], node_features, date, enriched_node_features=enriched_node_features)

    log.info(f'Loaded {len(datasets)} datasets ...')
    return datasets

def _train_edge_model(comm: Path, employees: Path):
    # Convert csv input data into torch_geometric.data.Data
    comm_df = pd.read_csv(comm)
    employees_df = pd.read_csv(employees)
    labels = employees_df['hasCommunicationIssues'].astype(int).values

    datasets = _load_data(comm_df, employees_df.drop(columns='hasCommunicationIssues'))

    data = datasets[list(datasets.keys())[0]]
    # Create a model for the Cora dataset
    model = EdgeModel(data)

    for epoch in range(1, 10):
        trn_loss, val_loss = model.train_one_epoch()
        log.info(f'Epoch: {epoch:03d}, Train: {trn_loss:.4f}, Val: {val_loss:.4f}')

    acc_val, pred_val = model.test()
    print(f'test prediction: {pred_val}')


def _train_node_model(ctx, comm: Path, employees: Path, output_path: Path, epochs: int, model_prefix: str = '', enriched_node_features=False):
    # Convert csv input data into torch_geometric.data.Data
    comm_df = pd.read_csv(comm)
    employees_df = pd.read_csv(employees)
    labels = employees_df['hasCommunicationIssues'].astype(int).values

    dataset_per_day = _load_data(comm_df, employees_df.drop(columns='hasCommunicationIssues'), enriched_node_features=enriched_node_features)

    # Only use first day
    gData = {date: GraphDataset(data, training=True, labels=labels) for date, data in dataset_per_day.items()}
    first_date = list(gData.keys())[0]

    # Create a model for the Cora dataset
    model = NodeModel(gData[first_date].num_features, nClasses=2)

    for epoch in range(epochs):
        for date, data in gData.items():
            trn_loss, val_loss = model.train_one_epoch(data)
            log.info(f'Epoch: {epoch:03d}, Day: {date}, Training Loss: {trn_loss:0.3f}, Validation Loss: {val_loss:0.3f}')
        
    #trn_acc, val_acc = model.test(gData)
    #log.info(f'Final: Training Acc: {trn_acc:0.3f}, Validation Acc: {val_acc:0.3f}')

    # save model
    oPath = output_path.joinpath(model_prefix+'model.pkl')
    log.info(f'Saving trained model to {oPath} ...')
    model.save(oPath)


def _predict_node(model_ckp: Path, comm: Path, employees: Path, output_path: Path,
    output_prefix: str = '', day: int = 1, enriched_node_features: bool = False):
    # Load previously saved model
    model = NodeModel.load(model_ckp)

    # Convert csv input data into torch_geometric.data.Data
    comm_df = pd.read_csv(comm)
    employees_df = pd.read_csv(employees)

    dataset_per_day = _load_data(comm_df, employees_df, enriched_node_features=enriched_node_features)
    raw_data = dataset_per_day[list(dataset_per_day.keys())[day]]

    gData = GraphDataset(raw_data, training=False)

    prediction = model.predict(gData)
    employees_df = employees_df.assign(hasCommunicationIssues=prediction)
    employees_df['hasCommunicationIssues'] = employees_df['hasCommunicationIssues'].astype(bool)
    oPath = output_path.joinpath(output_prefix + 'prediction.csv')
    log.info(f'Writing prediction to {oPath} ...')
    employees_df.to_csv(oPath, index=False)


def _basemodel(comm: Path, employees: Path, output_path: Path, output_prefix: str = '', day: int = 1):
    # Convert csv input data into torch_geometric.data.Data
    comm_df = pd.read_csv(comm)
    employees_df = pd.read_csv(employees)
    dataset_per_day = _load_data(comm_df, employees_df)
    date = list(dataset_per_day.keys())[day]
    data = comm_df.query('date == @date')

    # Identify employees who's total interactions are very low compared to all

    total_interactions = data.groupby('sender')['interactions'].sum()
    # Use the second bin of a fixed histogram (5 bins) as the threshold
    threshold = np.histogram(total_interactions, bins=5)[1][1]
    pred = total_interactions < threshold

    # Join result to employee dataframe
    pred.name = 'hasCommunicationIssues'
    pred = pred.reset_index()
    if 'hasCommunicationIssues' in employees_df:
        employees_df = employees_df.drop(columns=['hasCommunicationIssues'])
    result = pd.merge(left=employees_df, right=pred, left_on='idx', right_on='sender', how='outer')
    result = result.drop(columns=['sender'])
    # If there is no communication data for an employee, make sure we mark them
    result = result.fillna('True')

    oPath = output_path.joinpath(output_prefix + 'prediction.csv')
    log.info(f'Writing prediction to {oPath} ...')
    result.to_csv(oPath, index=False)

def test_predictions(truelabel, predictions):
    labels = pd.read_csv(truelabel).set_index('idx')['hasCommunicationIssues']
    preds = [pd.read_csv(prediction).set_index('idx')['hasCommunicationIssues'] for prediction in predictions]

    for prediction, filename in zip(preds, predictions):
        print(f'Prediction: {filename}\n\
        {classification_report(labels, prediction)}')


################################################################################
# Flask
################################################################################
def _deploy(model_path: str):
    # Name of the apps module package
    app = Flask(__name__)
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    # Load in our meta_data
    f = open("./meta_data.txt", "r")
    load_meta_data = json.loads(f.read())


    # Meta data endpoint
    @app.route('/', methods=['GET'])
    def meta_data():
        #return jsonify(load_meta_data)
        return render_template('index.html')

    @app.route('/predict', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            file_com = request.files['file_com']
            file_emp = request.files['file_emp']
            tmp = tempfile.mkdtemp()
            print(f'Creating temp file in {tmp} ...')
            file_com.save(os.path.join(tmp, file_com.filename))
            file_emp.save(os.path.join(tmp, file_emp.filename))

            _predict_node(Path(model_path),
                                Path(os.path.join(tmp, file_com.filename)),
                                Path(os.path.join(tmp, file_emp.filename)),
                                output_path=Path(tmp), day=0)

            @after_this_request
            def cleanup(response):
                shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
                return response

            return send_file(os.path.join(tmp, 'prediction.csv'), mimetype='file/text')
        else:
            return render_template('predict.html')

    return app

################################################################################
# Auxillary functions
################################################################################

def getLogger(module_name, filename, stdout=None):
    format = '%(asctime)s [%(name)s:%(levelname)s] %(message)s'
    logging.basicConfig(filename=filename,
                        level=logging.DEBUG,
                        filemode='a',
                        format=format,
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(module_name)

    # Add handler for stdout
    if stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(stdout)
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

@click.group()
@click.version_option(__version__)
@click.option('-v', '--verbose', count=True, show_default=True)
@click.option('-l', '--logfile', default=f'agora.log', show_default=True)
@click.pass_context
def agora(ctx, verbose, logfile):
    global log
    ctx.obj['workingdir'] = Path('.').absolute()
    loglevel = logging.WARNING
    if verbose > 1:
        loglevel = logging.INFO
    if verbose >= 2:
        loglevel = logging.DEBUG
    log = getLogger(__name__, ctx.obj['workingdir'].joinpath(logfile), stdout=loglevel)

################################################################################

################################################################################
# Input handlers
################################################################################

@agora.group('train')
@click.pass_context
def train(ctx):
    pass

@train.command()
@click.pass_context
@click.argument('communication', type=click.Path(exists=True))
@click.argument('employees', type=click.Path(exists=True))
@click.option('-e','--epochs', type=int, default=10, help='Number of epochs to train model')
@click.option('-o','--output', type=click.Path(exists=False), default='.', help='Path to store trained model')
@click.option('-p','--prefix', type=str, default='', help='Prefix used when storing the pickled model')
@click.option('-r','--enrich', is_flag=True, default=False, help="Add aggregated edge features to node features")
def node(ctx, communication, employees, epochs, output, prefix, enrich):

    # Commit changes
    _train_node_model(ctx,
        Path(communication).absolute(),
        Path(employees).absolute(),
        Path(output).absolute(),
        epochs,
        model_prefix=prefix,
        enriched_node_features=enrich)

@train.command()
@click.pass_context
@click.argument('communication', type=click.Path(exists=True))
@click.argument('employees', type=click.Path(exists=True))
def edge(ctx, communication, employees):

    # Commit changes
    _train_edge_model(
        Path(communication).absolute(),
        Path(employees).absolute())


@agora.group('predict')
@click.pass_context
def predict(ctx):
    pass

@predict.command()
@click.pass_context
@click.argument('model_ckp', type=click.Path(exists=True))
@click.argument('communication', type=click.Path(exists=True))
@click.argument('employees', type=click.Path(exists=True))
@click.option('-d','--day', type=int, default=1, help='What day to use prediction')
@click.option('-o','--output', type=click.Path(exists=False), default='.', help='Path to store prediction')
@click.option('-p','--prefix', type=str, default='', help='Prefix used when storing result')
@click.option('-r','--enrich', is_flag=True, default=False, help="Add aggregated edge features to node features")
def node(ctx, model_ckp, communication, employees, day, output, prefix, enrich):

    # Commit changes
    _predict_node(
        Path(model_ckp).absolute(),
        Path(communication).absolute(),
        Path(employees).absolute(),
        output_path=Path(output).absolute(),
        output_prefix=prefix,
        day=day,
        enriched_node_features=enrich)

@predict.command()
@click.pass_context
@click.argument('communication', type=click.Path(exists=True))
@click.argument('employees', type=click.Path(exists=True))
@click.option('-d','--day', type=int, default=1, help='What day to use prediction')
@click.option('-o','--output', type=click.Path(exists=False), default='.', help='Path to store prediction')
@click.option('-p','--prefix', type=str, default='', help='Prefix used when storing result')
def basemodel(ctx, communication, employees, day, output, prefix):

    # Commit changes
    _basemodel(
        Path(communication).absolute(),
        Path(employees).absolute(),
        output_path=Path(output).absolute(),
        output_prefix=prefix,
        day=day)


@predict.command()
@click.pass_context
@click.argument('communication', type=click.Path(exists=True))
@click.argument('employees', type=click.Path(exists=True))
def edge(ctx, communication, employees):
    raise NotImplementedError

    # Commit changes
    _predict_edge(ctx,
        Path(communication).absolute(),
        Path(employees).absolute())

@agora.group('test')
@click.pass_context
def test(ctx):
    pass

@test.command()
@click.pass_context
@click.argument('truelabel', type=click.Path(exists=True))
@click.argument('predictions', type=click.Path(exists=True), nargs=-1)
def node(ctx, truelabel, predictions):
    test_predictions(truelabel, predictions)

@agora.group('deploy')
@click.pass_context
@click.option('--port', default=5000, help='Port to serve model')
@click.option('--debug', default=True, help='Enable flask debugging', is_flag=True)
def deploy(ctx, port: int, debug: bool):
    ctx.obj['port'] = port
    ctx.obj['debug'] = debug
    pass

@deploy.command()
@click.pass_context
@click.argument('model_path', type=click.Path(exists=True))
def flask(ctx, model_path):
    app = _deploy(model_path)
    app.run(host='0.0.0.0', port=ctx.obj['port'], debug=ctx.obj['debug'])



################################################################################

# Create a main that is used by setup.cfg as console_script entry point
def main():
    agora(obj={})

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()