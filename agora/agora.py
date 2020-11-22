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

from pudb import set_trace as st

__version__ = "Alpha 0.5"
log = None

from node_model import NodeModel
from edge_model import EdgeModel
from graphdataset import GraphDataset

################################################################################
# GNN code 
################################################################################

def _generate_employee_tensor(df: DataFrame) -> Tuple[Tensor, pd.core.arrays.categorical.CategoricalAccessor]:
    df = df.sort_values('idx')
    df = df.astype({'department': 'category', 'remote': int, 'fulltime': int})
    df = df.assign(dep_codes = df['department'].cat.codes)

    node_features = torch.tensor(df[['dep_codes', 'remote', 'fulltime']].values, dtype=torch.float)

    return node_features, df['department'].cat

def _generate_comm_graph(df: DataFrame, node_features: Tensor, date: str, force_undirected: bool = False) -> torch_geometric.data.data.Data:
    log.debug(f"Generate graph for date {date} of shape {df.shape}")

    edge_index = torch.tensor([df['sender'].values, df['recver'].values], dtype=torch.long)
    # Flip edge direction
    if force_undirected:
        log.debug('Forcing undirected graph by having a reverse edge for each normal edge.')
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    edge_features = torch.tensor(df[['slack', 'email', 'zoom']].values, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    return data

def _load_data(comm: DataFrame, employees: DataFrame) -> Dict[str, Tensor] :
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
        datasets[date] = _generate_comm_graph(df, node_features, date)

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


def _train_node_model(ctx, comm: Path, employees: Path, output_path: Path, epochs: int, model_prefix: str = ''):
    # Convert csv input data into torch_geometric.data.Data
    comm_df = pd.read_csv(comm)
    employees_df = pd.read_csv(employees)
    labels = employees_df['hasCommunicationIssues'].astype(int).values

    dataset_per_day = _load_data(comm_df, employees_df.drop(columns='hasCommunicationIssues'))

    # Only use first day
    raw_data = dataset_per_day[list(dataset_per_day.keys())[0]]

    gData = GraphDataset(raw_data, training=True, labels=labels)

    # Create a model for the Cora dataset
    model = NodeModel(gData.num_features, nClasses=2)

    for epoch in range(epochs):
        trn_loss, val_loss = model.train_one_epoch(gData)
        log.info(f'Epoch: {epoch:03d}, Training Loss: {trn_loss:0.3f}, Validation Loss: {val_loss:0.3f}')
       
    trn_acc, val_acc = model.test(gData)
    log.info(f'Final: Training Acc: {trn_acc:0.3f}, Validation Acc: {val_acc:0.3f}')

    # save model
    oPath = output_path.joinpath(model_prefix+'model.pkl')
    log.info(f'Saving trained model to {oPath} ...')
    model.save(oPath)


def _predict_node(model_ckp: Path, comm: Path, employees: Path, output_path: Path, output_prefix: str = '', day: int = 1):
    # Load previously saved model
    model = NodeModel.load(model_ckp)

    # Convert csv input data into torch_geometric.data.Data
    comm_df = pd.read_csv(comm)
    employees_df = pd.read_csv(employees)

    dataset_per_day = _load_data(comm_df, employees_df)
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

    employees_df.loc[pred.index]['hasCommunicationIssues'] = pred

    oPath = output_path.joinpath(output_prefix + 'prediction.csv')
    log.info(f'Writing prediction to {oPath} ...')
    employees_df.to_csv(oPath, index=False)


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
def node(ctx, communication, employees, epochs, output, prefix):

    # Commit changes
    _train_node_model(ctx,
        Path(communication).absolute(),
        Path(employees).absolute(),
        Path(output).absolute(),
        epochs,
        model_prefix=prefix)

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
def node(ctx, model_ckp, communication, employees, day, output, prefix):

    # Commit changes
    _predict_node(
        Path(model_ckp).absolute(),
        Path(communication).absolute(),
        Path(employees).absolute(),
        output_path=Path(output).absolute(),
        output_prefix=prefix,
        day=day)

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

################################################################################

# Create a main that is used by setup.cfg as console_script entry point
def main():
    agora(obj={})

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()