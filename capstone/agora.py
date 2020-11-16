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

__version__ = 0.0
log = None

from node_model import NodeModel

################################################################################
# GNN code 
################################################################################

def _generate_employee_tensor(df: DataFrame) -> Tuple[Tensor, pd.core.arrays.categorical.CategoricalAccessor]:
    df = df.sort_values('idx')
    df = df.astype({'department': 'category', 'remote': int, 'fulltime': int})
    df = df.assign(dep_codes = df['department'].cat.codes)

    node_features = torch.tensor(df[['dep_codes', 'remote', 'fulltime']].values, dtype=torch.float)

    return node_features, df['department'].cat

def _generate_comm_graph(df: DataFrame, node_features: Tensor, date: str) -> torch_geometric.data.data.Data:
    log.debug(f"Generate graph for date {date} of shape {df.shape}")

    edge_index = torch.tensor([df['sender'].values, df['recver'].values], dtype=torch.long)
    edge_features = torch.tensor(df[['slack', 'email', 'zoom']].values, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    return data

def _load_data(comm: DataFrame, employees: DataFrame) -> Dict[str, Tensor] :

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

def _train_edge_model(ctx, comm: Path, employees: Path):
    raise NotImplementedError()

def _train_node_model(ctx, comm: Path, employees: Path):
    # Convert csv input data into torch_geometric.data.Data
    comm_df = pd.read_csv(comm)
    employees_df = pd.read_csv(employees)
    datasets = _load_data(comm_df, employees_df) 

    data = datasets[list(datasets.keys())[0]]
    # Create a model for the Cora dataset
    model = NodeModel(data, K=1)

    for epoch in range(1, 10):
        model.train_one_epoch()
        fmt = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
        log.info(fmt.format(epoch, *(model.test())))

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
@click.option('-l', '--logfile', default=f'log.log', show_default=True)
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
def node(ctx, communication, employees):

    # Commit changes
    _train_node_model(ctx,
        Path(communication).absolute(),
        Path(employees).absolute())

@train.command()
@click.pass_context
@click.argument('communication', type=click.Path(exists=True))
@click.argument('employees', type=click.Path(exists=True))
def edge(ctx, communication, employees):

    # Commit changes
    _train_edge_model(ctx,
        Path(communication).absolute(),
        Path(employees).absolute())


@agora.group('predict')
@click.pass_context
def predict(ctx):
    pass

@predict.command()
@click.pass_context
@click.argument('communication', type=click.Path(exists=True))
@click.argument('employees', type=click.Path(exists=True))
def node(ctx, communication, employees):
    raise NotImplementedError

    # Commit changes
    _predict_node(ctx,
        Path(communication).absolute(),
        Path(employees).absolute())

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