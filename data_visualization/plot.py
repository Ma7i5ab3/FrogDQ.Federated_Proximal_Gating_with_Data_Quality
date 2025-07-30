import plotly.graph_objects as go
from data_statistics.Stats import Stats
import numpy as np
import os
import torch


def plot_weights_over_rounds(stats_client: Stats, fl_algorithm: str, dq_type: str, type: str, experiment_name: str, iteration: int):

    # Select if plotting gates or weights
    weights_per_round_lst = stats_client.get_gate_updates(algorithm=fl_algorithm, dq_type=dq_type)[-1] if type == 'gate' else stats_client.get_weights_updates(algorithm=fl_algorithm, dq_type=dq_type)[-1] 

    weights_tensor = torch.stack(weights_per_round_lst).numpy()
    fig = go.Figure()

    marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star']

    for _, i in enumerate(stats_client.topk_features[-1]):
        fig.add_trace(
            go.Scatter(
                y=weights_tensor[:, i],
                mode="lines+markers",
                name=f"{type} {i}",
                marker=dict(
                    symbol=marker_symbols[i % len(marker_symbols)],
                    size=6
                )
            ),
        )

    fig.update_layout(
        title=f"{type} evolution over rounds - {dq_type}",
        xaxis_title="Round",
        yaxis_title="Values",
        template="simple_white",
    )

    os.makedirs(f'data_visualization/graphics/{experiment_name}/{iteration}/', exist_ok=True)

    # Save as PNG
    fig.write_image(
        f"data_visualization/graphics/{experiment_name}/{iteration}/{type}_evolution_{dq_type}_{fl_algorithm}.png", width=800, height=600, scale=1
    )



def plot_compare_weights_over_rounds(stats_client: Stats, corr_features: list, experiment_name: str, iteration: int):
    """
    Plot the evolution of weights/gates for both dirty and clean stats clients.
    Same color for the same feature, different marker symbol for clean/dirty.
    """

    weights_tensor_dirty = torch.stack(stats_client.get_gate_updates(algorithm='frog', dq_type='dirty')[-1]).numpy()
    weights_tensor_clean = torch.stack(stats_client.get_gate_updates(algorithm='frog', dq_type='clean')[-1]).numpy()

    marker_symbols = {'clean': 'circle', 'dirty': 'x'}
    '''colors = [f"rgba({r},{g},{b},1)" for r, g, b in [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194)
    ]]'''

    fig = go.Figure()

    for idx, i in enumerate(corr_features):
        # Clean
        fig.add_trace(
            go.Scatter(
                y=weights_tensor_clean[:, i],
                mode="lines",
                name=f"Gate {i} (clean)",
                line=dict(dash='solid'),
            )
        )
        # Dirty
        fig.add_trace(
            go.Scatter(
                y=weights_tensor_dirty[:, i],
                mode="lines",
                name=f"Gate {i} (dirty)",
                line=dict(dash='dot'),
            )
        )

    fig.update_layout(
        title=f"Gates Evolutions over Rounds: Clean Feature (q = 1) vs Dirty Feature (q = 0)",
        xaxis_title="Round",
        yaxis_title="Values",
        template="simple_white",
    )

    os.makedirs(f'data_visualization/graphics/{experiment_name}/{iteration}/', exist_ok=True)

    # Save as PNG
    fig.write_image(
        f"data_visualization/graphics/{experiment_name}/{iteration}/gate_evolution_compare_frog.png", width=800, height=600, scale=1
    ) 


def plot_accuracy_test_comparison(stats_client: Stats, experiment_name: str):
    """
    Plot the accuracy distributions for four methodologies as box plots, with mean and variance annotated.
    Args:
        stats_client: Stats. Client that manages statistical results.
    """
    
    fig = go.Figure()

    for i, sim_info in enumerate(stats_client.data.keys()):
        accs = stats_client.get_accuracy_test(algorithm=sim_info[0], dq_type=sim_info[1])
        mean = np.mean(accs)
        var = np.var(accs)
        fig.add_trace(
            go.Box(
                y=accs,
                name=f"{sim_info[0]}_{sim_info[1]}",
                boxmean=True,  # Show mean
                boxpoints='all',  # Show all points
                jitter=0.3,
                pointpos=-1.8,
            )
        )
        # Annotate mean and variance    
        fig.add_annotation(
            x=f"{sim_info[0]}_{sim_info[1]}",
            y=mean,
            text=f"Mean: {mean:.3f}",
            showarrow=False,
            yshift=20,
            font=dict(size=8, color="gray"),
        )

    fig.update_layout(
        title="Accuracy Comparison",
        xaxis_title="Methodology",
        yaxis_title="Accuracy",
        template="simple_white",
    )

    os.makedirs(f'data_visualization/graphics/{experiment_name}/', exist_ok=True)

    fig.write_image(
        f"data_visualization/graphics/{experiment_name}/accuracy_comparison.png", width=800, height=600, scale=1
    )


def plot_accuracy_val_over_rounds(stats_client: Stats, experiment_name: str, iteration: int):

    line_style = {'clean': 'solid', 'dirty': 'dot'}
    fig = go.Figure()

    for i, sim_info in enumerate(stats_client.data.keys()):

        fig.add_trace(
            go.Scatter(
                y=stats_client.get_accuracy_val(algorithm=sim_info[0], dq_type=sim_info[1]),
                mode="lines",
                name=f"{sim_info[0]} - {sim_info[1]}",
                line=dict(dash=line_style[sim_info[1]]),
            )
        )
    
    fig.update_layout(
        title="Accuracy on Validation Set over rounds",
        xaxis_title="Rounds",
        yaxis_title="Accuracy",
        template="simple_white",
    )

    os.makedirs(f'data_visualization/graphics/{experiment_name}/{iteration}/', exist_ok=True)

    fig.write_image(
        f"data_visualization/graphics/{experiment_name}/{iteration}/accuracy_rounds.png", width=800, height=600, scale=1
    )


def plot_compare_weights_original_model(stats_client: Stats, w_true: list, dq_type: str, feature: int, corrupted: bool, experiment_name: str, iteration: int):

    weights_tensor_fedavg = torch.stack(stats_client.get_weights_updates(algorithm='fedavg', dq_type=dq_type)[-1]).numpy()
    gates_tensor_frog = torch.stack(stats_client.get_gate_updates(algorithm='frog', dq_type=dq_type)[-1]).numpy()
    weights_tensor_frog = torch.stack(stats_client.get_weights_updates(algorithm='frog', dq_type=dq_type)[-1]).numpy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=np.full(len(weights_tensor_fedavg[:, feature]), (w_true[feature].numpy())),
            mode="lines",
            name=f"w_true",
            line=dict(dash='dash'),
        )
    )

    fig.add_trace(
        go.Scatter(
            y=weights_tensor_fedavg[:, feature],
            mode="lines",
            name=f"Fedavg",
            line=dict(dash='solid'),
        )
    )

    fig.add_trace(
        go.Scatter(
            y=weights_tensor_frog[:, feature]*gates_tensor_frog[:, feature],
            mode="lines",
            name=f"Frog",
            line=dict(dash='solid'),
        )
    )
    
    fig.update_layout(
        title=f"Values {'corrupted' if corrupted else 'clean'} feature {feature} on dirty setting",
        xaxis_title="Rounds",
        yaxis_title="Weight Value",
        template="simple_white",
    )

    os.makedirs(f'data_visualization/graphics/{experiment_name}/{iteration}/', exist_ok=True)

    fig.write_image(
        f"data_visualization/graphics/{experiment_name}/{iteration}/weight_comparison_on_feature_{feature}_{dq_type}_setting.png", width=800, height=600, scale=1
    )