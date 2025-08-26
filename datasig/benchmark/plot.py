# FIXME: this file has too many pyright errors and needs a fix
# I temproatilly disabled static checking in pyproject.toml

import pandas as pd
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from plotly.subplots import (
    make_subplots,
)  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from typing import TextIO


def plot_results(csv_data: TextIO, key="time_fingerprint"):
    # Colors for fingerprinting methods
    colors = ["#5479ff", "#a179ff", "#ffcd70", "#95e990", "#83fbe4", "#ff7789"]

    # Read the CSV data
    df = pd.read_csv(csv_data)

    # Get unique configs, datasets, and methods
    unique_configs = sorted(df["config_name"].unique())
    unique_datasets = sorted(df["dataset"].unique())
    unique_methods = sorted(df["fingerprint_method"].unique())

    # Create a subplot figure - one row per config value
    fig = make_subplots(
        rows=len(unique_configs),
        cols=1,
        subplot_titles=[f"Config: {config}" for config in unique_configs],
        vertical_spacing=0.3,  # Increased spacing between subplots
    )

    # For each unique config value
    for config_idx, config_value in enumerate(unique_configs):
        # Filter data for this config
        config_df = df[df["config_name"] == config_value]

        # Calculate positions for each dataset group
        group_width = len(unique_methods) + 1  # Width for each dataset group (methods + spacing)
        positions = {}
        dataset_tick_positions = []
        dataset_tick_labels = []

        for i, dataset in enumerate(unique_datasets):
            # Starting position for this dataset group
            start_pos = i * group_width

            # For each method in this dataset, calculate its position
            for j, method in enumerate(unique_methods):
                positions[(dataset, method)] = start_pos + j

            # Store the center position of this dataset group for ticks
            center_pos = i * group_width + (len(unique_methods) - 1) / 2
            dataset_tick_positions.append(center_pos)
            dataset_tick_labels.append(dataset)

        # For each method across all datasets
        for j, method in enumerate(unique_methods):
            # Create separate lists for time and accuracy
            x_positions_time = []
            y_values_time = []
            x_positions_accuracy = []
            y_values_accuracy = []
            custom_data_time = []
            custom_data_accuracy = []

            # Position offset for time and accuracy bars
            time_offset = -0.2  # Slightly to the left
            accuracy_offset = 0.2  # Slightly to the right

            # Collect data for each dataset
            for i, dataset in enumerate(unique_datasets):
                # Base position for this dataset-method combination
                base_pos = positions[(dataset, method)]

                # Filter data for this dataset and method
                data_point = config_df[
                    (config_df["dataset"] == dataset) & (config_df["fingerprint_method"] == method)
                ]

                if not data_point.empty:
                    # Add time data
                    x_positions_time.append(base_pos + time_offset)  # Offset to the left
                    y_values_time.append(data_point["time_fingerprint"].values[0])
                    custom_data_time.append(dataset)

                    # Add accuracy data
                    x_positions_accuracy.append(base_pos + accuracy_offset)  # Offset to the right
                    y_values_accuracy.append(data_point["accuracy_error"].values[0])
                    custom_data_accuracy.append(dataset)

            # Add bars for this method across all datasets
            if x_positions:  # Only add if we have data points
                fig.add_trace(
                    go.Bar(
                        x=x_positions,
                        y=y_values,
                        name=method,
                        marker_color=colors[j % len(colors)],
                        customdata=custom_data,
                        hovertemplate="Dataset: %{customdata}<br>Method: "
                        + method
                        + "<br>Time: %{y}<extra></extra>",
                        # Only show legend for the first config
                        showlegend=True if config_idx == 0 else False,
                    ),
                    row=config_idx + 1,
                    col=1,
                )

        # Set x-axis ticks for dataset names
        x_axis_name = "xaxis" if config_idx == 0 else f"xaxis{config_idx + 1}"
        fig.update_layout(
            **{
                x_axis_name: dict(
                    tickvals=dataset_tick_positions,
                    ticktext=dataset_tick_labels,
                    tickangle=0,
                    tickfont=dict(size=12),
                    showgrid=False,
                )
            }
        )

        # Update y-axis title for this subplot
        y_axis_name = "yaxis" if config_idx == 0 else f"yaxis{config_idx + 1}"
        fig.update_layout(**{y_axis_name: dict(title=key)})

    # Update overall layout
    fig.update_layout(
        title="Execution Time by Config, Dataset, and Method",
        height=450 * len(unique_configs),  # Adjust height based on number of configs
        legend_title="Methods",
        margin=dict(b=50, t=100),  # Adjusted margins
    )

    # Show the plot
    fig.show()


def plot_results2(csv_data: TextIO):
    # Read the CSV data
    df = pd.read_csv(csv_data)

    # Get unique configs, datasets, and methods
    unique_configs = sorted(df["config_name"].unique())
    unique_datasets = sorted(df["dataset"].unique())
    unique_methods = sorted(df["fingerprint_method"].unique())

    # Colors for time and accuracy
    time_colors = ["#5479ff", "#a179ff", "#ffcd70", "#95e990", "#83fbe4", "#ff7789"]
    accuracy_colors = ["#4469ef", "#59009b", "#efbd60", "#85d980", "#83fbe4", "#ff7789"]

    # Create a subplot figure - one row per config value
    fig = make_subplots(
        rows=len(unique_configs),
        cols=1,
        subplot_titles=[f"Config: {config}" for config in unique_configs],
        vertical_spacing=0.3,  # Increased spacing between subplots
        specs=[
            [{"secondary_y": True}] for _ in range(len(unique_configs))
        ],  # Enable secondary y-axis for each subplot
    )

    # For each unique config value
    for config_idx, config_value in enumerate(unique_configs):
        # Filter data for this config
        config_df = df[df["config_name"] == config_value]

        # Calculate positions for each dataset group
        group_width = len(unique_methods) + 1  # Width for each dataset group (methods + spacing)
        positions = {}
        dataset_tick_positions = []
        dataset_tick_labels = []

        for i, dataset in enumerate(unique_datasets):
            # Starting position for this dataset group
            start_pos = i * group_width

            # For each method in this dataset, calculate its position
            for j, method in enumerate(unique_methods):
                positions[(dataset, method)] = start_pos + j

            # Store the center position of this dataset group for ticks
            center_pos = i * group_width + (len(unique_methods) - 1) / 2
            dataset_tick_positions.append(center_pos)
            dataset_tick_labels.append(dataset)

        # For each method across all datasets
        for j, method in enumerate(unique_methods):
            # Create separate lists for time and accuracy
            x_positions_time = []
            y_values_time = []
            x_positions_accuracy = []
            y_values_accuracy = []
            custom_data_time = []
            custom_data_accuracy = []

            # Position offset for time and accuracy bars
            time_offset = -0.2  # Slightly to the left
            accuracy_offset = 0.2  # Slightly to the right

            # Maximum width for bars
            max_width = 0.35  # Maximum width for each bar

            # Collect data for each dataset
            for i, dataset in enumerate(unique_datasets):
                # Base position for this dataset-method combination
                base_pos = positions[(dataset, method)]

                # Filter data for this dataset and method
                data_point = config_df[
                    (config_df["dataset"] == dataset) & (config_df["fingerprint_method"] == method)
                ]

                if not data_point.empty:
                    # Add time data
                    x_positions_time.append(base_pos + time_offset)
                    y_values_time.append(
                        data_point["time_fingerprint"].values[0]
                        + data_point["time_canonization"].values[0]
                    )
                    custom_data_time.append(dataset)

                    # Add accuracy data
                    x_positions_accuracy.append(base_pos + accuracy_offset)
                    y_values_accuracy.append(data_point["accuracy_error"].values[0])
                    custom_data_accuracy.append(dataset)

            # Add bars for time values (primary y-axis)
            if x_positions_time:
                fig.add_trace(
                    go.Bar(
                        x=x_positions_time,
                        y=y_values_time,
                        name=f"{method}",
                        marker_color=time_colors[j],
                        customdata=custom_data_time,
                        width=max_width,
                        hovertemplate="Dataset: %{customdata}<br>Method: "
                        + method
                        + "<br>Time: %{y}<extra></extra>",
                        showlegend=True if config_idx == 0 else False,
                    ),
                    row=config_idx + 1,
                    col=1,
                    secondary_y=False,  # Use primary y-axis for time
                )

            # Add bars for accuracy values (secondary y-axis)
            if x_positions_accuracy:
                fig.add_trace(
                    go.Bar(
                        x=x_positions_accuracy,
                        y=y_values_accuracy,
                        # name=f"{method} - Accuracy error",
                        marker_color=time_colors[j],
                        marker_pattern_shape="/",
                        customdata=custom_data_accuracy,
                        width=max_width,
                        hovertemplate="Dataset: %{customdata}<br>Method: "
                        + method
                        + "<br>Accuracy error: %{y}<extra></extra>",
                        showlegend=False,
                        # showlegend=True if config_idx == 0 else False
                    ),
                    row=config_idx + 1,
                    col=1,
                    secondary_y=True,  # Use secondary y-axis for accuracy
                )

        # Set x-axis ticks for dataset names
        x_axis_name = "xaxis" if config_idx == 0 else f"xaxis{config_idx + 1}"
        fig.update_layout(
            **{
                x_axis_name: dict(
                    tickvals=dataset_tick_positions,
                    ticktext=dataset_tick_labels,
                    tickangle=0,
                    tickfont=dict(size=12),
                    showgrid=False,
                )
            }
        )

        # Update y-axis titles for this subplot
        y_axis_name = "yaxis" if config_idx == 0 else f"yaxis{config_idx + 1}"
        sec_y_axis_name = "yaxis2" if config_idx == 0 else f"yaxis{config_idx + 1}2"

        fig.update_layout(
            **{
                y_axis_name: dict(title="Time", side="left"),
                sec_y_axis_name: dict(
                    title="Accuracy", side="right", overlaying=y_axis_name.replace("axis", "")
                ),
            }
        )

    # Update overall layout
    fig.update_layout(
        title="Fingerprint Time and Accuracy Error",
        height=450 * len(unique_configs),  # Adjust height based on number of configs
        legend_title="Metrics",
        margin=dict(b=50, t=100),  # Adjusted margins
        barmode="group",  # Group bars together
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Show the plot
    fig.show()
