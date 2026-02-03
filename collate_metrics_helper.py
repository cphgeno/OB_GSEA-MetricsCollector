import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_top1_metrics_from_files(file_list, dataset_name):
    """
    Load top-1 metrics from list of files and concatenate.
    
    Returns:
        df_overall: DataFrame with overall metrics
        df_overall_NA: DataFrame with NA-values summary for tool
        metrics_dict: Dict of {tool_name: label_class_df} to get performance for each labbel class
    """
    #Initialize
    df_overall = pd.DataFrame()
    df_overall_NA = pd.DataFrame()
    metrics_dict = {}
    
    #Loop over the file path to each tool
    for file_path in file_list:
        filename_top1 = f"{dataset_name}-top1metrics.tsv"
        filename_top1_label_classes = f"{dataset_name}-top1metrics-label_classes.tsv"
        filename_NAs = f"{dataset_name}-NA_data.tsv"

        df_NA = pd.read_csv(f"{file_path}/{filename_NAs}", sep="\t", index_col=0)
        if int(df_NA['percentage_NA_values']) == 1:
            continue

        #Load overall metrics
        df_tool = pd.read_csv(f"{file_path}/{filename_top1}", sep="\t", index_col=0)
        
        if df_tool.empty:
            print(f"Warning: {file_path} is empty, skipping...")
            continue
            
        tool_name = df_tool.index[0]
        
        #Load label-class specific metrics if requested
        df_label_classes = pd.read_csv(f"{file_path}/{filename_top1_label_classes}", sep="\t", index_col=0)
        metrics_dict[tool_name] = df_label_classes

        #Load total samples we have computed NES scores for, for given tool
        total_number_of_samples = df_label_classes["Count"].sum()
        df_tool["Count"] = total_number_of_samples
        
        # df_NA = pd.read_csv(f"{file_path}/{filename_NAs}", sep="\t", index_col=0)
        
        #Merge metrics obtained for each tool
        df_overall = pd.concat([df_overall, df_tool])
        df_overall_NA = pd.concat([df_overall_NA, df_NA])
    
    return df_overall, df_overall_NA, metrics_dict

def load_auc_metrics_from_files(file_list, dataset_name):
    """
    Load AUC metrics from list of files and concatenate.
    
    Returns:
        df_overall: DataFrame with overall metrics
        metrics_dict: Dict of {tool_name: label_class_df} to get performance for each labbel class
    """
    #Initialize
    df_overall = pd.DataFrame()
    metrics_dict = {}
    
    #Loop over the file path to each tool
    for file_path in file_list:
        filename_top1 = f"{dataset_name}-aucmetrics.tsv"
        filename_top1_label_classes = f"{dataset_name}-aucmetrics_label_classes.tsv"

        #Load overall metrics
        df_tool = pd.read_csv(f"{file_path}/{filename_top1}", sep="\t", index_col=0)
        
        if df_tool.empty:
            print(f"Warning: {file_path} is empty, skipping...")
            continue
            
        tool_name = df_tool.index[0]
        
        #Load label-class specific metrics if requested
        try:
            df_label_classes = pd.read_csv(f"{file_path}/{filename_top1_label_classes}", sep="\t", index_col=0)
        except pd.errors.EmptyDataError:
            return np.nan, np.nan
        metrics_dict[tool_name] = df_label_classes
        
        #Merge metrics obtained for each tool
        df_overall = pd.concat([df_overall, df_tool])

    return df_overall, metrics_dict

def get_all_metrics(files_top1, files_auc, dataset_name):
    """ 
    Get all metrics in proper formats to save and to plot. 

    Returns: 
        df_metrics: Dataframe with overall merged metrics
        top1_metrics_tools_dict: Dict of {tool_name: label_class_df} to get top-1 performance for each labbel class
        auc_metrics_tools_dict: Dict of {tool_name: label_class_df} to get top1 performance for each labbel class
    """

    #Load metrics
    df_top1, df_na, top1_metrics_tools_dict = load_top1_metrics_from_files(files_top1, dataset_name)
    df_auc, auc_metrics_tools_dict = load_auc_metrics_from_files(files_auc, dataset_name)

    if isinstance(df_auc, pd.DataFrame):
        #Merge all "overall" metrics and save to tsv file
        df_metrics = pd.concat([df_top1, df_auc, df_na], axis=1, join='outer')
        return df_metrics, top1_metrics_tools_dict, auc_metrics_tools_dict
    elif np.isnan(df_auc):
        return pd.concat([df_top1, df_na], axis=1, join='outer'), np.nan, np.nan
    else:
        raise('Not working!')


def format_tool_name(tool_name):
    """
    Convert toolname string: name1_name2 -> name1 (name2).
    """
    if '_' in tool_name:
        parts = tool_name.split('_', 1)
        return f"{parts[0]} ({parts[1]})"
    return tool_name


#Create mapping for display names
def format_metric_name(metric, mcc_scaled = False):
    """
    Convert metric names to display format for plots.
    """
    if '_Macro' in metric:
        base = metric.replace('_Macro', '')
        return f'{base} Macro average'
    elif '_Weighted' in metric:
        base = metric.replace('_Weighted', '')
        return f'{base} Weighted'
    elif mcc_scaled and metric == 'MCC':
        return '(MCC+1)/2'
    else:
        return metric


def plot_metrics_with_annotations(df_metrics, output_dir, dataset_name, file_basename, metrics_to_plot=None, single_label_class = False, tool_colors = None):
    """
    Plot all tools together with annotations in x-axis labels.
    
    Args:
        df_metrics: DataFrame with metrics
        output_dir: Output directory for plots
        metrics_to_plot: List of metric columns to plot (if we only want specific ones and not all)
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['F1_Weighted', 'Precision_Weighted', 'Recall_Weighted', 'MCC', 'AUC_Weighted']

        #Filter out metrics where ALL values are NA (e.g. AUC for datasets with few samples)
        metrics_to_plot = [m for m in metrics_to_plot if m in df_metrics.columns and not df_metrics[m].isna().all()]

    #Identify reference (most common/frequent) sample size if sample sizes differ (they should not)
    reference_count = df_metrics['Count'].mode()[0]
    
    #Create tool labels with annotations and formatted tool names
    tool_labels = []
    tool_annotations = {}
    for idx, row in df_metrics.iterrows():
        label = format_tool_name(idx)
        annotations = []
        
        #Add NA annotation if any NAs are present for tool
        if row['percentage_NA_values'] > 0:
            annotations.append(f"NA={row['percentage_NA_values']:.1%}")
        
        #Add sample size difference annotation if sample size differs for tool
        if row['Count'] != reference_count:
            diff = int(row['Count'] - reference_count)
            sign = '+' if diff > 0 else ''
            annotations.append(f"*n{sign}={diff}")
        
        tool_labels.append(label)
        tool_annotations[idx] = ', '.join(annotations) if annotations else ''
    
    
    # Split into first three and last two
    metrics_first = metrics_to_plot[:3]
    metrics_second = metrics_to_plot[3:]

    fig, ax = plt.subplots(figsize=(max(12, len(metrics_to_plot) * 1.5), 8))

    # Compute x positions with a gap between groups
    gap = 0.5  # one empty slot as visual separator
    x_first = np.arange(len(metrics_first))  # e.g., [0,1,2]
    x_second = np.arange(len(metrics_second)) + len(metrics_first) + gap  # e.g., [4,5] if n1=3, gap=1

    # Prepare bar width
    width = 0.8 / len(df_metrics)  # Adjust width based on number of tools

    # Plot the grouped bars
    for i, (tool_idx, row) in enumerate(df_metrics.iterrows()):
        offset = (i - len(df_metrics)/2 + 0.5) * width
        label = format_tool_name(tool_idx)
        label_noannot = format_tool_name(tool_idx)
        if tool_annotations[tool_idx]:
            label += f" ({tool_annotations[tool_idx]})"

        # Values for both groups
        values_first = [
            ((row[m] + 1) / 2) if m == "MCC" and not pd.isna(row[m])
            else (row[m] if not pd.isna(row[m]) else 0)
            for m in metrics_first
        ]
        values_second = [
            ((row[m] + 1) / 2) if m == "MCC" and not pd.isna(row[m])
            else (row[m] if not pd.isna(row[m]) else 0)
            for m in metrics_second
        ]

        # Bars for the first group (include labels to drive single legend)
        ax.bar(x_first + offset, values_first, width, label=label, alpha=0.8,
            color=tool_colors[label_noannot if tool_annotations[tool_idx] else label])

        # Bars for the second group (no labels to avoid duplicate legend entries)
        ax.bar(x_second + offset, values_second, width, alpha=0.8,
            color=tool_colors[label_noannot if tool_annotations[tool_idx] else label])

    # Axis labeling & ticks
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_title(f'Tool Performance Across Metrics (Analysis: {file_basename} [#{int(reference_count)}])')

    # Combine positions and labels
    x_all = np.concatenate([x_first, x_second])
    labels_all = [format_metric_name(m) for m in metrics_first] + [format_metric_name(m, mcc_scaled = True) for m in metrics_second]
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels_all, rotation=0, ha='center', fontsize=10)

    # Y range and grid
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Draw a vertical separator line in the gap between groups
    separator_x = len(metrics_first) + (gap - 0.5)  # with gap=1, this is n1 + 0.5 (center of gap)
    ax.axvline(separator_x, color='lightgray', linestyle='-', linewidth=1)

    # Draw dashed horizontal line ONLY under the second group
    if len(metrics_second) > 0:
        xmin = x_second.min() - 0.5  # pad to cover grouped bars including offsets
        xmax = x_second.max() + 0.5
        ax.hlines(y=0.5, xmin=xmin, xmax=xmax, color='gray', linestyle='--', linewidth=1)

    # Single legend (from labels added on first group)
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles, leg_labels, loc='upper center', bbox_to_anchor=(0.5, -0.08),
            ncol=min(3, len(df_metrics)//2), fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{file_basename}-metrics_all_tools_combined.png', dpi=300, bbox_inches='tight')
    plt.close()


    
    #Plot 2: Individual metric plots (each tool gets its own color)
    if not single_label_class:
        all_metrics = metrics_to_plot + ['Recall_Weighted']
        n_metrics = len(all_metrics)

        #Dynamically calculate grid size based on number of metrics to plot
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols  #Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
        axes = axes.flatten()

        #Create x-axis labels with annotations for individual plots
        xlabels_individual = []
        for idx, row in df_metrics.iterrows():
            label = format_tool_name(idx)
            annotations = []
            
            #Add NA annotation if any NAs are present for tool
            if row['percentage_NA_values'] > 0:
                annotations.append(f"NA={row['percentage_NA_values']:.1%}")
            
            #Add sample size difference annotation if sample size differs for tool
            if row['Count'] != reference_count:
                diff = int(row['Count'] - reference_count)
                sign = '+' if diff > 0 else ''
                annotations.append(f"*n{sign}={diff}")
            
            if annotations:
                label += '\n' + ', '.join(annotations)
            
            xlabels_individual.append(label)
        
        #Create color map for tools
        n_tools = len(df_metrics)
        if n_tools <= 10:
            colors = [plt.cm.tab10(i) for i in range(n_tools)]
        else:
            colors = [plt.cm.tab20(i) for i in range(n_tools)]

        for idx, metric in enumerate(all_metrics):
            ax = axes[idx]
            
            #Create bar plot for this metric with tool colors
            x_pos = np.arange(len(df_metrics))
            metric_values = [df_metrics.loc[tool_idx, metric] for tool_idx in df_metrics.index]
            
            bars = ax.bar(x_pos, metric_values, alpha=0.8, color=colors[:len(df_metrics)])
            
            ax.set_ylabel('Score')
            ax.set_title(format_metric_name(metric))
            ax.set_xticks(x_pos)
            ax.set_xticklabels(xlabels_individual, rotation=30, ha='right', fontsize=8)
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)

        #Remove extra empty subplots if we have any
        if n_metrics < len(axes):
            for idx in range(n_metrics, len(axes)):
                fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{file_basename}-metrics_individual_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        #Print summary
        print("\n=== Tool Comparability Summary ===")
        print(f"Reference sample size: {int(reference_count)}")
        
        different_count_tools = df_metrics[df_metrics['Count'] != reference_count]
        if len(different_count_tools) > 0:
            print("\nTools with different sample size:")
            for tool, row in different_count_tools.iterrows():
                diff = int(row['Count'] - reference_count)
                print(f"  - {format_tool_name(tool)}: n={int(row['Count'])} ({diff:+d} vs reference)")
        
        na_tools = df_metrics[df_metrics['percentage_NA_values'] > 0]
        if len(na_tools) > 0:
            print("\nTools with NAs:")
            for tool, row in na_tools.iterrows():
                print(f"  - {format_tool_name(tool)}: {row['percentage_NA_values']:.2%} NAs "
                    f"({row['percentage_samples_with_NAs']:.2%} of samples affected)")
        
        comparable_tools = df_metrics[(df_metrics['Count'] == reference_count) & 
                                    (df_metrics['percentage_NA_values'] == 0)]
        print(f"\nDirectly comparable tools (n={len(comparable_tools)}): "
            f"{', '.join([format_tool_name(t) for t in comparable_tools.index.tolist()])}")


def plot_heatmap_from_dict(metrics_dict, file_basename, metric_column='F1', figsize=(10, 8)):
    """Create heatmap with tools as columns and individual label classes as rows.

    Parameters:
    -----------
    metrics_dict : dict
        Dict of DataFrames (tool -> DataFrame with Label_Class as index)
    metric_column : str
        Column name to plot ('F1', 'Precision', 'Recall')
    figsize : tuple
        Figure size
    """
    
    #Get all unique label classes
    all_labels = set()
    for df in metrics_dict.values():
        all_labels.update(df.index)
    all_labels = sorted(all_labels)

    #Build data dict: tool -> {label_class -> score}
    data_dict = {}
    for tool, df in metrics_dict.items():
        formatted_tool = format_tool_name(tool)
        data_dict[formatted_tool] = {}
        for label in all_labels:
            if label in df.index:
                data_dict[formatted_tool][label] = df.loc[label, metric_column]
            else:
                data_dict[formatted_tool][label] = np.nan

    #Get true label distribution from first tool
    first_tool = list(metrics_dict.keys())[0]
    true_label_distributions = metrics_dict[first_tool]['Count']

    #Calculate macro and weighted averages
    macro_data = {}
    weighted_data = {}
    for tool, df in metrics_dict.items():
        formatted_tool = format_tool_name(tool)
        macro_data[formatted_tool] = df[metric_column].mean()
        total_count = df['Count'].sum()
        weighted_data[formatted_tool] = (df[metric_column] * df['Count']).sum() / total_count

    #Convert to DataFrame
    df = pd.DataFrame(data_dict).T
    df.columns = [label.capitalize() for label in all_labels]

    #Add macro and weighted rows
    macro_row = pd.DataFrame(macro_data, index=['Macro Average']).T
    weighted_row = pd.DataFrame(weighted_data, index=['Weighted Average']).T
    df = pd.concat([df.T, macro_row.T, weighted_row.T])
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': metric_column},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    #Add horizontal line to separate aggregate rows
    separator_pos = len(df) - 2
    ax.axhline(y=separator_pos, color='black', linewidth=2)

    #Update y-axis labels with sample counts
    y_labels = []
    for label in df.index:
        if label in ['Macro Average', 'Weighted Average']:
            y_labels.append(label)
        else:
            #Find original label (case-insensitive)
            original_label = None
            for orig in all_labels:
                if orig.upper() == label.upper():
                    original_label = orig
                    break
            if original_label:
                count = true_label_distributions.get(original_label.upper(), 0)
                y_labels.append(f"{label} (n={count})")
            else:
                y_labels.append(label)
    ax.set_yticklabels(y_labels, rotation=0)
    
    #Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_title(f'{metric_column} Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Tool', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Label classes for Analysis: {file_basename}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig
