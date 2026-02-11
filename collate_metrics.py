import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import seaborn as sns
from collate_metrics_helper import *

###Join together metrics calculated with different tools for a given dataset

#Parse arguments from terminal
parser = argparse.ArgumentParser(description='Run method on files.')
parser.add_argument('--output_dir', type=str, help='Output directory')
parser.add_argument('--name',type=str, help='Dataset name')
parser.add_argument('--MetricsValidation.flag', type=str, nargs='+', help='Tools output collector')
parser.add_argument('--analysis', type=str, help='Dataset:Geneset collection combination of input')
args, _ = parser.parse_known_args()

#set variable names for arguments from terminal
output_dir = getattr(args, 'output_dir')
dataset_name = getattr(args, 'name')
analysis_name = getattr(args, 'analysis')
flag_filepaths = getattr(args, 'MetricsValidation.flag')


#########################################################
#################### Main ###############################
#########################################################

output_dir = f"{output_dir}/{analysis_name}"
os.makedirs(f"./{output_dir}", exist_ok=True)

dir_paths_auc = []
dir_paths_top1 = []

TOOL_COLOURS_EXTRA = {
    # fgsea
    "fgsea (DeltaCentroid)": (0.1216, 0.4667, 0.7059, 1.0),   # tab10 blue
    "fgsea (RankExpr)":  (0.6824, 0.7804, 0.9098, 1.0),   # light blue

    # gsva
    "gsva (RankReference)":           (0.1725, 0.6275, 0.1725, 1.0),   # green
    "plage (RankReference)":          (0.8902, 0.4667, 0.7608, 1.0),   # pink
    "zscore (RankReference)":         (1.0000, 0.4980, 0.0549, 1.0),   # orange

    "gsva (RankExpr)": (0.276, 0.722, 0.276, 1.0),
    "gsva (DeltaCentroid)":  (0.138, 0.502, 0.138, 1.0),

    "plage (RankExpr)": (0.912, 0.570, 0.816, 1.0),
    "plage (DeltaCentroid)":  (0.712, 0.373, 0.608, 1.0),

    "zscore (RankExpr)": (1.0, 0.598, 0.1549, 1.0),
    "zscore (DeltaCentroid)":  (0.800, 0.398, 0.0439, 1.0),

    # singscore
    "singscore (DeltaCentroid)": (0.8392, 0.1529, 0.1569, 1.0),  # red
    "singscore (RankExpr)":   (1.0000, 0.5961, 0.5882, 1.0),  # light red

    # ssgsea
    "ssgsea (DeltaCentroid)": (0.5804, 0.4039, 0.7412, 1.0),     # purple
    "ssgsea (RankExpr)":   (0.7725, 0.6902, 0.8353, 1.0),     # light purple

    # ucell
    "ucell (DeltaCentroid)":  (0.5490, 0.3373, 0.2941, 1.0),     # brown
    "ucell (RankExpr)":    (0.7686, 0.6118, 0.5804, 1.0),     # light brown
}


#Extract the directory paths to the top1 metrics files and auc metrics files
for flag_filepaths in flag_filepaths: 
    metrics_type = flag_filepaths.split("/")[-3]
    dir_path = os.path.dirname(flag_filepaths)
    
    assert metrics_type in ["top1validation", "AUCvalidation"], f"Did not find metrics directory properly! Metrics type extracted: {metrics_type}"

    if metrics_type == "top1validation": 
        dir_paths_top1.append(dir_path)
    elif metrics_type == "AUCvalidation":
        dir_paths_auc.append(dir_path)

#Get all metrics and save overall metrics for each tool to tsv file
df_metrics, top1_metrics_tools_dict, auc_metrics_tools_dict = get_all_metrics(dir_paths_top1, dir_paths_auc, dataset_name)
df_metrics = df_metrics[~df_metrics.index.duplicated(keep='first')]
df_metrics = df_metrics.sort_index()
df_metrics.reset_index().rename(columns={'index': 'Algorithm'}).to_csv(f"{output_dir}/{analysis_name}-metrics_all_tools.tsv", sep='\t', index=False)

# filtering step to remove tools with all NAs
tools_to_remove = df_metrics.index[(np.isnan(df_metrics["percentage_NA_values"])) | (df_metrics["percentage_NA_values"] == 1)]
df_metrics = df_metrics.drop(tools_to_remove)

#Marker for only having one true label class; MCC (and all other metrics except Recall_Weighted) will be returned as nan
if not isinstance(auc_metrics_tools_dict, dict):# np.isnan(auc_metrics_tools_dict):
    plot_metrics_with_annotations(df_metrics, output_dir, dataset_name,
                                  analysis_name, metrics_to_plot=["Recall_Weighted"],
                                  single_label_class = True, tool_colors = TOOL_COLOURS_EXTRA)

else:
    #Plot overall metrics
    plot_metrics_with_annotations(df_metrics, output_dir, dataset_name, analysis_name, tool_colors = TOOL_COLOURS_EXTRA)

    #Plot specific metrics individually for each label class
    for metric in ['F1', 'Precision', 'Recall']:
        # Heatmap
        fig = plot_heatmap_from_dict(top1_metrics_tools_dict, dataset_name, metric_column=metric)
        plt.savefig(f'{output_dir}/{analysis_name}-label_class_{metric}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Similarly for AUC metrics
    for metric in ['AUC']:
        fig = plot_heatmap_from_dict(auc_metrics_tools_dict, dataset_name, metric_column=metric)
        plt.savefig(f'{output_dir}/{analysis_name}-label_class_{metric}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
