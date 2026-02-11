import numpy as np
import pandas as pd
import os
import argparse
import subprocess
import pathlib
import json
from plotting_helpers import *

#Parse arguments from terminal
parser = argparse.ArgumentParser(description='Run method on files.')
parser.add_argument('--output_dir', type=str, help='Output directory')
parser.add_argument('--MetricsValidation.flag', type=str, nargs='+', help='Tools output collector')
args, _ = parser.parse_known_args()

#set variable names for arguments from terminal
output_dir = getattr(args, 'output_dir')
flag_filepaths = getattr(args, 'MetricsValidation.flag')

os.makedirs(f"./{output_dir}", exist_ok=True)

script_dir = pathlib.Path(__file__).resolve().parent

TOOL_COLOURS = {
    # fgsea
    "fgsea_DeltaCentroid": (0.1216, 0.4667, 0.7059, 1.0),   # tab10 blue
    "fgsea_RankExpr":  (0.6824, 0.7804, 0.9098, 1.0),   # light blue

    # gsva
    "gsva_RankReference":           (0.1725, 0.6275, 0.1725, 1.0),   # green
    "plage_RankReference":          (0.8902, 0.4667, 0.7608, 1.0),   # pink
    "zscore_RankReference":         (1.0000, 0.4980, 0.0549, 1.0),   # orange

    "gsva_RankExpr": (0.276, 0.722, 0.276, 1.0),
    "gsva_DeltaCentroid":  (0.138, 0.502, 0.138, 1.0),

    "plage_RankExpr": (0.912, 0.570, 0.816, 1.0),
    "plage_DeltaCentroid":  (0.712, 0.373, 0.608, 1.0),

    "zscore_RankExpr": (1.0, 0.598, 0.1549, 1.0),
    "zscore_DeltaCentroid":  (0.800, 0.398, 0.0439, 1.0),

    # singscore
    "singscore_DeltaCentroid": (0.8392, 0.1529, 0.1569, 1.0),  # red
    "singscore_RankExpr":   (1.0000, 0.5961, 0.5882, 1.0),  # light red

    # ssgsea
    "ssgsea_DeltaCentroid": (0.5804, 0.4039, 0.7412, 1.0),     # purple
    "ssgsea_RankExpr":   (0.7725, 0.6902, 0.8353, 1.0),     # light purple

    # ucell
    "ucell_DeltaCentroid":  (0.5490, 0.3373, 0.2941, 1.0),     # brown
    "ucell_RankExpr":    (0.7686, 0.6118, 0.5804, 1.0),     # light brown
}


# fetch dataset name
dfs = []
for fp in flag_filepaths:
    # read geneset from parameters
    with open ('/'.join(fp.split('/')[:4]) + '/parameters.json', 'r') as f:
        paramdict = json.load(f)
    df = fp.split('/')[2]
    params = '-'.join(paramdict.values())
    dataset_name = df + ':' + params
    df = pd.DataFrame({"flag_filepath": fp, "dataset": [dataset_name]})
    dfs.append(df)
# combine all into one DataFrame
df_all = pd.concat(dfs, ignore_index=True)

for input_data in df_all["dataset"].drop_duplicates():
    print(input_data)
    df_inputdata = df_all[df_all["dataset"] == input_data]
    subprocess.run(f"python {script_dir}/collate_metrics.py --output_dir {output_dir}\
    --name {input_data.split(":")[0]} --analysis {input_data} \
    --MetricsValidation.flag {" ".join(df_inputdata["flag_filepath"].astype(str))}",
    shell = True)


# addition for precision vs recall plotting together
all_metrics = []

# Iterate over each dataset folder
for dataset_folder in pathlib.Path(output_dir).iterdir():
    if dataset_folder.is_dir():
        dataset_name = dataset_folder.name
        metrics_file = pathlib.Path(f"{dataset_folder}/{dataset_name}-metrics_all_tools.tsv")
        if metrics_file.exists():
            df = pd.read_csv(metrics_file, sep='\t', index_col = 0)
            df.index.name = "Method"
            df = df.reset_index()
            df['Analysis'] = dataset_name
            all_metrics.append(df)

# Combine all dataframes
metrics_df = pd.concat(all_metrics, ignore_index=True)


# --- Define dataset groups here ---
group_A = ["CellCycleDB", 'O2DB', "GSE214654"]
group_B = ["GTEX", "TCGAnormal", "TCGAtumour", "TCGABRCA"]
group_C = ["CoPPO", "CPHBreast"]

groups = [
    ("In Vitro Data", group_A),
    ("GTEX/TCGA", group_B),
    ("In-House Data", group_C),
]

# --- Build filtered DataFrames for each group ---
group_dfs_full = []
group_dfs_universal = []
group_dfs_customised = []

for label, ds_list in groups:
    df_g = metrics_df[metrics_df["Analysis"].str.split(':').str[0].isin(ds_list)].copy()
    if df_g.empty: continue
    df_g["__GroupLabel__"] = label
    df_g['Input'] = df_g['Analysis'].str.split('-', expand = True)[0]

    conditions = [
        df_g['Analysis'].str.contains('VSdf', na=False),
        df_g['Analysis'].str.contains('WT', na=False)
    ]
    choices = ['VSdf', 'WT']
    df_g['Reference'] = np.select(
        conditions,
        choices,
        default='Universal'
    )

    group_dfs_full.append(df_g)
    group_dfs_universal.append(df_g[df_g['Reference'] == 'Universal'])
    group_dfs_customised.append(df_g[df_g['Reference'].isin(['VSdf', 'WT'])])


# --- Consistent Method colors across all subplots ---
all_tools = sorted(metrics_df["Method"].dropna().unique().tolist())

precision_vs_recall_group_plotting(group_dfs_full, groups, all_tools, 'Full', args.output_dir, TOOL_COLOURS)
precision_vs_recall_group_plotting(group_dfs_universal, groups, all_tools, 'Universal', args.output_dir, TOOL_COLOURS)
precision_vs_recall_group_plotting(group_dfs_customised, groups, all_tools, 'Same-Cohort', args.output_dir, TOOL_COLOURS)

# group all dfs together for sina plotting of MCC distribution
plot_sina(args.output_dir, TOOL_COLOURS)

pathlib.Path(f"{output_dir}/plotting_wrapper_complete.flag").touch()
