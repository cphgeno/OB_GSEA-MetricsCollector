import pandas as pd
from pathlib import Path
import matplotlib.colors as mcolors
from plotnine import (
    ggplot, aes, geom_violin, geom_sina, theme_bw,
    labs, theme, element_text, scale_fill_manual, 
    scale_color_manual
)
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.pyplot as plt



def precision_vs_recall_group_plotting(df_list, groups, all_tools, list_type, output_dir, TOOL_COLOURS):
    # --- Create the subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)

    # Determine global axis limits for consistent scales
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1

    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.05
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.05

    cols_for_plot = ['Precision_Weighted', 'Recall_Weighted']

    # --- Plot each group ---
    for ax, (label, _), df_g in zip(axes, groups, df_list):
        # Sanity check to remove analyses which are missing either of metric value from plotting
        for analysis in df_g['Analysis'].drop_duplicates():
            df_g_analysis = df_g[df_g['Analysis'] == analysis]
            if any(df_g_analysis[col].isna().all() for col in cols_for_plot):
                df_g = df_g[df_g['Analysis'] != analysis]
        
        sns.scatterplot(
            data=df_g,
            x="Recall_Weighted",
            y="Precision_Weighted",
            hue="Method",
            style="Analysis",
            palette=TOOL_COLOURS,  # identical colors for Tools across panels
            s=200,
            ax=ax,
            legend="full"
        )

        # Axis cosmetics
        ax.set_title(label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # --- Gray dashed diagonal line: y = x ---
        # Use the overlapping range to avoid tails outside visible area.
        diag_min = max(x_min - x_pad, y_min - y_pad)
        diag_max = min(x_max + x_pad, y_max + y_pad)
        ax.plot([diag_min, diag_max], [diag_min, diag_max],
                color="gray", linestyle="--", linewidth=1, zorder=0)

        # --- Build per-subplot legend UNDER the axis for DATASET shapes only ---
        leg = ax.get_legend()
        if leg is not None:
            handles, labels = ax.get_legend_handles_labels()

            present_datasets = set(df_g["Analysis"].dropna().unique().tolist())
            tools_set = set(all_tools)

            dataset_handles, dataset_labels = [], []
            for h, l in zip(handles, labels):
                # keep only dataset entries (exclude tool labels)
                if l in present_datasets and l not in tools_set:
                    dataset_handles.append(h)
                    dataset_labels.append(l)

            # Replace with dataset-only legend and position it underneath the subplot
            leg.remove()
            if dataset_handles:
                sorted_pairs = sorted(zip(dataset_labels, dataset_handles), key=lambda x: x[0])
                dataset_labels, dataset_handles = zip(*sorted_pairs)

                ax.legend(
                    handles=dataset_handles,
                    labels=dataset_labels,
                    title="Analysis",
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.22),  # centered below the axes
                    frameon=True,
                    fontsize=9,
                    ncol=1,
                    borderaxespad=0.0
                )

    # --- Single global legend for TOOL colors on the right ---
    tool_handles = [
        Line2D([0], [0],
            marker="o",
            linestyle="",
            color=color,
            markerfacecolor=color,
            markersize=9,
            label=tool)
        for tool, color in TOOL_COLOURS.items()
    ]

    if tool_handles:
        fig.legend(
            handles=tool_handles,
            title="Method",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.,
            frameon=True
        )

    # Add a bit more bottom margin so the under-subplot legends are visible
    plt.subplots_adjust(bottom=0.22, wspace=0.25)

    fig.suptitle(f"Precision vs Recall Across Analyses (grouped) - {f'{list_type} Reference' if list_type != 'Full' else 'All References'}", fontsize=14, y=0.98)
    plt.savefig(f"{output_dir}/precision_vs_recall_by_group-{list_type}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return


def plot_sina(output_dir, TOOL_COLOURS):
    TOOL_COLOURS_HEX = {
        k: mcolors.to_hex(v) for k, v in TOOL_COLOURS.items()
    }

    # Collect MCC columns
    mcc_tables = []
    for d in sorted(Path(output_dir).iterdir()):
        if not d.is_dir():
            continue
        dataset_geneset = d.name
        tsv_file = d / f"{dataset_geneset}-metrics_all_tools.tsv"
        if not tsv_file.exists():
            continue
        df = pd.read_csv(tsv_file, sep="\t")
        # First column = tools, second column = MCC
        mcc = df.iloc[:, [0, 1]].copy()
        mcc.columns = ["Method", dataset_geneset]
        mcc_tables.append(mcc.set_index("Method"))

    # Combine all MCC columns
    mcc_wide = pd.concat(mcc_tables, axis=1)

    # Save compiled table
    mcc_wide.to_csv(Path(output_dir, 'compiled_mcc_all_datasets.tsv'), sep="\t")

    # Prepare data for plotnine 
    mcc_long = (
        mcc_wide
        .reset_index()
        .melt(id_vars="Method", var_name="dataset_geneset", value_name="MCC")
        .dropna()
    )

    # Sina plot
    p = (
        ggplot(mcc_long, aes(x="Method", y="MCC", fill="Method", color="Method"))
        + geom_violin(width=0.9, alpha=0.25)
        + geom_sina(size=3, alpha=0.8)
        + scale_fill_manual(values=TOOL_COLOURS_HEX)
        + scale_color_manual(values=TOOL_COLOURS_HEX)
        + theme_bw()
        + labs(
            x="Method",
            y="MCC",
            title="MCC distribution across analyses - Universal reference",
            fill ="Method",
            color = "Method"
        )
        + theme(
            axis_text_x=element_text(rotation=45, ha="right")
        )
    )
    p.save(Path(output_dir, 'mcc_sina_plot.png'), width=12, height=6, dpi=300)
    return

