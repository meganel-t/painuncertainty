# Import pandas to load excel table
from scipy.io import loadmat
import pandas as pd

# Import matplotlip to plot
import matplotlib.pyplot as plt

# Import numpy for vector/matrices
import numpy as np

# Import seabonr for nicer plots
import seaborn as sns

# Import os to manipulate files and folders
import os
from os.path import join as opj

# Stats package
from scipy import stats

import urllib.request
import matplotlib.font_manager as fm
import matplotlib


# Get nice font
urllib.request.urlretrieve(
    "https://github.com/gbif/analytics/raw/master/fonts/Arial%20Narrow.ttf",
    "arialnarrow.ttf",
)
fm.fontManager.addfont("arialnarrow.ttf")
matplotlib.rc("font", family="Arial Narrow")

# Path for the bids root
bids_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
)
out_fig = opj(bids_root, "derivatives", "figures")
out_stats = opj(bids_root, "derivatives", "stats")
if not os.path.exists(out_fig):
    os.mkdir(out_fig)
if not os.path.exists(out_stats):
    os.mkdir(out_stats)

data = pd.read_csv(opj(bids_root, "derivatives", "all_part_data.csv"))

# Exclude participants using participants.tsv
participants = pd.read_csv(opj(bids_root, "participants.tsv"), sep="\t")
participants = participants[participants.excluded == 0]["participant_id"].values

# Parameters recovery
win_model = "HGF2_uncertainty"
all_frames = []
for p in participants:

    # Load actual parameters of winning model
    obs_params = pd.read_csv(
        opj(
            bids_root,
            "derivatives",
            p,
            p + "_" + "HGF2_uncertainty" + "_obsparams.csv",
        )
    )
    perc_params = pd.read_csv(
        opj(
            bids_root,
            "derivatives",
            p,
            p + "_" + "HGF2_uncertainty" + "_prcparams.csv",
        )
    )

    # Load parameters recovered from simuliations and average
    recovered = pd.DataFrame(
        pd.read_csv(
            opj(
                bids_root,
                "derivatives",
                p,
                p + "_" + "HGF2" + "_recovered_params.csv",
            )
        ).mean()
    ).T

    # Rename columns
    recovered.columns = ["recovered_" + c.replace("b", "be") for c in recovered.columns]
    # Concatenate all in one df
    all_frame = pd.concat([obs_params, perc_params, recovered], axis=1)
    all_frame.index = [p]
    # Collect all frames from all participants
    all_frames.append(all_frame)


# Concatenate all frames
all_frames = pd.concat(all_frames)
# Compute correlation matrix
corr_matrix = all_frames.corr(method="spearman")
plot_matrix = corr_matrix.loc[
    ["om2", "be0", "be1", "be2", "ze"],
    [
        "recovered_om2",
        "recovered_be0",
        "recovered_be1",
        "recovered_be2",
        "recovered_ze",
    ],
]

plot_matrix.to_csv(opj(out_fig, "parameters_recovery_plotdata.csv"))
# Heatmap plot
fig, ax = plt.subplots(figsize=(2.5, 2))
sns.heatmap(
    plot_matrix,
    cmap="RdBu_r",
    center=0,
    annot=True,
    ax=ax,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="black",
    cbar_kws={"label": r"Spearman $\rho$"},
    annot_kws={"fontsize": 5},
    fmt=".2f",
)
ax.set_xticklabels(
    [
        r"$\omega_2$",
        r"$\beta_0$",
        r"$\beta_1$",
        r"$\beta_2$",
        r"$\zeta$",
    ],
    rotation=0,
    fontsize=10,
)
ax.set_yticklabels(
    [
        r"$\omega_2$",
        r"$\beta_0$",
        r"$\beta_1$",
        r"$\beta_2$",
        r"$\zeta$",
    ],
    rotation=0,
    fontsize=10,
)

plt.xlabel("Mean recovered parameters", fontsize=12)
plt.ylabel("Simulated parameter", fontsize=12)
for _, spine in ax.spines.items():
    spine.set_visible(True)
plt.savefig(opj(out_fig, "parameters_recovery.png"), dpi=800, bbox_inches="tight")


# Make regplot for each parameter and add correlation coefficient and p value
for param, label in zip(
    ["om2", "be0", "be1", "be2", "ze"],
    [
        r"$\omega_2$",
        r"$\beta_0$",
        r"$\beta_1$",
        r"$\beta_2$",
        r"$\zeta$",
    ],
):
    fig, ax = plt.subplots(figsize=(2.5, 2))
    sns.regplot(x=param, y="recovered_" + param, data=all_frames, ax=ax, ci=95)
    ax.set_xlabel("Simulated " + label, fontsize=12)
    ax.set_ylabel("Mean recovered " + label, fontsize=12)
    ax.set_title(
        "R = %.2f"
        % (stats.spearmanr(all_frames[param], all_frames["recovered_" + param])[0]),
        fontsize=10,
    )
    plt.savefig(
        opj(out_fig, "parameters_recovery_" + param + ".png"),
        dpi=800,
        bbox_inches="tight",
    )


# Model recovery
labels = ["HGF2", "HGF2pu", "HGF3", "HGF3pu", "rw", "ph"]
frame_ef = pd.DataFrame(
    index=labels, columns=["HGF2", "HGF2pu", "HGF3", "HGF3pu", "rw", "ph"]
)
frame_ep = pd.DataFrame(
    index=labels, columns=["HGF2", "HGF2pu", "HGF3", "HGF3pu", "rw", "ph"]
)

for sim_mod in ["HGF2", "HGF2pu", "HGF3", "HGF3pu", "rw", "ph"]:

    famcomp = loadmat(
        opj(bids_root, "derivatives/VBA_BMC_" + sim_mod + "_sim_recovery.mat")
    )

    ep = famcomp["out"]["pep"][0][0][0]
    ef = [float(ef) * 100 for ef in famcomp["out"]["Ef"][0][0]]

    frame_ef[sim_mod] = ef
    frame_ep[sim_mod] = ep

frame_ef = frame_ef.astype(float) / 100
fig, ax = plt.subplots(figsize=(2.5, 2))
sns.heatmap(
    frame_ef,
    cmap="viridis",
    center=0.5,
    annot=True,
    ax=ax,
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="black",
    cbar_kws={"label": "Proportion"},
    annot_kws={"fontsize": 5},
    fmt=".2f",
)
ax.set_xticklabels(labels, rotation=90, fontsize=10)
ax.set_yticklabels(labels, rotation=0, fontsize=10)

plt.xlabel("Simulated model", fontsize=12)
plt.ylabel("Recovered model", fontsize=12)
for _, spine in ax.spines.items():
    spine.set_visible(True)
plt.savefig(opj(out_fig, "model_recovery_ef.png"), dpi=800, bbox_inches="tight")


labels = ["HGF2", "HGF2pu", "HGF3", "HGF3pu", "RW", "PH"]
