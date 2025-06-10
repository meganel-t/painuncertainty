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
import urllib.request
import matplotlib.font_manager as fm
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import ptitprince as pt

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

# Parameters
labelsize = 8
ticksize = 7
figurelettersize = 10
pointsize = 2
scale = 0.5
#############################################################
# Figure 1 task and model free results
#############################################################
# Load model free data averaged from the c04 script
fig_dat = pd.read_csv(opj(bids_root, "derivatives", "figures", "model_free_data.csv"))

fig = plt.figure(figsize=(7.5, 3.5), dpi=600)
gs = fig.add_gridspec(nrows=30, ncols=70)
gs[0, 0]
ax_error = fig.add_subplot(gs[0:13, 40:52])
ax_rt1 = fig.add_subplot(gs[0:12, 58:70])
ax_rt2 = fig.add_subplot(gs[12:13, 58:70])
ax_pain1 = fig.add_subplot(gs[17:30, 40:52])
ax_pain2 = fig.add_subplot(gs[17:30, 58:70])
ax_cont = fig.add_subplot(gs[17:, 0:34])
# Proprortion of errors

cont_data = pd.read_csv(
    opj(bids_root, "derivatives", "figures", "actual_contingencies.csv")
)
# Add contingency plot
ax_cont.set_xlim(1, 192)
ax_cont.set_xlabel("Trial", fontsize=labelsize)
ax_cont.set_ylabel("P(High Pain | Tone)", fontsize=labelsize)
ax_cont.set_ylim([0, 1])
ax_cont.plot(
    cont_data["actual_contingencies_hightT"],
    color="k",
    linestyle="--",
    alpha=1,
    label="High tone",
)
ax_cont.plot(
    cont_data["actual_contingencies_lowT"],
    label="Low tone",
    color="gray",
    linestyle="--",
    alpha=1,
)
ax_cont.legend(
    fontsize=ticksize, frameon=True, ncol=1, columnspacing=0.8, handletextpad=0.1
)
ax_cont.tick_params(axis="both", labelsize=ticksize)

fig_dat["type_offset"] = (
    fig_dat["type"].map({"E": -0.5, "N": 0.9, "UE": 1.9}).astype(float)
)

# Get first color from set1
colors = sns.color_palette("Set1")[0]
sns.pointplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="prop_errors",
    hue="stimulus_",
    ci=95,
    edgecolor="black",
    dodge=0.25,
    markers=["s", "s"],
    zorder=5,
    scale=scale,
    errwidth=scale,
    ax=ax_error,
)

# Offset the blue point line a bit
offset_blue = 0.20
offset_red = 0.05

for idx, c in enumerate(ax_error.get_children()):
    if isinstance(c, matplotlib.lines.Line2D) and idx < 4:
        c.set_xdata(c.get_xdata() + offset_red)
        c.set_linewidth(1)
        # if idx == 3:
        #     c.set_color("k")
        #     c.set_linewidth(1)
        #     c.set_zorder(10)
    if isinstance(c, matplotlib.lines.Line2D) and idx > 4:
        c.set_xdata(c.get_xdata() + offset_blue)
        c.set_linewidth(1)
    if isinstance(c, matplotlib.collections.PathCollection) and idx < 6:
        a = c.get_offsets().copy()
        for i in range(a.shape[0]):
            a[i][0] = a[i][0] + offset_red
        c.set_offsets(a)
        c.set_edgecolor("black")
        c.set_zorder(10)
        c.set_linewidth(0.5)

    if isinstance(c, matplotlib.collections.PathCollection) and idx > 4:
        a = c.get_offsets().copy()
        for i in range(a.shape[0]):
            a[i][0] = a[i][0] + offset_blue
        c.set_offsets(a)
        c.set_zorder(10)
        c.set_edgecolor("black")
        c.set_linewidth(0.5)

pt.half_violinplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="prop_errors",
    hue="stimulus_",
    inner=None,
    width=0.7,
    offset=0,
    cut=0,
    ax=ax_error,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.5,
    dodge=1,
    zorder=1,
)
pt.stripplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="prop_errors",
    hue="stimulus_",
    edgecolor="black",
    s=pointsize,
    linewidth=0.5,
    alpha=0.5,
    move=0.1,
    dodge=0.2,
    ax=ax_error,
    zorder=2,
)

ax_error.set_xlabel("", fontsize=labelsize)
ax_error.set_xticks([0, 1, 2], ["Exp.", "Neut.", "Unexp."], fontsize=ticksize)

ax_error.set_ylabel("Proportion of errors", fontsize=labelsize)
ax_error.tick_params(labelsize=ticksize)
ax_error.tick_params(axis="x", labelsize=labelsize, rotation=0)
ax_error.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))

h, l = ax_error.get_legend_handles_labels()
ax_error.legend(
    h[2:4],
    ["High pain", "Low pain"],
    title="",
    fontsize=ticksize,
    frameon=False,
    loc="upper left",
)

ax_rt2.set_ylim(0, 300)
ax_rt2.set_yticks([0])
ax_rt1.set_ylim(300, 1600)
ax_rt2.spines["top"].set_visible(False)
ax_rt1.spines["bottom"].set_visible(False)

plt.sca(ax_rt1)

pt.half_violinplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="rt_ms",
    hue="stimulus_",
    inner=None,
    width=0.7,
    offset=0,
    cut=0,
    ax=ax_rt1,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.5,
    dodge=True,
    zorder=1,
)
pt.stripplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="rt_ms",
    hue="stimulus_",
    edgecolor="black",
    s=pointsize,
    linewidth=0.5,
    alpha=0.5,
    move=0.1,
    dodge=0.2,
    ax=ax_rt1,
    zorder=2,
)

sns.pointplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="rt_ms",
    hue="stimulus_",
    ci=95,
    edgecolor="black",
    dodge=0.25,
    zorder=5,
    scale=scale,
    markers=["s", "s"],
    errwidth=scale,
    ax=ax_rt1,
)

# Offset the blue point line a bit
for idx, c in enumerate(ax_rt1.get_children()):
    if idx > 15 and idx < 26:
        if isinstance(c, matplotlib.lines.Line2D) and idx < 21 and idx > 15:
            c.set_xdata(c.get_xdata() + offset_red)
            c.set_zorder(10)
            c.set_linewidth(1)
        if isinstance(c, matplotlib.lines.Line2D) and idx > 20 and idx > 15:
            c.set_xdata(c.get_xdata() + offset_blue)
            c.set_zorder(10)
            c.set_linewidth(1)
        if (
            isinstance(c, matplotlib.collections.PathCollection)
            and idx < 21
            and idx > 15
        ):
            a = c.get_offsets().copy()
            for i in range(a.shape[0]):
                a[i][0] = a[i][0] + offset_red
            c.set_offsets(a)
            c.set_zorder(10)
            c.set_edgecolor("black")
            c.set_linewidth(0.5)
        if isinstance(c, matplotlib.collections.PathCollection) and idx > 20:
            a = c.get_offsets().copy()
            for i in range(a.shape[0]):
                a[i][0] = a[i][0] + offset_blue
            c.set_offsets(a)
            c.set_zorder(10)
            c.set_edgecolor("black")
            c.set_linewidth(0.5)

ax_rt1.legend().remove()

# Axis breaks
d = 0.02  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_rt1.transAxes, color="k", clip_on=False)
ax_rt1.plot((-d, +d), (0, 0), **kwargs, linewidth=0.5)  # top-left diagonal
kwargs.update(transform=ax_rt2.transAxes)  # switch to the bottom axes
ax_rt2.plot((-d, +d), (1, 1), **kwargs, linewidth=0.5)  # bottom-left diagonal

plt.sca(ax_rt2)
sns.pointplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="rt_ms",
    hue="stimulus_",
    zorder=10,
    dodge=True,
    scale=scale,
    errwidth=scale,
    ci=95,
    markers=["s", "s"],
    legend=False,
)
ax_rt2.set_ylim(0, 300)

ax_rt1.set_xlabel("", fontsize=labelsize)
ax_rt1.set_xticks([0, 1, 2], ["Exp.", "Neut.", "Unexp."])

ax_rt1.set_ylabel("Response time (ms)", fontsize=labelsize)
ax_rt1.tick_params(labelsize=ticksize)
ax_rt2.tick_params(axis="x", labelsize=labelsize, rotation=0)
# ax_rt1.tick_params(axis="x", labelsize=10, rotation=30)
ax_rt1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

ax_rt1.tick_params(
    left=True, right=False, labelleft=True, labelbottom=False, bottom=False
)
ax_rt2.set_ylabel("", fontsize=12)
ax_rt2.set_xlabel("", fontsize=12)

ax_rt2.tick_params(
    left=False,
    right=False,
    labelleft=True,
    labelbottom=True,
    bottom=True,
    labelsize=ticksize,
)

ax_rt2.set_xticks([0, 1, 2], ["Exp.", "Neut.", "Unexp."], fontsize=labelsize)

ax_rt2.set_yticks([0])
# Set the axis color
ax_rt2.spines["bottom"].set_color("black")

ax_rt1.legend().remove()
ax_rt2.legend().remove()

plt.sca(ax_pain1)

pt.half_violinplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="pain_rating",
    hue="stimulus_",
    inner=None,
    width=0.7,
    offset=0,
    cut=0,
    ax=ax_pain1,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.5,
    dodge=True,
    zorder=1,
)
pt.stripplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="pain_rating",
    hue="stimulus_",
    edgecolor="black",
    s=pointsize,
    linewidth=0.5,
    alpha=0.5,
    move=0.1,
    dodge=0.2,
    ax=ax_pain1,
    zorder=2,
)
sns.pointplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="pain_rating",
    hue="stimulus_",
    ci=95,
    edgecolor="black",
    dodge=0.25,
    zorder=5,
    scale=scale,
    errwidth=scale,
    markers=["s", "s"],
    ax=ax_pain1,
)

# Offset the blue point line a bit
for idx, c in enumerate(ax_pain1.get_children()):
    if idx > 15 and idx < 26:
        if isinstance(c, matplotlib.lines.Line2D) and idx < 22:
            c.set_xdata(c.get_xdata() + offset_red)
            c.set_zorder(10)
            c.set_linewidth(1)
        if isinstance(c, matplotlib.lines.Line2D) and idx > 19:
            c.set_xdata(c.get_xdata() + offset_blue)
            c.set_zorder(10)
            c.set_linewidth(1)
        if isinstance(c, matplotlib.collections.PathCollection) and idx < 22:
            a = c.get_offsets().copy()
            for i in range(a.shape[0]):
                a[i][0] = a[i][0] + offset_red
            c.set_offsets(a)
            c.set_zorder(10)
            c.set_edgecolor("black")
            c.set_linewidth(0.5)
        if isinstance(c, matplotlib.collections.PathCollection) and idx > 22:
            a = c.get_offsets().copy()
            for i in range(a.shape[0]):
                a[i][0] = a[i][0] + offset_blue
            c.set_offsets(a)
            c.set_zorder(10)
            c.set_edgecolor("black")
            c.set_linewidth(0.5)

ax_rt1.legend().remove()

ax_pain1.set_xlabel("", fontsize=14)
ax_pain1.set_xticks([0, 1, 2], ["Exp.", "Neut.", "Unexp."])

ax_pain1.set_ylabel("Pain rating", fontsize=labelsize)
ax_pain1.tick_params(labelsize=ticksize)
ax_pain1.tick_params(axis="x", labelsize=labelsize, rotation=0)
ax_pain1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

ax_rt1.set_title("d", fontsize=figurelettersize, fontweight="bold", x=-0.3, y=1.01)
ax_error.set_title("c", fontsize=figurelettersize, fontweight="bold", x=-0.3, y=1.01)
ax_pain1.set_title("e", fontsize=figurelettersize, fontweight="bold", x=-0.3, y=1.01)
ax_pain2.set_title("f", fontsize=figurelettersize, fontweight="bold", x=-0.3, y=1.01)
ax_cont.set_title("b", fontsize=figurelettersize, fontweight="bold", x=-0.08, y=1.0)

plt.sca(ax_pain2)
pt.half_violinplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="ratings_mean_centered",
    hue="stimulus_",
    inner=None,
    width=0.7,
    offset=0,
    cut=0,
    ax=ax_pain2,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.5,
    dodge=True,
    zorder=1,
)
pt.stripplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="ratings_mean_centered",
    hue="stimulus_",
    edgecolor="black",
    s=pointsize,
    linewidth=0.5,
    alpha=0.5,
    move=0.1,
    dodge=0.2,
    ax=ax_pain2,
    zorder=2,
)
sns.pointplot(
    data=fig_dat,
    palette="Set1",
    x="type",
    y="ratings_mean_centered",
    hue="stimulus_",
    ci=95,
    edgecolor="black",
    dodge=0.25,
    zorder=5,
    scale=scale,
    markers=["s", "s"],
    errwidth=scale,
    ax=ax_pain2,
)

for idx, c in enumerate(ax_pain2.get_children()):
    if idx > 15 and idx < 26:
        if isinstance(c, matplotlib.lines.Line2D) and idx < 22:
            c.set_xdata(c.get_xdata() + offset_red)
            c.set_zorder(10)
            c.set_linewidth(1)
        if isinstance(c, matplotlib.lines.Line2D) and idx > 19:
            c.set_xdata(c.get_xdata() + offset_blue)
            c.set_zorder(10)
            c.set_linewidth(1)
        if isinstance(c, matplotlib.collections.PathCollection) and idx < 22:
            a = c.get_offsets().copy()
            for i in range(a.shape[0]):
                a[i][0] = a[i][0] + offset_red
            c.set_offsets(a)
            c.set_zorder(10)
            c.set_edgecolor("black")
            c.set_linewidth(0.5)
        if isinstance(c, matplotlib.collections.PathCollection) and idx > 22:
            a = c.get_offsets().copy()
            for i in range(a.shape[0]):
                a[i][0] = a[i][0] + offset_blue
            c.set_offsets(a)
            c.set_zorder(10)
            c.set_edgecolor("black")
            c.set_linewidth(0.5)

ax_pain2.set_xlabel("", fontsize=labelsize)
ax_pain2.set_xticks([0, 1, 2], ["Exp.", "Neut.", "Unexp."], fontsize=labelsize)
ax_pain2.set_ylabel("Pain rating (mean centered)", fontsize=labelsize)
ax_pain2.tick_params(labelsize=ticksize)
ax_pain2.tick_params(axis="x", labelsize=labelsize, rotation=0)
ax_pain2.yaxis.set_major_formatter(FormatStrFormatter("%.01f"))
ax_pain2.legend().remove()
ax_pain1.legend().remove()

plt.tight_layout()
plt.savefig(opj(out_fig, "model_free_results.svg"), dpi=1200, bbox_inches="tight", transparent=True)

############################################################################
# Figure 2 - Model fitting
############################################################################
plot_matrix = pd.read_csv(
    opj(bids_root, "derivatives", "figures", "parameters_recovery_plotdata.csv"),
    index_col=0,
)

axd = plt.figure(layout="constrained", figsize=(7, 6), dpi=600).subplot_mosaic(
    """
    AAABBB
    AAACCC
    AAADDD
    EEEFFF
    EEEFFF
    """
)
# Trajectory plots
labelsize = labelsize + 2
ticksize = ticksize + 2
figurelettersize += 2
data = pd.read_csv(opj(bids_root, "derivatives", "figures", "model_based_data.csv"))
data.index = data["participant_id"]
# Pick representative participant
data_plot = data[data.index == "sub-048"]
axd["D"].set_xlim(1, 192)
axd["D"].set_xlabel("Trial", fontsize=ticksize)
par1 = axd["D"].twinx()
par1.set_ylim([0, 1])
axd["D"].set_ylim([0, 1])
axd["D"].plot(
    cont_data["actual_contingencies_hightT"],
    label="Actual",
    color="k",
    linestyle="--",
    alpha=0.5,
)
axd["D"].plot(
    cont_data["actual_contingencies_lowT"],
    label="Low tone",
    color="gray",
    linestyle="--",
    alpha=0.5,
)

sns.lineplot(
    data=data_plot, x="trial", y="abs_da1", ax=par1, color="#75A9CF", linewidth=2
)
par1.tick_params(axis="y", colors="#75A9CF", labelsize=ticksize)
par1.set_yticks([])
axd["D"].tick_params(axis="both", labelsize=ticksize)
axd["D"].set_ylabel("", fontsize=labelsize)
par1.set_ylabel("Absolute\nprediction error", fontsize=labelsize)
axd["D"].set_xlabel("Trial", fontsize=labelsize)
axd["D"].set_xticklabels(labels=axd["D"].get_xticklabels(), fontsize=ticksize)
par1.yaxis.label.set_color("#75A9CF")
par1.spines["right"].set_edgecolor("#75A9CF")

axd["C"].set_xlim(1, 192)
axd["C"].set_xlabel("Trial", fontsize=10)
par1 = axd["C"].twinx()
par1.set_ylim([0, 0.4])
axd["C"].set_ylim([0, 1])
axd["C"].plot(
    cont_data["actual_contingencies_hightT"],
    label="Actual",
    color="k",
    linestyle="--",
    alpha=0.5,
)
axd["C"].plot(
    cont_data["actual_contingencies_lowT"],
    label="Low tone",
    color="gray",
    linestyle="--",
    alpha=0.5,
)
sns.lineplot(
    data=data_plot, x="trial", y="sahat_1", color="#0370B0", ax=par1, linewidth=2
)
par1.tick_params(axis="y", colors="#0370B0", labelsize=ticksize)
par1.set_yticks([])

axd["C"].tick_params(axis="both", labelsize=ticksize)
axd["C"].set_ylabel("", fontsize=labelsize)
par1.set_ylabel("Irreducible\nuncertainty", fontsize=labelsize)
axd["C"].set_xlabel("", fontsize=labelsize)
par1.yaxis.label.set_color("#0370B0")
par1.spines["right"].set_edgecolor("#0370B0")
axd["C"].set_xticks([])

axd["B"].set_xlim(1, 192)
axd["B"].set_xlabel("Trial", fontsize=labelsize)
par1 = axd["B"].twinx()
par1.set_ylim([0.2, 0.8])
axd["B"].set_ylim([0, 1])
axd["B"].plot(
    cont_data["actual_contingencies_hightT"],
    label="Actual",
    color="k",
    linestyle="--",
    alpha=0.5,
)
axd["B"].plot(
    cont_data["actual_contingencies_lowT"],
    label="Low tone",
    color="gray",
    linestyle="--",
    alpha=0.5,
)
sns.lineplot(data=data_plot, x="trial", y="sahat_2", color="#0A5394", linewidth=2)
par1.tick_params(axis="y", colors="#0A5394", labelsize=ticksize)
axd["B"].tick_params(axis="both", labelsize=ticksize)
axd["B"].set_ylabel("", fontsize=labelsize)
par1.set_ylabel("Estimation\nuncertainty", fontsize=labelsize)
par1.set_yticks([])

axd["B"].set_xlabel("Trial", fontsize=labelsize)
par1.yaxis.label.set_color("#0A5394")
axd["B"].set_xlabel("", fontsize=labelsize)
axd["B"].set_xticks([])
par1.spines["right"].set_edgecolor("#0A5394")

axd["D"].tick_params(
    left=False,
    labelleft=False,
)

axd["B"].tick_params(
    left=False,
    labelleft=False,
)
axd["C"].tick_params(
    left=False,
    labelleft=False,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cax = inset_axes(
    axd["F"],
    width="3%",  # width: 40% of parent_bbox width
    height="100%",  # height: 10% of parent_bbox height
    loc="lower left",
    bbox_to_anchor=(1.1, 0, 1, 1),
    bbox_transform=axd["F"].transAxes,
    borderpad=0,
)

sns.heatmap(
    plot_matrix,
    cmap="cividis",
    center=0,
    annot=True,
    ax=axd["F"],
    vmin=-1,
    vmax=1,
    cbar_ax=cax,
    square=True,
    linewidths=0.5,
    linecolor="black",
    cbar_kws={"label": r"Spearman $\rho$", "shrink": 0.75},
    annot_kws={"fontsize": ticksize - 2},
    fmt=".2f",
)

cax = axd["F"].figure.axes[-1]
cax.yaxis.label.set_size(labelsize)

cax.tick_params(labelsize=ticksize)
axd["F"].set_xticklabels(
    [
        r"$\omega_2$",
        r"$\beta_0$",
        r"$\beta_1$",
        r"$\beta_2$",
        r"$\zeta$",
    ],
    rotation=0,
    fontsize=ticksize,
)
axd["F"].set_yticklabels(
    [
        r"$\omega_2$",
        r"$\beta_0$",
        r"$\beta_1$",
        r"$\beta_2$",
        r"$\zeta$",
    ],
    rotation=0,
    fontsize=ticksize,
)

axd["F"].set_xlabel("Mean recovered parameter", fontsize=labelsize)
axd["F"].set_ylabel("Simulated parameter", fontsize=labelsize)
for _, spine in axd["F"].spines.items():
    spine.set_visible(True)

# Model comparison

famcomp = loadmat(opj(bids_root, "derivatives/VBA_BMC.mat"))

modnames = ["HGF2", "HGF2_PU", "HGF3", "HGF3_PU", "RW", "PH"]

ep = famcomp["out"]["pep"][0][0][0]
ef = [float(ef) * 100 for ef in famcomp["out"]["Ef"][0][0]]

par1 = axd["E"].twinx()
color1 = "#7293cb"
color2 = "#e1974c"

x = np.arange(0.5, (len(ep)) * 0.75, 0.75)
x2 = [c + 0.25 for c in x]
p1 = axd["E"].bar(x, ep, width=0.25, color=color1, linewidth=1, edgecolor="k")
p2 = par1.bar(x2, ef, width=0.25, color=color2, linewidth=1, edgecolor="k")

axd["E"].set_ylim(0, 1)
par1.set_ylim(0, 100)

# host.set_xlabel("Distance")
axd["E"].set_ylabel("Protected exceedance probability", fontsize=labelsize)
par1.set_ylabel("Model Frequency (%)", fontsize=labelsize)

for ax in [par1]:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    plt.setp(ax.spines.values(), visible=False)
    ax.spines["right"].set_visible(True)

axd["E"].yaxis.label.set_color(color1)
par1.yaxis.label.set_color(color2)

axd["E"].spines["left"].set_edgecolor(color1)
par1.spines["right"].set_edgecolor(color2)

axd["E"].set_xticks([i + 0.125 for i in x])
axd["E"].set_xticklabels(modnames, size=ticksize)

axd["E"].tick_params(axis="x", labelsize=ticksize, rotation=30)
axd["E"].set_xlabel("Model", fontsize=labelsize)

axd["E"].tick_params(axis="y", colors=color1, labelsize=ticksize)
par1.tick_params(axis="y", colors=color2, labelsize=ticksize)

# fig.savefig("VBA_model_comparison.png", dpi=800)
axd["E"].set_title(
    "c", fontsize=figurelettersize + 1, fontweight="bold", x=-0.2, y=1.01
)
axd["F"].set_title(
    "d", fontsize=figurelettersize + 1, fontweight="bold", x=-0.2, y=1.01
)
axd["B"].set_title(
    "b", fontsize=figurelettersize + 1, fontweight="bold", x=-0.05, y=1.01
)
# axd["C"].set_title("c", fontsize=figurelettersize, fontweight="bold", x=-0.3, y=1.01)
# axd["d"].set_title("d", fontsize=figurelettersize, fontweight="bold", x=-0.3, y=1.01)

axd["A"].set_visible(False)

plt.savefig(opj(out_fig, "Figure2_model_fitting_results.svg"), dpi=1200, transparent=True)

############################################################################
# Figure 3 - Model fitting results
############################################################################

axd = plt.figure(layout="constrained", figsize=(8, 4), dpi=600).subplot_mosaic(
    """
    AB
    """
)

data = pd.read_csv(opj(bids_root, "derivatives", "figures", "model_based_data.csv"))
data_high = data[data["pain"] == 1]
data_low = data[data["pain"] == 0]
# Plot slopes

colors = sns.color_palette("Blues", 50)
for idx, p in enumerate(data["participant_id"].unique()):
    pfile = data_low[data_low["participant_id"] == p]

    sns.regplot(
        data=pfile,
        x="sahat_2_log",
        y="ratings",
        color=colors[idx],
        scatter=False,
        ci=None,
        truncate=True,
        line_kws={"alpha": 0.3, "linewidth": 1},
        ax=axd["A"],
    )

from statsmodels.formula.api import mixedlm

# Fit for low
md = mixedlm(
    "ratings ~ sahat_2_log",
    data_low,
    re_formula="sahat_2_log",
    groups=data_low["participant_id"],
)
mdf = md.fit()

np.min(data_low["sahat_2_log"]), np.max(data_low["sahat_2_log"])
# Create line with average slope
x = np.linspace(-3, 2, 100)
y = mdf.params["sahat_2_log"] * x + mdf.params["Intercept"]
sns.regplot(
    x=x,
    y=y,
    scatter=False,
    ci=None,
    truncate=True,
    line_kws={"alpha": 1, "color": "darkblue", "linewidth": 3},
    ax=axd["A"],
)

colors = sns.color_palette("Reds", 50)
for idx, p in enumerate(data["participant_id"].unique()):
    pfile = data_high[data_high["participant_id"] == p]
    sns.regplot(
        data=pfile,
        x="sahat_2_log",
        y="ratings",
        color=colors[idx],
        scatter=False,
        ci=None,
        truncate=True,
        line_kws={"alpha": 0.3, "linewidth": 1.5},
        ax=axd["A"],
    )

# Fit for high
md = mixedlm(
    "ratings ~ sahat_2_log",
    data_high,
    re_formula="sahat_2_log",
    groups=data_low["participant_id"],
)
mdf = md.fit()

# Create line with average slope
x = np.linspace(-3, 2, 100)
y = mdf.params["sahat_2_log"] * x + mdf.params["Intercept"]
sns.regplot(
    x=x,
    y=y,
    scatter=False,
    ci=None,
    color="darkred",
    truncate=True,
    line_kws={"alpha": 1, "color": "darkred", "linewidth": 3},
    ax=axd["A"],
)

axd["A"].set_xlabel("Estimation uncertainty", fontsize=12)
axd["A"].set_ylabel("Pain ratings", fontsize=12)
axd["A"].set_xticks([-3, 2], ["Low", "High"])
axd["A"].set_ylim(0, 100)
axd["A"].set_xlim(-3, 2)

colors = sns.color_palette("Blues", 50)
for idx, p in enumerate(data["participant_id"].unique()):
    pfile = data_low[data_low["participant_id"] == p]

    sns.regplot(
        data=pfile,
        x="abs_da1",
        y="ratings",
        color=colors[idx],
        scatter=False,
        ci=None,
        truncate=True,
        line_kws={"alpha": 0.3, "linewidth": 1},
        ax=axd["B"],
    )

# Fit for low
md = mixedlm(
    "ratings ~ abs_da1",
    data_low,
    re_formula="abs_da1",
    groups=data_low["participant_id"],
)
mdf = md.fit()

# Create line with average slope
x = np.linspace(0, 1, 100)
y = mdf.params["abs_da1"] * x + mdf.params["Intercept"]
sns.regplot(
    x=x,
    y=y,
    scatter=False,
    ci=None,
    truncate=True,
    line_kws={"alpha": 1, "color": "darkblue", "linewidth": 3},
    ax=axd["B"],
)

colors = sns.color_palette("Reds", 50)
for idx, p in enumerate(data["participant_id"].unique()):
    pfile = data_high[data_high["participant_id"] == p]
    sns.regplot(
        data=pfile,
        x="abs_da1",
        y="ratings",
        color=colors[idx],
        scatter=False,
        ci=None,
        truncate=True,
        line_kws={"alpha": 0.3, "linewidth": 1.5},
        ax=axd["B"],
    )

# Fit for high
md = mixedlm(
    "ratings ~ abs_da1",
    data_high,
    re_formula="abs_da1",
    groups=data_low["participant_id"],
)
mdf = md.fit()
mdf.summary()

# Create line with average slope
y = mdf.params["abs_da1"] * x + mdf.params["Intercept"]
sns.regplot(
    x=x,
    y=y,
    scatter=False,
    ci=None,
    color="darkred",
    truncate=True,
    line_kws={"alpha": 1, "color": "darkred", "linewidth": 3},
    ax=axd["B"],
)

axd["B"].set_xlabel("Absolute prediction errors", fontsize=12)
axd["B"].set_ylabel("Pain ratings", fontsize=12)
axd["B"].set_xticks([0, 1], ["Low", "High"])
axd["B"].set_ylim(0, 100)
axd["B"].set_xlim(0, 1)

axd["A"].set_title(
    "a", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

axd["B"].set_title(
    "b", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

plt.tight_layout()
plt.savefig(opj(out_fig, "Figure3_uncertainty_prederrors.svg"), dpi=1200, transparent=True)

from scipy import stats
from statsmodels.stats.multitest import multipletests

# Store p-values and r-values
p_values = []
r_values = []
quest_labels = []
dagagp_quests = []

# Ensure data has 'participant_id' only as a column
if "participant_id" in data.index.names and "participant_id" in data.columns:
    data = data.drop(columns=["participant_id"]).reset_index()
elif "participant_id" in data.index.names:
    data = data.reset_index()

for quest in ["pcs_tot", "stai_state_tot", "stai_trait_tot", "bdi_tot"]:
    dagagp = data[["participant_id", quest, "om2"]].groupby("participant_id").mean().reset_index()
    dagagp_quest = dagagp.dropna(subset=[quest])
    r, p = stats.pearsonr(dagagp_quest[quest], dagagp_quest["om2"])

    r_values.append(r)
    p_values.append(p)
    quest_labels.append(quest)
    dagagp_quests.append(dagagp_quest)

# Apply Holm correction
rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')

# Create figure and mosaic
fig = plt.figure(layout="constrained", figsize=(8, 2.25), dpi=600)
axd = fig.subplot_mosaic("cdef")

# Plot each subplot
for i, (quest, label, textx, ax) in enumerate(zip(
    quest_labels,
    ["PCS total", "STAI state total", "STAI trait total", "BDI total"],
    [16, 37, 41, 9.5],
    [axd["c"], axd["d"], axd["e"], axd["f"]],
)):
    sns.regplot(
        data=dagagp_quests[i],
        x=quest,
        y="om2",
        label="OM2",
        ax=ax,
        scatter_kws={"alpha": 0.5, "s": 10},
    )
    
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel(r"$\omega_2$ (A.U.)", fontsize=10)

    r_text = f"r({len(dagagp_quests[i])-2})= {r_values[i]:.2f},\np = {pvals_corrected[i]:.3f}"
    if rejected[i]:
        r_text += " *"  # Mark significant results
    ax.text(textx, -1.4, r_text, fontsize=8)

# Add panel labels (c, d, e, f) outside top-left
for i, key in enumerate(["c", "d", "e", "f"]):
    ax = axd[key]
    label = chr(99 + i)  # 'c' = 99 in ASCII
    ax.text(
        -0.4, 1,  # x, y in axes fraction coordinates
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="right"
    )

plt.savefig(opj(out_fig, "Figure3_questionnaires_omega.svg"), dpi=1200, transparent=True)

############################################################################
# Supplementary figures
############################################################################

participants = pd.read_csv(opj(bids_root, "participants.tsv"), sep="\t")
participants = participants[participants.excluded == 0]["participant_id"].values

# Parameters recovery
win_model = "HGF2"
all_frames = []
for p in participants:

    # Load actual parameters of winning model
    obs_params = pd.read_csv(
        opj(
            bids_root,
            "derivatives",
            p,
            p + "_" + "HGF2" + "_uncertainty_obsparams.csv",
        )
    )
    perc_params = pd.read_csv(
        opj(
            bids_root,
            "derivatives",
            p,
            p + "_" + "HGF2" + "_uncertainty_prcparams.csv",
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

# Make regplot for each parameter and add correlation coefficient and p value
fig = plt.figure(figsize=(10, 6))
axd = fig.subplot_mosaic(
    """
    AB
    """
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
        "/Users/meganelacombe-thibault/Downloads/Étude 1/painuncertainty_codeMars2025/derivatives/VBA_BMC_"
        + sim_mod
        + "_sim_recovery.mat"
    )

    ep = famcomp["out"]["pep"][0][0][0]
    ef = [float(ef) * 100 for ef in famcomp["out"]["Ef"][0][0]]

    frame_ef[sim_mod] = ef
    frame_ep[sim_mod] = ep

frame_ef = frame_ef.astype(float) / 100
frame_ep = frame_ep.astype(float)

sns.heatmap(
    frame_ef,
    cmap="viridis",
    center=0.5,
    annot=True,
    ax=axd["A"],
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="black",
    cbar_kws={"label": "Proportion", "shrink": 0.6},
    annot_kws={"fontsize": 10},
    fmt=".2f",
)
cbar = axd["A"].collections[0].colorbar
cbar.ax.yaxis.label.set_size(10)
axd["A"].set_xticklabels(labels, rotation=90, fontsize=10)
axd["A"].set_yticklabels(labels, rotation=0, fontsize=10)
axd["A"].set_xlabel("Simulated model", fontsize=10)
axd["A"].set_ylabel("Recovered model", fontsize=10)
for _, spine in ax.spines.items():
    spine.set_visible(True)

sns.heatmap(
    frame_ep,
    cmap="viridis",
    center=0.5,
    annot=True,
    ax=axd["B"],
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="black",
    cbar_kws={"label": "Protected exceedance\nprobability", "shrink": 0.6},
    annot_kws={"fontsize": 10},
    fmt=".2f",
)
cbar = axd["B"].collections[0].colorbar
cbar.ax.yaxis.label.set_size(10)
axd["B"].set_xticklabels(labels, rotation=90, fontsize=10)
axd["B"].set_yticklabels(labels, rotation=0, fontsize=10)
axd["B"].set_xlabel("Simulated model", fontsize=10)
axd["B"].set_ylabel("Recovered model", fontsize=10)
for _, spine in ax.spines.items():
    spine.set_visible(True)

axd["A"].set_title(
    "a", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

axd["B"].set_title(
    "b", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

plt.tight_layout()
plt.savefig(
    opj(out_fig, "Supp_figure_1.png"),
    dpi=1200,

    transparent=True,
)
plt.show()

fig = plt.figure(figsize=(10, 2.4))
axd = fig.subplot_mosaic(
    """
    CDEFG
    """
)

params = ["om2", "be0", "be1", "be2", "ze"]

labels =[
        r"$\omega_2$",
        r"$\beta_0$",
        r"$\beta_1$",
        r"$\beta_2$",
        r"$\zeta$",
    ]

axes_positions = ["C", "D", "E", "F", "G"]

for i, (param, label) in enumerate(zip(params, labels)):
    ax = axd[axes_positions[i]]
    sns.regplot(x=param, y="recovered_" + param, data=all_frames, ax=ax, ci=95)

    # Calculate the R value
    r_value, p_value = stats.pearsonr(all_frames[param], all_frames["recovered_" + param])

    # Set labels and title for the subplot
    ax.set_xlabel("Simulated " + label, fontsize=10)
    ax.set_ylabel("Mean recovered " + label, fontsize=10)
    p_string = "p < 0.001" if p_value < 0.001 else f"{p_value:.2e}"
    ax.text(0.05, 0.95, f"R = {r_value:.2f}\n{p_string}", transform=ax.transAxes, fontsize=10, ha='left', va='top')

axd["C"].set_title(
    "c", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

axd["D"].set_title(
    "d", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

axd["E"].set_title(
    "e", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

axd["F"].set_title(
    "f", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

axd["G"].set_title(
    "g", fontsize=figurelettersize + 1, fontweight="bold", x=-0.15, y=1.01
)

plt.tight_layout()
plt.savefig(
    opj(out_fig, "Supp_figures_2.png"),
    dpi=1200,

    transparent=True,
)
plt.show()