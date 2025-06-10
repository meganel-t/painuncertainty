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
import pingouin as pg
from scipy import stats
from statsmodels.formula.api import mixedlm
import urllib.request
import matplotlib.font_manager as fm
import matplotlib
from matplotlib.ticker import FormatStrFormatter

import pickle


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

# Get nice font
urllib.request.urlretrieve(
    "https://github.com/gbif/analytics/raw/master/fonts/Arial%20Narrow.ttf",
    opj(out_fig, "arialnarrow.ttf"),
)
fm.fontManager.addfont(opj(out_fig, "arialnarrow.ttf"))
matplotlib.rc("font", family="Arial Narrow")

# Load data
data = pd.read_csv(opj(bids_root, "derivatives", "all_part_data.csv"))

# Exclude participants using participants.tsv
participants = pd.read_csv(opj(bids_root, "participants.tsv"), sep="\t")
participants = participants[participants.excluded == 0]
data = data[data["participant"].isin(participants["participant_id"])]


# Collect model-based trajectories and add to data
win_model = "HGF2_uncertainty"
all_files = []
for p in data.participant.unique():
    pfile = pd.read_csv(
        opj(bids_root, "derivatives", p, p + "_" + win_model + "_traj.csv")
    )
    pfile["participant"] = p
    pfile["trial"] = np.arange(1, len(pfile) + 1)

    pfile["ratings_mean_centered"] = np.where(
        pfile["pain"] == 1,
        pfile["ratings"] - np.mean(pfile[pfile["pain"] == 1]["ratings"]),
        pfile["ratings"] - np.mean(pfile[pfile["pain"] == 0]["ratings"]),
    )

    all_files.append(pfile)

# Concatenate all files
all_files = pd.concat(all_files).reset_index(drop=True)
all_files = all_files.sort_values(by=["participant", "trial"]).reset_index(drop=True)
data = data.sort_values(by=["participant", "trial"]).reset_index(drop=True)

# MAke sure everything is in the right order
assert list(data["participant"].values) == list(all_files["participant"].values)
assert list(data["trial"].values) == list(all_files["trial"].values)

# Add the model-based data to the data
data = pd.concat([data, all_files], axis=1).drop_duplicates()
data = data.loc[:, ~data.columns.duplicated()].copy()
data.index = data["participant"]

# Drop the participant column
data = data.loc[:, ~data.columns.isin(["participant"])].copy()

# Add the fitted parameters data
for p in data.index.unique():
    pfile_prc = pd.read_csv(
        opj(
            bids_root,
            "derivatives",
            p,
            p + "_" + win_model + "_prcparams.csv",
        )
    )
    for c in pfile_prc.columns:
        data.loc[p, c] = pfile_prc[c].values[0]
    pfile_obs = pd.read_csv(
        opj(
            bids_root,
            "derivatives",
            p,
            p + "_" + win_model + "_obsparams.csv",
        )
    )
    for c in pfile_obs.columns:
        data.loc[p, c] = pfile_obs[c].values[0]

# Calculate stats on number of trials in model fitting
all_remaining = []
for p in data.index.unique():
    pfile = pd.read_csv(opj(bids_root, "derivatives", p, p + "_ignored_trials.csv"))
    all_remaining.append(pfile["remaining"].values[0])

model_fit_trial_stats = pd.DataFrame(
    {
        "participant": data.index.unique(),
        "remaining_trials": all_remaining,
    }
)
model_fit_trial_stats.to_csv(opj(out_stats, "model_fit_trial_stats.csv"), index=False)
model_fit_trial_stats.describe().to_csv(
    opj(out_stats, "model_fit_trial_stats_desc.csv"), index=False
)


######################################################################
# Model free figures and analyses
######################################################################

# Use any participant to get the task structure
actual_contingencies_hightT = []
actual_contingencies_lowT = []
blocks = data["block_name"].unique()

# Select one participant to get the actual contingencies
part_dat = data[data.index == "sub-015"]
for b in blocks:
    datab = part_dat[part_dat["block_name"] == b]  # Data for this block
    ntrials = list(datab["block_ntrials"])[0]  # ntrials for this block
    # Number of high pain following a low tone divided by 50% of ntrials
    prob_highp_lowt = len(
        datab[(datab["stimulus_"] == "forte") & (datab["cue_"] == "grave")]
    ) / (ntrials / 2)
    # Number of high pain following a high tone divided by 50% of ntrials
    prob_highp_hight = len(
        datab[(datab["stimulus_"] == "forte") & (datab["cue_"] == "aigu")]
    ) / (ntrials / 2)
    actual_contingencies_lowT += [prob_highp_lowt] * int(ntrials)
    actual_contingencies_hightT += [prob_highp_hight] * int(ntrials)


cont_data = pd.DataFrame(
    {
        "trial": np.arange(1, len(actual_contingencies_lowT) + 1),
        "actual_contingencies_lowT": actual_contingencies_lowT,
        "actual_contingencies_hightT": actual_contingencies_hightT,
    }
)
cont_data.to_csv(opj(out_fig, "actual_contingencies.csv"), index=False)

# Add the model free contingencies to the data
# If prob 20
# E = aigu mod/grave fort
# UE = aigu fort/grave mod
# If prob 80
# E = aigu fort/grave mod
# UE = aigu mod/grave fort
# If prob 50
# All  = N
type = []
for idx in range(len(data)):
    row = data.iloc[idx]
    if row["block_trials_file"] == "prob20hthp_12trials.xlsx":
        if row["stimulus_"] == "forte":
            if row["cue_"] == "grave":
                type.append("E")
            else:
                type.append("UE")
        elif row["stimulus_"] == "moderee":
            if row["cue_"] == "aigu":
                type.append("E")
            else:
                type.append("UE")
    elif row["block_trials_file"] == "prob80hthp_12trials.xlsx":
        if row["stimulus_"] == "forte":
            if row["cue_"] == "aigu":
                type.append("E")
            else:
                type.append("UE")
        elif row["stimulus_"] == "moderee":
            if row["cue_"] == "grave":
                type.append("E")
            else:
                type.append("UE")
    elif row["block_trials_file"] == "prob50_50_4trials.xlsx":
        type.append("N")

data["type"] = type

data["rt_fit"] = data["rt"].copy()

# New column for RT
data["rt"] = data["choice_pain.rt"]


# Average rate across participants and conditions
dat_avg_rate = (
    data.groupby(["participant", "stimulus_", "type"])
    .mean(numeric_only=True)
    .reset_index()
)

# Figure RT by type and stimulus across participants
dat_avg_rate["rt_ms"] = dat_avg_rate["choice_pain.rt"] * 1000

# RT figure with line break
fig, (ax, ax2) = plt.subplots(
    2,
    1,
    figsize=(2.5, 2.5),
    sharex=True,
    facecolor="w",
    gridspec_kw={"hspace": 0.01, "height_ratios": [30, 2]},
)
fig.patch.set_facecolor("#EFEFEF")
ax2.set_ylim(0, 700)
ax2.set_yticks([0])
ax.set_ylim(700, 900)
ax2.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)


d = 0.02  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
ax.plot((-d, +d), (0, 0), **kwargs)  # top-left diagonal
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (0.5, 0.5), **kwargs)  # bottom-left diagonal

plt.sca(ax)
sns.pointplot(
    data=dat_avg_rate,
    palette="Set1",
    x="type",
    y="rt_ms",
    hue="stimulus_",
    zorder=10,
    dodge=True,
    scale=1,
    ci=68,
)

ax.set_xlabel("", fontsize=14)
ax.set_xticks([0, 1, 2], ["Expected", "Neutral", "Unexpected"])
ax.set_ylim([700, 900])

ax.set_ylabel("Response time (ms)", fontsize=12)
ax.tick_params(labelsize=10)
ax.tick_params(axis="x", labelsize=10, rotation=30, size=0)
ax2.tick_params(axis="x", labelsize=10, rotation=30)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

h, l = ax.get_legend_handles_labels()
ax.legend(h[0:2], ["High pain", "Low pain"], title="", fontsize=10, frameon=False)
ax.set_facecolor("#EFEFEF")
ax2.set_facecolor("#EFEFEF")
# Set the axis color
ax2.spines["bottom"].set_color("black")

fig.savefig(
    opj(out_fig, "rt_bytypestin_across_participants.png"), dpi=800, bbox_inches="tight"
)

# Statistics on RT
out = pg.rm_anova(
    data=dat_avg_rate,
    dv="choice_pain.rt",
    within=["type", "stimulus_"],
    subject="participant",
    detailed=True,
)
out.to_csv(
    opj(out_stats, "rm_anova_rt_bytype_by_stimulus_across_participants.tsv"),
    sep="\t",
    index=False,
)
out = pg.pairwise_tests(
    data=dat_avg_rate,
    dv="choice_pain.rt",
    within=["type", "stimulus_"],
    subject="participant",
)
out.to_csv(
    opj(out_stats, "pairwise_tests_rt_bytype_bystimulus_across_participants.tsv"),
    sep="\t",
    index=False,
)

# Proportion of errors
dat_avg_rate["prop_errors"] = 1 - dat_avg_rate["correct"]

# Figure RT by type and stimulus across participants
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
fig.patch.set_facecolor("#EFEFEF")
sns.pointplot(
    data=dat_avg_rate,
    palette="Set1",
    x="type",
    y="prop_errors",
    hue="stimulus_",
    zorder=10,
    dodge=True,
    scale=1,
    ci=68,
)
ax.set_xlabel("", fontsize=14)
ax.set_xticks([0, 1, 2], ["Expected", "Neutral", "Unexpected"])
ax.set_ylim([0, 0.1])
ax.set_ylabel("Proportion of errors", fontsize=12)
ax.tick_params(labelsize=10)
ax.tick_params(axis="x", labelsize=10, rotation=0)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))
h, l = ax.get_legend_handles_labels()
ax.legend(h[0:2], ["High pain", "Low pain"], title="", fontsize=10, frameon=False)
ax.set_facecolor("#EFEFEF")
fig.tight_layout()
fig.savefig(
    opj(out_fig, "correct_bytypestin_across_participants.png"),
    dpi=800,
    bbox_inches="tight",
    transparent=True,
)

# Statistics on errors
out = pg.rm_anova(
    data=dat_avg_rate,
    dv="correct",
    within=["type", "stimulus_"],
    subject="participant",
    detailed=True,
)
out.to_csv(
    opj(out_stats, "rm_anova_correct_bytype_by_stimulus_across_participants.tsv"),
    sep="\t",
    index=False,
)
out = pg.pairwise_tests(
    data=dat_avg_rate, dv="correct", within=["type", "stimulus_"], subject="participant"
)
out.to_csv(
    opj(out_stats, "pairwise_tests_correct_bytype_bystimulus_across_participants.tsv"),
    sep="\t",
    index=False,
)


# Ratings
# Figure pain rating by type and stimulus across participants
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
sns.pointplot(
    data=dat_avg_rate,
    palette="Set1",
    x="type",
    y="pain_rating",
    hue="stimulus_",
    zorder=10,
    dodge=False,
    join=True,
    scale=1,
    ci=68,
)
ax.set_xlabel("", fontsize=14)
ax.set_xticks([0, 1, 2], ["Expected", "Neutral", "Unexpected"])
ax.set_ylabel("Pain rating", fontsize=12)
ax.tick_params(labelsize=10)
ax.tick_params(axis="x", labelsize=10, rotation=30)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

h, l = ax.get_legend_handles_labels()
ax.legend(h, ["High pain", "Low pain"], title="", fontsize=10, frameon=False)
fig.savefig(
    opj(out_fig, "pain_bytypestin_across_participants.png"),
    dpi=800,
    bbox_inches="tight",
)

# Same figure but mean centered
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
fig.patch.set_facecolor("#EFEFEF")
sns.pointplot(
    data=dat_avg_rate,
    palette="Set1",
    x="type",
    hue="stimulus_",
    y="ratings_mean_centered",
    zorder=10,
    dodge=False,
    join=True,
    scale=1,
    ci=68,
)
ax.set_xlabel("", fontsize=14)
ax.set_xticks([0, 1, 2], ["Expected", "Neutral", "Unexpected"])
ax.set_ylabel("Pain rating (mean centered)", fontsize=12)
ax.tick_params(labelsize=10)
ax.tick_params(axis="x", labelsize=10, rotation=0)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.01f"))
h, l = ax.get_legend_handles_labels()
ax.legend(h, ["High pain", "Low pain"], title="", fontsize=10, frameon=False)
ax.set_facecolor("#EFEFEF")
fig.tight_layout()
fig.savefig(
    opj(out_fig, "painmeancentered_bytype_across_participants.png"),
    dpi=800,
    bbox_inches="tight",
    transparent=True,
)


# Statistics on ratings
out = pg.rm_anova(
    data=dat_avg_rate,
    dv="pain_rating",
    within=["type", "stimulus_"],
    subject="participant",
    detailed=True,
)
out.to_csv(
    opj(out_stats, "rm_anova_pain_bytype_by_stimulus_across_participants.tsv"),
    sep="\t",
    index=False,
)

out = pg.pairwise_tests(
    data=dat_avg_rate,
    dv="pain_rating",
    within=["stimulus_", "type"],
    subject="participant",
)
out.to_csv(
    opj(out_stats, "pairwise_tests_pain_bytype_bystimulus_across_participants.tsv"),
    sep="\t",
    index=False,
)

# Save data for model free plots in figures script
dat_avg_rate.to_csv(opj(out_fig, "model_free_data.csv"), index=False)


######################################################################
# Model based analyses on ratings
######################################################################

data["participant_id"] = list(data.index)

# Log transform sahat_2 and suprise
data["sahat_2_log"] = np.log(data["sahat_2"])

# Absolute value of prediction errors
data["abs_da1"] = np.abs(data["da_1"])

# Square sahat_1
data["sahat_1_sq"] = data["sahat_1"] ** 2

# Seperate low and high
data_high = data[data["pain"] == 1]
data_low = data[data["pain"] == 0]


# Save data for plot
data.to_csv(opj(out_fig, "model_based_data.csv"), index=False)
## Ratings

# Absoulte prediction error
md = mixedlm(
    "ratings~ pain*abs_da1",
    data,
    re_formula="pain*abs_da1",  # Simpler covariance structure to prevent convergence issues
    groups=data["participant_id"],
)
mdf = md.fit(method="lbfgs")
print(mdf.summary())
mdf.summary().tables[1].to_csv(opj(out_stats, "mixedlm_ratings_da_pain.csv"), sep="\t")

# Fit for low
md = mixedlm(
    "ratings ~ abs_da1",
    data_low,
    re_formula="abs_da1",
    groups=data_low["participant_id"],
)
mdf = md.fit(method="lbfgs")
print(mdf.summary())
# Save summary as csv
mdf.summary().tables[1].to_csv(opj(out_stats, "mixedlm_ratings_da_low.csv"), sep="\t")

# Fit for high
md = mixedlm(
    "ratings ~ abs_da1",
    data_high,
    re_formula="abs_da1",
    groups=data_high["participant_id"],
)
mdf = md.fit(method="lbfgs")
print(mdf.summary())
# Save summary as csv
mdf.summary().tables[1].to_csv(opj(out_stats, "mixedlm_ratings_da_high.csv"), sep="\t")


# Test 1st level uncertainty*pain interaction
md = mixedlm(
    "ratings~ sahat_1_sq*pain",
    data,
    re_formula="sahat_1_sq*pain",
    groups=data["participant_id"],
)
mdf = md.fit(method="lbfgs")
print(mdf.summary())
# Save summary as csv
mdf.summary().tables[1].to_csv(
    opj(out_stats, "mixedlm_ratings_sahat1_pain.csv"), sep="\t"
)

# Test 2nd level uncertainty*pain interaction
md = mixedlm(
    "ratings~ pain*sahat_2_log",
    data,
    re_formula="pain*sahat_2_log",
    groups=data["participant_id"],
)
mdf = md.fit(method="lbfgs")
print(mdf.summary())
# Save summary as csv
mdf.summary().tables[1].to_csv(
    opj(out_stats, "mixedlm_ratings_sahat2_pain.csv"), sep="\t"
)

# Test 2nd level uncertainty*pain interaction
# Fit for low
md = mixedlm(
    "ratings ~ sahat_2_log",
    data_low,
    re_formula="sahat_2_log",
    groups=data_low["participant_id"],
)
mdf = md.fit(method="lbfgs")
print(mdf.summary())
# Save summary as csv
mdf.summary().tables[1].to_csv(
    opj(out_stats, "mixedlm_ratings_sahat2_low.csv"), sep="\t"
)

# Fit for high
md = mixedlm(
    "ratings ~ sahat_2_log",
    data_high,
    re_formula="sahat_2_log",
    groups=data_high["participant_id"],
)
mdf = md.fit(method="lbfgs")
print(mdf.summary())
# Save summary as csv
mdf.summary().tables[1].to_csv(
    opj(out_stats, "mixedlm_ratings_sahat2_high.csv"), sep="\t"
)

# Correlation with questionnaires
for quest in ["pcs_tot", "stai_state_tot", "stai_trait_tot", "bdi_tot"]:
    # Questionnaire plots
    dagagp = data.groupby("participant").mean([quest, "om2"]).reset_index()
    dagagp_quest = dagagp.dropna(subset=[quest])
    r2, p2 = stats.pearsonr(dagagp_quest[quest], dagagp_quest["om2"])

    pd.DataFrame(
        {
            "pearson_r": [r2],
            "p_value": [p2],
        }
    ).to_csv(opj(out_stats, "corr_om2_" + quest + ".csv"), index=False)
