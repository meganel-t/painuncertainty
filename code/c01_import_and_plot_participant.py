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

# import MNE to get the report
from mne.report import Report
from scipy import stats

# Path for the bids root (upper folder from code)
bids_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
)

# Load participant frame
participants = pd.read_csv(opj(bids_root, "sourcedata", "participants.csv"))
participants.index = participants["participant_id"]

# Find all participants number using data in sourcedata
part = [
    s
    for s in os.listdir(bids_root + "/sourcedata")
    if "sub-" in s and s[:7] not in ["sub-055", "sub-059"]
]  # Excluded for low accuracy
# Sort participants
part.sort()

# Print how many participants found
print("Found " + str(len(part)) + " participants")

# Make derivative folder
if not os.path.exists(opj(bids_root, "derivatives")):
    os.mkdir(opj(bids_root, "derivatives"))
    os.mkdir(opj(bids_root, "derivatives", "figures"))

# Loop through participants
all_part_data = []
for p in part:
    # Create an html report to save figures
    report = Report(subject=p[:7], title="PainUncertainty_" + p[:7])

    # Find the main task csv in the sourcedata/part folder
    taskfile = [
        s
        for s in os.listdir(bids_root + "/sourcedata/" + p)
        if "PainUncertainty" in s and ".csv" in s
    ]

    # Make sure there is only one file
    assert len(taskfile) == 1, "Found more than one task file"

    # Load the task file
    task = pd.read_csv(bids_root + "/sourcedata/" + p + "/" + taskfile[0])

    # Drop practice trials (block_name = nan)
    task = task[~task["block_name"].isna()].reset_index(drop=True)

    # Drop pause rows (stimulus = nan)
    task = task[~task["stimulus_"].isna()].reset_index(drop=True)
    participants.loc[p[:7], "ntrials"] = len(task)

    # Make sure we have 192 trials
    assert len(task) == 192, "Found " + str(len(task)) + " trials"

    # Save file in the bids root
    if not os.path.exists(opj(bids_root, p[:7])):
        os.mkdir(opj(bids_root, p[:7]))
    if not os.path.exists(opj(bids_root, "derivatives", p[:7])):
        os.mkdir(opj(bids_root, "derivatives", p[:7]))

    # Make some plots for this participant
    # Plot the pain ratings
    fig = plt.figure()
    sns.lineplot(
        data=task, x=np.arange(1, 193), y="ratingScale.response", hue="stimulus_"
    )
    plt.xlabel("Trials")
    plt.ylabel("Pain rating (VAS)")
    plt.title("Pain ratings " + p[:7])
    report.add_figure(fig, title="Pain ratings " + p[:7])
    participants.loc[p[:7], "mean_hpain_rating"] = np.mean(
        task[task["stimulus_"] == "forte"]["ratingScale.response"]
    )
    participants.loc[p[:7], "mean_lpain_rating"] = np.mean(
        task[task["stimulus_"] == "moderee"]["ratingScale.response"]
    )

    # Plot the response times
    fig = plt.figure()
    sns.lineplot(data=task, x=np.arange(1, 193), y="choice_pain.rt", hue="stimulus_")
    plt.xlabel("Trials")
    plt.ylabel("Response time (s)")
    plt.title("Response time " + p[:7])
    report.add_figure(fig, title="Response time" + p[:7])
    participants.loc[p[:7], "mean_hpain_rt"] = np.mean(
        task[task["stimulus_"] == "forte"]["choice_pain.rt"]
    )
    participants.loc[p[:7], "mean_lpain_rt"] = np.mean(
        task[task["stimulus_"] == "moderee"]["choice_pain.rt"]
    )

    # Repsonse times distribution
    fig = plt.figure()
    sns.histplot(
        task[task["stimulus_"] == "forte"]["choice_pain.rt"], color="red", label="Forte"
    )
    sns.histplot(
        task[task["stimulus_"] == "moderee"]["choice_pain.rt"],
        color="blue",
        label="Moderee",
    )
    plt.legend()
    report.add_figure(fig, title="Response time distributions " + p[:7])

    # Get accuracy and plot
    task["correct"] = 999
    # Assign accuracy to each trial
    task.loc[
        (task["stimulus_"] == "forte") & (task["choice_pain.keys"] == "m"), "correct"
    ] = 1
    task.loc[
        (task["stimulus_"] == "forte") & (task["choice_pain.keys"] == "n"), "correct"
    ] = 0
    task.loc[
        (task["stimulus_"] == "moderee") & (task["choice_pain.keys"] == "n"), "correct"
    ] = 1
    task.loc[
        (task["stimulus_"] == "moderee") & (task["choice_pain.keys"] == "m"), "correct"
    ] = 0
    # When no response, incorrect
    task.loc[task["choice_pain.keys"].isna(), "correct"] = 0
    task.loc[task["choice_pain.keys"] == "None", "correct"] = 0

    task["pain_rating"] = task["ratingScale.response"]
    task["pain_rating_z"] = stats.zscore(task["ratingScale.response"])

    task["accuracy"] = np.sum(task["correct"]) / len(task)

    print("Accuracy : " + str(task["accuracy"][0]))
    participants.loc[p[:7], "accuracy"] = np.sum(task["correct"]) / len(task)
    participants.loc[p[:7], "accuracy_high"] = np.sum(
        task[task["stimulus_"] == "forte"]["correct"]
    ) / len(task[task["stimulus_"] == "forte"])
    participants.loc[p[:7], "accuracy_low"] = np.sum(
        task[task["stimulus_"] == "moderee"]["correct"]
    ) / len(task[task["stimulus_"] == "moderee"])
    participants.loc[p[:7], "n_correct"] = np.sum(task["correct"])
    participants.loc[p[:7], "n_correct_above200"] = np.sum(task["correct"]) - np.sum(
        task["choice_pain.rt"] > 0.2
    )

    fig = plt.figure()
    sns.barplot(data=task, y="correct", x="stimulus_")
    plt.text(
        0.5, 0.5, "Accuracy : " + str(np.round(task["accuracy"][0], 2)), fontsize=15
    )
    plt.xlabel("Stimulus")
    plt.ylabel("Proporiton of correct answers")
    report.add_figure(fig, title="Accuracy " + p[:7])

    fig = plt.figure()
    sns.stripplot(data=task, x="block_name", y="choice_pain.rt", linewidth=1)
    sns.boxplot(data=task, x="block_name", y="choice_pain.rt")
    report.add_figure(fig, title="Response time by block " + p[:7])

    fig = plt.figure()
    sns.stripplot(data=task, x="block_prob_highP_lowT", y="choice_pain.rt", linewidth=1)
    sns.boxplot(data=task, x="block_prob_highP_lowT", y="choice_pain.rt")
    report.add_figure(fig, title="Response time by contingencies " + p[:7])

    # Save report
    report.save(
        opj(bids_root, "derivatives", p[:7], p[:7] + "_report.html"),
        open_browser=False,
        overwrite=True,
    )
    # Add to list of all dataframes
    task["trial"] = np.arange(1, len(task) + 1)
    all_part_data.append(task.copy())

    task.to_csv(
        opj(bids_root, p[:7], p[:7] + "_task-PainUncertainty_beh.tsv"),
        sep="\t",
        index=False,
    )

    plt.close("all")

# Save all data
all_part_data = pd.concat(all_part_data)
participants.to_csv(opj(bids_root, "participants.tsv"), index=False, sep="\t")
all_part_data.to_csv(opj(bids_root, "derivatives", "all_part_data.csv"), index=False)

# RECODE STAI QUESTIONNAIRE AND ADD QUESTIONNAIRE SCORES TO PARTICIPANTS
iasta = pd.read_csv(opj(bids_root, "sourcedata", "iasta.csv"))

recode_map = {1: 4, 2: 3, 3: 2, 4: 1}
columns_to_recode = [
    "1.Je me sens calme",
    "1. Je me sens en sécurité",
    "1. Je me sens tranquille",
    "1. Je me sens comblé(e)",
    "1. Je me sens à l'aise",
    "1. Je me sens sûr(e) de moi",
    "1. Je suis détendu(e)",
    "1. Je me sens satisfait(e)",
    "1. Je sens que j'ai les nerfs solides",
    "1.Je me sens bien",
    "2.Je me sens bien",
    "2. Je me sens content(e) de moi-même",
    "2. Je me sens reposé(e)",
    "2. Je suis d'un grand calme",
    "2. Je suis heureux(se)",
    "2.Je me sens en sécurité",
    "2.Prendre des décisions m'est facile",
    "2. Je suis satisfait(e)",
    "2.Je suis une personne qui a les nerfs solides",
]

iasta[columns_to_recode] = iasta[columns_to_recode].replace(recode_map)

# Columns for IASTAE_total and IASTAT_total calculations
columns_iastae = [
    "1.Je me sens calme",
    "1. Je me sens en sécurité",
    "1. Je suis tendu(e)",
    "1. Je me sens surmené(e)",
    "1. Je me sens tranquille",
    "1. Je me sens bouleversé(e)",
    "1. Je suis préoccupé(e) actuellement par des malheurs possibles",
    "1. Je me sens comblé(e)",
    "1. Je me sens effrayé(e)",
    "1. Je me sens à l'aise",
    "1. Je me sens sûr(e) de moi",
    "1. Je me sens nerveux(se)",
    "1. Je suis affolé(e)",
    "1. Je me sens indécis(e)",
    "1. Je suis détendu(e)",
    "1. Je me sens satisfait(e)",
    "1. Je suis préoccupé(e)",
    "1.Je me sens tout mêlé(e)",
    "1. Je sens que j'ai les nerfs solides",
    "1.Je me sens bien",
]
columns_iastat = [
    "2.Je me sens bien",
    "2. Je me sens nerveux(se) et agité(e)",
    "2. Je me sens content(e) de moi-même",
    "2. Je voudrais être aussi heureux(se) que les autres semblent l'être",
    "2.J'ai l'impression d'être un(e) raté(e)",
    "2. Je me sens reposé(e)",
    "2. Je suis d'un grand calme",
    "2. Je sens que les difficultés s'accumulent au point où je n'arrive pas à les surmonter",
    "2. Je m'en fais trop pour des choses qui n'en valent pas vraiment la peine",
    "2. Je suis heureux(se)",
    "2. J'ai des pensées troublantes",
    "2. Je manque de confiance en moi",
    "2.Je me sens en sécurité",
    "2.Prendre des décisions m'est facile",
    "2. Je sens que je ne suis pas à la hauteur de la situation",
    "2. Je suis satisfait(e)",
    "2. Des idées sans importance me passent par la tête et me tracassent",
    "2.Je prends les désappointements tellement à coeur que je n'arrive pas à les chasser de mon esprit",
    "2.Je suis une personne qui a les nerfs solides",
    "2. Je deviens tendu(e) ou bouleversé(e) quand je songe à mes préoccupations et à mes intérêts récents.",
]

# Calculate IASTAE_total and IASTAT_total by summing their respective columns
iasta["IASTAE_total"] = iasta[columns_iastae].sum(axis=1)
iasta["IASTAT_total"] = iasta[columns_iastat].sum(axis=1)
all_part_data.index = all_part_data["participant"]
for p in iasta["Participant"].unique():
    if p in all_part_data["participant"]:
        if p == "sub-028":
            participants.loc[p, "stai_state_tot"] = np.nan
            all_part_data.loc[p, "stai_state_tot"] = np.nan

        else:
            participants.loc[p, "stai_state_tot"] = (
                iasta[iasta["Participant"] == p]["IASTAE_total"].values[0].astype(float)
            )
            all_part_data.loc[p, "stai_state_tot"] = (
                iasta[iasta["Participant"] == p]["IASTAE_total"].values[0].astype(float)
            )

        if p == "sub-028":
            participants.loc[p, "stai_trait_tot"] = np.nan
            all_part_data.loc[p, "stai_trait_tot"] = np.nan
        else:
            participants.loc[p, "stai_trait_tot"] = (
                iasta[iasta["Participant"] == p]["IASTAT_total"].values[0].astype(float)
            )
            all_part_data.loc[p, "stai_trait_tot"] = (
                iasta[iasta["Participant"] == p]["IASTAT_total"].values[0].astype(float)
            )

# PCS
pcs = pd.read_csv(opj(bids_root, "sourcedata", "pcs.csv"))

for p in pcs["Participant"].unique():
    if p in all_part_data["participant"]:
        if p == "sub-036":  # Missing data
            participants.loc[p, "pcs_tot"] = np.nan
            all_part_data.loc[p, "pcs_tot"] = np.nan
        else:
            participants.loc[p, "pcs_tot"] = (
                pcs[pcs["Participant"] == p]["PCS_total"].values[0].astype(float)
            )
            all_part_data.loc[p, "pcs_tot"] = (
                pcs[pcs["Participant"] == p]["PCS_total"].values[0].astype(float)
            )

bdi = pd.read_csv(opj(bids_root, "sourcedata", "bdi.csv"))
all_part_data["bdi_tot"] = np.nan
for p in bdi["Participant"].unique():
    if p in all_part_data["participant"]:
        if p == "sub-046":
            participants.loc[p, "bdi_tot"] = np.nan
            all_part_data.loc[p, "bdi_tot"] = np.nan

        else:
            participants.loc[p, "bdi_tot"] = (
                bdi[bdi["Participant"] == p]["BDI_total"].values[0].astype(float)
            )
            all_part_data.loc[p, "bdi_tot"] = (
                bdi[bdi["Participant"] == p]["BDI_total"].values[0].astype(float)
            )
all_part_data.to_csv(opj(bids_root, "derivatives", "all_part_data.csv"), index=False)

# Save participants
participants.to_csv(opj(bids_root, "participants.tsv"), sep="\t", index=False)
