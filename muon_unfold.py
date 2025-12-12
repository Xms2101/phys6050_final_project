#!/usr/bin/env python3
"""
Build response & efficiencies (detector-only),
run iterative_unfold, and plot:
- Efficiency vs truth energy
- Response matrix (row-normalized heatmap)
- Observed data vs MC vs unfolded
- Unfolded signal-only (apply physics selection at truth level)
"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import uproot
from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger

# -------------------------
# USER PARAMETERS
# -------------------------
MEV_TO_GEV = 1000.0

# Key parameters
bins = np.linspace(2, 9, 7) #Make sure these divide well!! Do the math!!
bin_centers = 0.5 * (bins[:-1] + bins[1:])
iterations = 13 #max number of iterations
ts = 1e-3 #chi^2 cut. Best at 1e-2


# Files (adjust paths if necessary)
d_files = 24
mc_file_count = 7
mc_files   = [f"data/monte_carlo/1p_mc_snip_{i}.root" for i in range(1, mc_file_count + 1)]
data_files = [f"data/reconstructed/1p_data_snip_{i}.root" for i in range(1, d_files + 1)]

mc_paths = {file: "MasterAnaDev" for file in mc_files}
data_paths = {file: "MasterAnaDev" for file in data_files}

# -------------------------
# READ BRANCHES
# -------------------------
mc_arr = uproot.concatenate(
    mc_paths,
    filter_name=["gamma1_E", "gamma2_E", "mc_vtx", "MasterAnaDev_vtx",
                 "MasterAnaDev_muon_E","MasterAnaDev_muon_P","MasterAnaDev_muon_Px","MasterAnaDev_muon_Py","MasterAnaDev_muon_Pz","MasterAnaDev_muon_theta",
                 "MasterAnaDev_pion_E","MasterAnaDev_pion_P","MasterAnaDev_pion_Px","MasterAnaDev_pion_Py","MasterAnaDev_pion_Pz","MasterAnaDev_pion_theta",
                 "MasterAnaDev_minos_trk_qp", "MasterAnaDev_proton_score",
                 "mc_incomingE", "pi0_E", "truth_muon_E"],
    library="np"
)

data_arr = uproot.concatenate(
    data_paths,
    filter_name=["gamma1_E", "gamma2_E", "MasterAnaDev_vtx",
                 "MasterAnaDev_muon_E","MasterAnaDev_muon_P","MasterAnaDev_muon_Px","MasterAnaDev_muon_Py","MasterAnaDev_muon_Pz","MasterAnaDev_muon_theta",
                 "MasterAnaDev_pion_E","MasterAnaDev_pion_P","MasterAnaDev_pion_Px","MasterAnaDev_pion_Py","MasterAnaDev_pion_Pz","MasterAnaDev_pion_theta",
                 "MasterAnaDev_minos_trk_qp", "MasterAnaDev_proton_score", "MasterAnaDev_pion_score", "MasterAnaDev_pion_score1", "MasterAnaDev_pion_score2",
                 "muon_track_cluster_ID_sz"],
    library="np"
)

# -------------------------
# BUILD RECO ARRAYS (DATA)
# -------------------------
mu_E_reco_MeV   = np.where(data_arr["MasterAnaDev_muon_E"] < 0, 0, data_arr["MasterAnaDev_muon_E"])
mu_E_reco       = mu_E_reco_MeV / MEV_TO_GEV
mu_P_reco_MeV   = data_arr["MasterAnaDev_muon_P"]
mu_P_reco       = mu_P_reco_MeV / MEV_TO_GEV
mu_P_x_reco     = data_arr["MasterAnaDev_muon_Px"] / MEV_TO_GEV
mu_P_y_reco     = data_arr["MasterAnaDev_muon_Py"] / MEV_TO_GEV
mu_P_z_reco     = data_arr["MasterAnaDev_muon_Pz"] / MEV_TO_GEV

pi_E_candidates_MeV   = np.where(data_arr["MasterAnaDev_pion_E"] < 0, 0, data_arr["MasterAnaDev_pion_E"])
pi_E_candidates       = pi_E_candidates_MeV / MEV_TO_GEV
pi_P_candidates_MeV   = data_arr["MasterAnaDev_pion_P"]
pi_P_candidates       = pi_P_candidates_MeV / MEV_TO_GEV
pi_P_x_candidates     = data_arr["MasterAnaDev_pion_Px"] / MEV_TO_GEV
pi_P_y_candidates     = data_arr["MasterAnaDev_pion_Py"] / MEV_TO_GEV
pi_P_z_candidates     = data_arr["MasterAnaDev_pion_Pz"] / MEV_TO_GEV
pi_theta_candidates   = data_arr["MasterAnaDev_pion_theta"]

best_pion_index = np.argmax(pi_E_candidates, axis=1)
rows = np.arange(len(best_pion_index))

# Selected best pion in data
pi_E_r   = pi_E_candidates[rows, best_pion_index]
pi_P_r   = pi_P_candidates[rows, best_pion_index]
pi_P_x_r = pi_P_x_candidates[rows, best_pion_index]
pi_P_y_r = pi_P_y_candidates[rows, best_pion_index]
pi_P_z_r = pi_P_z_candidates[rows, best_pion_index]
pi_theta_r = pi_theta_candidates[rows, best_pion_index]
cos_pi_theta_r = np.cos(pi_theta_r)

q_over_p_reco = data_arr["MasterAnaDev_minos_trk_qp"]
p_score_reco = data_arr["MasterAnaDev_proton_score"]

vtx_x_reco = data_arr["MasterAnaDev_vtx"][..., 0]
vtx_y_reco = data_arr["MasterAnaDev_vtx"][..., 1]
vtx_z_reco = data_arr["MasterAnaDev_vtx"][..., 2]

# neutrino energy estimate for reco
nu_E_reco = mu_E_reco + pi_E_r

# -------------------------
# BUILD MC ARRAYS (truth & reco)
# -------------------------
mu_E_mc_truth_MeV       = mc_arr["truth_muon_E"]
mu_E_mc_truth           = mu_E_mc_truth_MeV / MEV_TO_GEV
mu_E_mc_reco_precut_MeV = np.where(mc_arr["MasterAnaDev_muon_E"] < 0, 0, mc_arr["MasterAnaDev_muon_E"])
mu_E_mc_reco_precut     = mu_E_mc_reco_precut_MeV / MEV_TO_GEV

mu_P_mc_precut_MeV      = mc_arr["MasterAnaDev_muon_P"]
mu_P_mc_precut          = mu_P_mc_precut_MeV / MEV_TO_GEV
mu_P_x_mc               = mc_arr["MasterAnaDev_muon_Px"] / MEV_TO_GEV
mu_P_y_mc               = mc_arr["MasterAnaDev_muon_Py"] / MEV_TO_GEV
mu_P_z_mc               = mc_arr["MasterAnaDev_muon_Pz"] / MEV_TO_GEV

pi_E_mc_reco_all_MeV     = np.where(mc_arr["MasterAnaDev_pion_E"] < 0, 0, mc_arr["MasterAnaDev_pion_E"])
pi_E_mc_reco_all         = pi_E_mc_reco_all_MeV / MEV_TO_GEV
pi_P_mc_all_MeV          = mc_arr["MasterAnaDev_pion_P"]
pi_P_mc_all              = pi_P_mc_all_MeV / MEV_TO_GEV
pi_P_x_mc_candidates     = mc_arr["MasterAnaDev_pion_Px"] / MEV_TO_GEV
pi_P_y_mc_candidates     = mc_arr["MasterAnaDev_pion_Py"] / MEV_TO_GEV
pi_P_z_mc_candidates     = mc_arr["MasterAnaDev_pion_Pz"] / MEV_TO_GEV
pi_theta_mc_all          = mc_arr["MasterAnaDev_pion_theta"]

rows_mc = np.arange(len(pi_E_mc_reco_all))
best_idx_t = np.argmax(pi_E_mc_reco_all, axis=1)
pi_E_mc_reco  = pi_E_mc_reco_all[rows_mc, best_idx_t]
pi_P_mc  = pi_P_mc_all[rows_mc, best_idx_t]
pi_P_x_mc  = pi_P_x_mc_candidates[rows_mc, best_idx_t]
pi_P_y_mc  = pi_P_y_mc_candidates[rows_mc, best_idx_t]
pi_P_z_mc  = pi_P_z_mc_candidates[rows_mc, best_idx_t]
pi_theta_mc = pi_theta_mc_all[rows_mc, best_idx_t]
cos_pi_theta_mc = np.cos(pi_theta_mc)

nu_E_mc = mu_E_mc_reco_precut + pi_E_mc_reco

q_over_p_mc = mc_arr["MasterAnaDev_minos_trk_qp"]
p_score_mc = mc_arr["MasterAnaDev_proton_score"]

vtx_x_mc_truth = mc_arr["mc_vtx"][:, 0]
vtx_y_mc_truth = mc_arr["mc_vtx"][:, 1]
vtx_z_mc_truth = mc_arr["mc_vtx"][:, 2]

# -------------------------
# DETECTOR-ONLY MASKS (for response & efficiencies)
# -------------------------
# Fiducial
fid_reco = (
    (vtx_x_reco > -170) & (vtx_x_reco < 170) &
    (vtx_y_reco > -170) & (vtx_y_reco < 170) &
    (vtx_z_reco > 5980) & (vtx_z_reco < 8422)
)
fid_mc = (
    (vtx_x_mc_truth > -170) & (vtx_x_mc_truth < 170) &
    (vtx_y_mc_truth > -170) & (vtx_y_mc_truth < 170) &
    (vtx_z_mc_truth > 5980) & (vtx_z_mc_truth < 8422)
)

# Basic sanity
reco_sanity = np.isfinite(mu_E_reco) & np.isfinite(pi_E_r) & (mu_E_reco > 0)
mc_sanity   = np.isfinite(mu_E_mc_truth) & (mu_E_mc_truth > 0)

# Minimal acceptance
min_mu_p = 1.4 #E_mu > 1.5GeV, |p| = sqrt(E_mu^2 - m_mu^2) = sqrt(1.5^2 - 0.105^2) ~ 1.496
mu_accept = (mu_P_reco > min_mu_p)
mu_accept_mc = (mu_P_mc_precut > min_mu_p)

reco_detector_mask = fid_reco & reco_sanity & mu_accept
mc_detector_mask   = fid_mc & mc_sanity & mu_accept_mc

# Arrays used for response building: pairs from MC where reco passes detector mask
TRUTH_all = mu_E_mc_truth
RECO_all  = mu_E_mc_reco_precut

truth_for_response = TRUTH_all[mc_detector_mask]
reco_for_response  = RECO_all[mc_detector_mask]

# Denominator for efficiency: generated truth in unfolding domain (optionally restrict to fiducial gen)
denom_for_eff = TRUTH_all[fid_mc]  # truth generated inside fiducial volume

# Restrict to unfolding domain
low_edge, high_edge = bins[0], bins[-1]
in_range_mask = (truth_for_response >= low_edge) & (truth_for_response <= high_edge)
truth_for_response = truth_for_response[in_range_mask]
reco_for_response  = reco_for_response[in_range_mask]

denom_in_range_mask = (denom_for_eff >= low_edge) & (denom_for_eff <= high_edge)
denom_for_eff = denom_for_eff[denom_in_range_mask]

# -------------------------
# BUILD HISTOGRAMS & RESPONSE
# -------------------------
data_hist, _ = np.histogram(mu_E_reco, bins=bins)   # observed data (detector-level; no physics cuts here)
data_err = np.sqrt(data_hist)

truth_hist, _ = np.histogram(denom_for_eff, bins=bins)
resp_counts, xedges, yedges = np.histogram2d(truth_for_response, reco_for_response, bins=[bins, bins])

# Normalize rows to build P(reco|truth)
resp_matrix = np.zeros_like(resp_counts, dtype=float)
response_err = np.zeros_like(resp_counts, dtype=float)
for i in range(resp_counts.shape[0]):
    row_sum = np.sum(resp_counts[i])
    if row_sum > 0:
        resp_matrix[i] = resp_counts[i] / row_sum
        response_err[i] = np.sqrt(resp_counts[i]) / row_sum
    else:
        resp_matrix[i] = np.zeros(resp_counts.shape[1])
        response_err[i] = np.zeros(resp_counts.shape[1])

#resp_matrix = resp_matrix.T
#response_err = response_err.T

# Efficiencies
truth_passed_hist, _ = np.histogram(truth_for_response, bins=bins)
truth_all_hist, _    = np.histogram(denom_for_eff, bins=bins)

efficiencies = np.zeros_like(truth_all_hist, dtype=float)
efficiencies_err = np.zeros_like(truth_all_hist, dtype=float)
for i in range(len(truth_all_hist)):
    if truth_all_hist[i] > 0:
        efficiencies[i] = truth_passed_hist[i] / truth_all_hist[i]
        efficiencies_err[i] = np.sqrt(efficiencies[i] * (1 - efficiencies[i]) / truth_all_hist[i])
    else:
        efficiencies[i] = 0.0
        efficiencies_err[i] = 0.0

print("MC generated in-range:", truth_all_hist.sum())
print("MC passed detector reco in-range:", truth_passed_hist.sum())
print("efficiencies:", efficiencies)

# -------------------------
# TRUTH-LEVEL PHYSICS SELECTION (to apply AFTER unfolding)
# Compute fraction of truth that is signal (physics selection)
# -------------------------
# Define the physics-level cuts on MC truth (these are the "signal" selection)
cut_mu_P_mc       = (mu_P_mc_precut > 1.5)
cut_mu_charge_mc  = (q_over_p_mc < 0)
cut_pi_P_mc       = (pi_P_mc > 0.1)
cut_p_score_mc    = (p_score_mc < 0.4)
cut_nu_E_mc       = (nu_E_mc > 2.0) & (nu_E_mc < 20.0)
cut_pi_angle_mc   = (cos_pi_theta_mc > 0.75)

t_mc = np.abs((nu_E_mc - mu_E_mc_reco_precut - pi_E_mc_reco)**2 -
              ((mu_P_x_mc + pi_P_x_mc)**2 +
               (mu_P_y_mc + pi_P_y_mc)**2 +
               (nu_E_mc - mu_P_z_mc - pi_P_z_mc)**2))
cut_t_mc = (t_mc < 0.125)

physics_signal_mask_truth = fid_mc & mc_sanity & cut_mu_P_mc & cut_mu_charge_mc & cut_pi_P_mc & cut_p_score_mc & cut_nu_E_mc & cut_pi_angle_mc & cut_t_mc

# signal truth distribution (in same bins, and restrict to same unfolding domain)
signal_truth = mu_E_mc_truth[physics_signal_mask_truth]
signal_truth_inrange = signal_truth[(signal_truth >= low_edge) & (signal_truth <= high_edge)]
signal_truth_hist, _ = np.histogram(signal_truth_inrange, bins=bins)

# fraction of truth per bin that is signal
signal_fraction = np.zeros_like(truth_all_hist, dtype=float)
with np.errstate(divide='ignore', invalid='ignore'):
    nonzero = truth_all_hist > 0
    signal_fraction[nonzero] = signal_truth_hist[nonzero] / truth_all_hist[nonzero]
    signal_fraction[~nonzero] = 0.0

# -------------------------
# RUN PYUNFOLD
# -------------------------
unfolded_results = iterative_unfold(
    data             = data_hist,
    data_err         = data_err,
    response         = resp_matrix,
    response_err     = response_err,
    efficiencies     = efficiencies,
    efficiencies_err = efficiencies_err,
    callbacks        = [Logger()],
    max_iter         = iterations,
    ts_stopping      = ts
)

unf = unfolded_results['unfolded'].astype(float)
stat_err = np.array(unfolded_results.get('stat_err', np.zeros_like(unf)), dtype=float)
sys_err  = np.array(unfolded_results.get('sys_err', np.zeros_like(unf)), dtype=float)

# apply signal fraction to the unfolded result to get signal-only truth
unf_signal = unf * signal_fraction
unf_signal_err = np.sqrt((stat_err * signal_fraction)**2 + (unf * np.sqrt(signal_fraction * (1 - signal_fraction) / np.maximum(truth_all_hist, 1)))**2)

# For plotting compare to MC truth (truth_all_hist) and MC reco (projected reco)
mc_reco_hist, _ = np.histogram(reco_for_response, bins=bins)
mc_truth_hist = truth_all_hist.copy()

# scale MC truth/reco to data integral for plotting
data_sum = data_hist.sum()
mc_truth_sum = mc_truth_hist.sum() if mc_truth_hist.sum() > 0 else 1.0
mc_scale = data_sum / mc_truth_sum
mc_reco_scaled = mc_reco_hist * mc_scale
mc_truth_scaled = mc_truth_hist * mc_scale

# scale unfolded to match data integral for plotting (optional)
unf_sum = np.sum(unf)
scale_to_data = data_sum / unf_sum if unf_sum > 0 else 1.0
unf_scaled = unf * scale_to_data
unf_err_scaled = np.sqrt(stat_err**2 + sys_err**2) * scale_to_data
unf_signal_scaled = unf_signal * scale_to_data
unf_signal_err_scaled = unf_signal_err * scale_to_data

# -------------------------
# PLOTS
# -------------------------
#plt.style.use('seaborn-deep')

# 1) Efficiency vs truth energy
fig, ax = plt.subplots(figsize=(8,4))
ax.errorbar(bin_centers, efficiencies, yerr=efficiencies_err, fmt='o-', capsize=3)
ax.set_xlabel("Truth muon energy [GeV]")
ax.set_ylabel("Efficiency (reco | truth)")
ax.set_title("Efficiency vs truth energy")
ax.set_ylim(bottom=0)
plt.tight_layout()

# 2) Response matrix heatmap (row-normalized)
fig, ax = plt.subplots(figsize=(7,6))
# show resp_matrix as is (rows are truth bins, columns reco bins)
im = ax.imshow(resp_matrix, origin='lower', aspect='auto',
               extent=[bins[0], bins[-1], bins[0], bins[-1]])
ax.set_xlabel("Reconstructed muon energy [GeV]")
ax.set_ylabel("Truth muon energy [GeV]")

ax.set_title("Response matrix P(reco | truth) (row-normalized)")
plt.colorbar(im, ax=ax, label='P(reco|truth)')
plt.tight_layout()
#plt.savefig("plots/E_mu_matrix.png")
#plt.savefig("plots/E_mu_matric_i1")
#plt.savefig("plots/E_mu_matric_i5")
plt.savefig(f"plots/E_mu_matrix")

# 3) Observed data vs MC reco vs unfolded
fig, ax = plt.subplots(figsize=(9,6))
ax.errorbar(bin_centers, data_hist, yerr=np.sqrt(data_hist), fmt='s', label='Observed data', ms=6, capsize=3, color='k')
ax.step(bin_centers, mc_reco_scaled, where='mid', lw=2, label='MC Reco', color='C0')
ax.step(bin_centers, mc_truth_scaled, where='mid', lw=2, alpha=0.6, label='MC Truth', color='C1')
ax.errorbar(bin_centers, unf_scaled, yerr=unf_err_scaled, fmt='o', label='Unfolded', ms=6, capsize=3, color='C3')
ax.set_xlabel("Muon energy [GeV]")
ax.set_ylabel("Counts")
iteration_text = f'Iterations: {iterations}'
plt.text(0.92, 0.92, # Coordinates (near bottom-right)
         iteration_text, 
         ha='right', # Horizontal alignment
         va='bottom', # Vertical alignment
         transform=plt.gca().transAxes, # Use axes coordinates
         fontsize=15, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax.legend()
plt.tight_layout()
#plt.savefig("plots/E_mu_full.png")
#plt.savefig("plots/E_mu_full_i1.png")
plt.savefig(f"plots/E_mu_full_i{iterations}")


# 4) Unfolded signal-only (apply truth-level physics selection fraction)
# 4) Unfolded signal vs MC truth (apply truth-level physics selection fraction)
fig, ax = plt.subplots(figsize=(9,6))

# Unfolded reco signal (with errors)
ax.errorbar(
    bin_centers, 
    unf_signal_scaled, 
    yerr=unf_signal_err_scaled, 
    fmt='o', 
    label='Unfolded reco signal', 
    ms=6, capsize=3, color='C2'
)

# MC truth signal (scaled)
mc_signal_truth_scaled = signal_truth_hist * mc_scale
ax.step(
    bin_centers, 
    mc_signal_truth_scaled, 
    where='mid', 
    lw=2, 
    label='MC truth (signal) scaled', 
    color='C4'
)

ax.set_xlabel("Muon energy [GeV]")
ax.set_ylabel("Counts (signal-only)")
ax.legend()
plt.tight_layout()
plt.show()

print(efficiencies)
print(efficiencies_err)