
# Matlab path
alias matlab="/Applications/MATLAB_R2024a.app/bin/matlab"  

# # Import and plot participants' data
python code/c01_import_and_plot_participant.py || exit

# # Fit models
matlab -nodisplay -nosplash -nodesktop -r "run('code/c02_fit_models_painuncertainty.m'); exit;" || exit

# Compare models
matlab -nodisplay -nosplash -nodesktop -r "run('code/c03_compare_models_painuncertainty.m'); exit;" || exit

# Stats and plots
python code/c04_group_plots_stats.py || exit;

# # Fit models on simulation
matlab -nodisplay -nosplash -nodesktop -r "run('code/c05_fit_models_recovery.m'); exit;" || exit

# Parameters recovery
python code/c06_parameters_recovery.py || exit

# Final figures
python code/c07_figures_publication.py || exit
