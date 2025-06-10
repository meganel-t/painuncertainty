%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit learning models on pain uncertainty task data using the HGF toolbox
% @MP Coll, 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Turn off figures during execution
set(0,'DefaultFigureVisible','on')

% Base path
basepath = fileparts(fileparts(mfilename('fullpath')));
outpath = fullfile(basepath, 'derivatives');
% Load participant file
partfile = readtable(fullfile(basepath, 'participants.tsv'), "FileType","text", 'Delimiter', '\t');

% Remove exclusions
partfile = partfile(partfile.excluded == 0, :);

% Get participants to include
participants = unique(partfile.participant_id);

for  p = 1:length(participants)
    % Make an output directory
    part_path = fullfile(outpath, participants{p});
    lmepart = readtable(fullfile(part_path, 'LMEs.csv'));
    display(part_path)
    display(lmepart)
    
    % First part create matrix
    if p == 1
        LMEs =  zeros( height(lmepart), length(participants));
        labels = lmepart.model_table;
    end
    LMEs(:, p) = lmepart.Var2;
end

% % Bayesian model comparison (need vba toolbox in path
% % https://github.com/MBB-team/VBA-toolbox)
options.modelNames = labels';
[posterior,out] = VBA_groupBMC(LMEs, options) ;
out.pep =(1-out.bor)*out.ep + out.bor/length(out.ep); % From https://mbb-team.github.io/VBA-toolbox/wiki/BMS-for-group-studies/#rfx-bms
save(fullfile(outpath, 'VBA_BMC.mat'), 'out')

% Save LMEs to file
LMEs = num2cell(LMEs');
LMEs = [participants, LMEs];
T = array2table(LMEs, 'VariableNames', [{'participant_id'}, labels']);
writetable(T, fullfile(outpath, 'LMEs.csv'))

close all

