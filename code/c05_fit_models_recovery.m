%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit learning models on pain uncertainty task data using the HGF toolbox
% @MP Coll, 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Turn off figures during execution
set(0,'DefaultFigureVisible','off')

% Basepath
basepath = fileparts(fileparts(mfilename('fullpath')));
outpath = fullfile(basepath, 'derivatives');

% Read data table created by python script
data = readtable(fullfile(outpath, 'all_part_data.csv'), "FileType","text", 'Delimiter', ',');


diary_name = fullfile(basepath, 'derivatives', 'c05_recovery_CommandWinOutput.txt');
diary(diary_name)

% Get participants
participants = unique(data.participant);

% Add models to path
addpath(genpath(fullfile(basepath, 'code', 'models'))) % Add custom models to matlab path
%
parfor  p = 1:length(participants)
    
    LMEs = [];
    models =  {};
    labels = {};
    
    % Make an output directory for this part
    part_path = fullfile(outpath, participants{p});
    mkdir(part_path)
    
    %% Get and format data
    
    % Get data from one participant
    dat = data(strcmp(participants{p}, data.participant), :);
    
    % Get variables of interest
    
    % Cues
    cues_string = dat.cue_;
    % Change for a number vector, easier to work with
    cues = repmat([999], size(cues_string, 1), 1); % Fill empty vector
    for s = 1:length(cues_string) % Replace "aigu" by 0 and "grave" by 1
        if strcmp(cues_string{s},  'aigu')
            cues(s) =  0;
        end
        if strcmp(cues_string{s},  'grave')
            cues(s) = 1;
        end
    end
    
    % Make sure all rows were filled
    assert(~any(cues > 1),  'Double-check cues assignement')
    
    % Outcome
    pain_string = dat.stimulus_;
    % Change for a number vector, easier to work with
    pain = repmat([999], size(pain_string, 1), 1); % Fill empty vector
    for s = 1:length(pain_string) % Replace "moderee" by 0 and "forte" by 1
        if strcmp(pain_string{s},  'moderee')
            pain(s) =  0;
        end
        if strcmp(pain_string{s},  'forte')
            pain(s) = 1;
        end
    end
    
    assert(~any(pain > 1),  'Double-check pain assignement')
    
    % Responses
    choice_key = dat.choice_pain_keys;
    choice = repmat([999], size(choice_key, 1), 1); % Fill empty vector
    for s = 1:length(choice_key) % Replace "m" by 1 and "n" by 0
        if strcmp(choice_key{s},  'n')
            choice(s) =  0;
        elseif strcmp(choice_key{s},  'm')
            choice(s) = 1;
        else  % Note MP: Ici il est possible qu'il n'y ait pas eu de réponse. On va utiliser le NaN dans ce cas
            choice(s) = NaN;
        end
        
    end
    assert(~any(choice > 1),  'Double-check choice assignement')
    
    
    % Response time
    rt = dat.choice_pain_rt;
    rating = dat.ratingScale_response;
    
    % Change cue-pain into contingency coding (see HGF manual, p.7)
    % Cue 0  = 'aigu', Cue 1 = 'grave',  Pain 0 = 'moderee', Pain 1 = 'forte'
    % Contingency coding
    % 'aigu' + 'moderee'  = 0 = 0, 0
    % 'grave' + 'forte'  = 0  = 1, 1
    % 'aigu' + 'forte'  = 1
    % 'grave' + 'moderee'  = 1
    
    contingencies = repmat([999], length(cues), 1); % Fill empty vector
    for s = 1:length(cues)
        if cues(s) == 0 && pain(s) == 0
            contingencies(s) = 0;
        end
        
        if cues(s) == 1 && pain(s) == 1
            contingencies(s) = 0;
        end
        
        if cues(s) == 1 && pain(s) == 0
            contingencies(s) =1;
        end
        
        if cues(s) == 0 && pain(s) == 1
            contingencies(s) =1;
        end
    end
    
    % Get response correct
    correct = repmat([999], length(choice), 1); % Fill empty vector
    for s = 1:length(choice)
        if choice(s) == pain(s)
            correct(s) = 1;
        else
            correct(s) = 0;
        end
    end
    
    
    %% Fit models
    % Optimizer
    opt_algo =  tapas_quasinewton_optim_config();
    opt_algo.maxIter = 1000; % Increase maxiter to favor convergence.
    opt_algo.maxRst = 1000;
    
    % Create an input matrix with contigencies and pain_intensity
    % Add ones at the end HGF uses last column as time index.
    u = [contingencies, pain, ones(length(contingencies), 1)];
    

    % Load the simulated data, fit all models on it, save LMEs
    sim_mods = {'rw',  'ph', 'HGF3pu', 'HGF3', 'HGF2pu', 'HGF2'};
    for sim_mod = 1:length(sim_mods)
        for sim = 1:10
            % Load simulated data
            sim_data = readmatrix(fullfile(part_path,  [participants{p}, '_', sim_mods{sim_mod}, '_simulated_resp.csv']));
            % Use logRT for model fitting
            logrt = log(sim_data(:, sim));
            
            %% HGF2
            hgf_obs_model = tapas_logrt_linear_binary_hgf2_config();
            
            % Get default priors
            c = tapas_ehgf_binary_config();
            
            % Remove third level for 2 levels HGF
            c.logkamu = [log(1), -Inf];
            c.logkasa = [     0,      0];
            
            % Set new mean priors and fix third level
            c.ommu(2) = -5;
            c.omsa = [NaN,   16,   0];
            c.priormus = [c.mu_0mu, c.logsa_0mu, c.rhomu, c.logkamu, c.ommu];
            c.priorsas = [c.mu_0sa, c.logsa_0sa, c.rhosa, c.logkasa, c.omsa];
            
            % Fit model with perceptual
            est  = tapas_fitModel(logrt, ...  % Responses are log rt (missed trials == NaN)
                u, ...  % Inputs are contingencies
                c, ...  % Perceptual model
                hgf_obs_model, ... % Response model
                opt_algo);
            est.prc_model = 'hgf2';
            
            % Save values of this model
            LMEs(sim,  1) = est.optim.LME;
            labels{1} = 'hgf2';
            
            
            %% HGF2PU
            hgf_obs_model = tapas_logrt_linear_binary_hgf2_config();
            
            % Get default priors
            c = tapas_ehgf_binary_pu_config();
            
            % Remove third level
            % Remove third level for 2 levels HGF
            c.logkamu = [log(1), -Inf];
            c.logkasa = [     0,      0];
            
            % Set new mean priors and fix third level
            c.ommu(2) = -5;
            c.omsa = [NaN,   16,   0];
                    
            c.priormus = [c.mu_0mu, c.logsa_0mu, c.rhomu, c.logkamu, c.ommu,  c.logalmu  c.eta0mu, c.eta1mu];
            c.priorsas = [c.mu_0sa, c.logsa_0sa, c.rhosa, c.logkasa, c.omsa, c.logalsa  c.eta0sa, c.eta1sa];
            
            % Fit model with perceptual
            est  = tapas_fitModel(logrt, ...  % Responses are log rt (missed trials == NaN)
                u, ...  % Inputs are contingencies
                c, ...  % Perceptual model
                hgf_obs_model, ... % Response model
                opt_algo);
            est.prc_model = 'hgf2pu';
            
            % Save values of this model
            LMEs(sim, 2) = est.optim.LME;
            labels{2} = 'hgf2pu';
            
        
            %% HGF3
            hgf_obs_model = tapas_logrt_linear_binary_uncertainty_config();
            
            % Get default priors
            c = tapas_hgf_binary_config();
            
            c.ommu = [NaN, -5, -6];
            c.omsa = [NaN, 16, 16];
            c.priormus = [c.mu_0mu, c.logsa_0mu, c.rhomu, c.logkamu, c.ommu];
            c.priorsas = [c.mu_0sa, c.logsa_0sa, c.rhosa, c.logkasa, c.omsa];
            
            % Fit model with perceptual
            est  = tapas_fitModel(logrt, ...  % Responses are log rt (missed trials == NaN)
                u, ...  % Inputs are contingencies
                c, ...  % Perceptual model
                hgf_obs_model, ... % Response model
                opt_algo);
            est.prc_model = 'hgf3';
            
            % Save values of this model
            LMEs(sim, 3) = est.optim.LME;
            models{end+1} = est;
            labels{3} = 'hgf3';
            
            %% HGF3 pu
            hgf_obs_model = tapas_logrt_linear_binary_uncertainty_config();
            
            % Get default priors
            c = tapas_ehgf_binary_pu_config();
            
            c.ommu = [NaN, -5, -6];
            c.omsa = [NaN, 16, 16];
            c.priormus = [c.mu_0mu, c.logsa_0mu, c.rhomu, c.logkamu, c.ommu,  c.logalmu  c.eta0mu, c.eta1mu];
            c.priorsas = [c.mu_0sa, c.logsa_0sa, c.rhosa, c.logkasa, c.omsa, c.logalsa  c.eta0sa, c.eta1sa];
            
            % Fit model with perceptual
            est  = tapas_fitModel(logrt, ...  % Responses are log rt (missed trials == NaN)
                u, ...  % Inputs are contingencies
                c, ...  % Perceptual model
                hgf_obs_model, ... % Response model
                opt_algo);
            est.prc_model = 'hgf3pu';
            
            % Save values of this model
            LMEs(sim, 4) = est.optim.LME;
            models{end+1} = est;
            labels{4} = 'hgf3pu';
            

            %% Rescolar-Wagner
            rw_config = tapas_rw_binary_config();
            rw_obs = tapas_logrt_linear_binary_rw_config();
            rw_config.logitv_0sa = 1;
            rw_config.priorsas = [rw_config.logitv_0sa,rw_config.logitalsa];
        
            % Fit model with perceptual
            est  = tapas_fitModel(logrt, ...  % Responses are log rt (missed trials == NaN)
                u, ...  % Inputs are contingencies
                rw_config, ...  % Perceptual model
                rw_obs, ... % Response model
                opt_algo);
            est.prc_model = 'tapas_rw_binary_config';
            
            models{end+1} = est;
            LMEs(sim, 5) = est.optim.LME;
            labels{5} = 'rw';
            
            
            %% Pearce-Hall
            ph_config = tapas_ph_binary_config();
            ph_obs = tapas_logrt_linear_binary_ph_config();
            ph_config.logitv_0sa = 1;

            ph_config.priorsas = [
                ph_config.logitv_0sa,...
                ph_config.logital_0sa,...
                ph_config.logSsa,...
                     ];
            % Fit model with perceptual
            est  = tapas_fitModel(logrt, ...  % Responses are log rt (missed trials == NaN)
                u, ...  % Inputs are contingencies
                ph_config, ...  % Perceptual model
                ph_obs, ... % Response model
                opt_algo);
            
            LMEs (sim, 6) = est.optim.LME;
            est.prc_model = 'ph';
            models{end+1} = est;
            labels{6} = 'ph';
            
            % Save LMEs
            if ~isreal(LMEs) % Make sure all ok
                stophere
            end
            
            
            close all % Close all plots
        end
        lme_part = LMEs;
        model_table = labels';
        tab = [table(model_table), array2table(lme_part')];
        writetable(tab, fullfile(part_path, ['LMEs_sims_'  sim_mods{sim_mod}   ' .csv']))
        
    end
    
    
end



% Load participant file
partfile = readtable(fullfile(basepath, 'participants.tsv'), "FileType","text", 'Delimiter', '\t');

% Remove exclusions
partfile = partfile(partfile.excluded == 0, :);

% Get participants to include
participants = unique(partfile.participant_id);
sim_mods = {'rw', 'ph', 'HGF3pu', 'HGF3', 'HGF2pu', 'HGF2'};
for sim_model = 1:6
    c = 0;
    for  p = 1:length(participants)
        % Make an output directory
        part_path = fullfile(outpath, participants{p});
        lmepart = readtable(fullfile(part_path, ['LMEs_sims_' sim_mods{sim_model} ' .csv']));
        
        % First part create matrix
        if p == 1
            LMEs =  zeros( height(lmepart), length(participants)*10);
            labels = lmepart.model_table;
        end
        for sim = 1:10
            c = c +1;
            LMEs(:, c) = table2array(lmepart(:, sim+1));
        end
    end
    
    % % Bayesian model comparison (need vba toolbox in path
    % % https://github.com/MBB-team/VBA-toolbox)
    options.modelNames = labels';
    [posterior,out] = VBA_groupBMC(LMEs, options) ;
    out.pep =(1-out.bor)*out.ep + out.bor/length(out.ep); % From https://mbb-team.github.io/VBA-toolbox/wiki/BMS-for-group-studies/#rfx-bms
    save(fullfile(outpath, ['VBA_BMC_' sim_mods{sim_model} '_sim_recovery.mat']), 'out')
    close all
    
end

diary off