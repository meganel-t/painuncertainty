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

% Save command window output to spot issues and warning messages
diary_name = fullfile(basepath, 'derivatives', 'c02_modelfit_CommandWinOutput.txt');
diary(diary_name)

% Get participants
participants = unique(data.participant);

% Add models to path
addpath(genpath(fullfile(basepath, 'code', 'models'))) % Add custom models to matlab path

parfor  p = 1:length(participants)
    display(p);
    LMEs = [];
    models =  {};
    labels = {};
    
    % Make an output directory for this part
    part_path = fullfile(outpath, participants{p});

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
    % 'aigu' + 'forte'  = 1 = 0, 1
    % 'grave' + 'moderee'  =  1=  1, 0
    
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
    opt_algo.maxIter = 2000; % Increase maxiter to favor convergence.
    opt_algo.maxRst = 2000;
    opt_algo.seedRandInit = 1; % set seed
    
    
    % Ignore incorrect response in response model
    rt(correct == 0) = NaN;
    rm_correct = sum(isnan(rt));

    % Ignore response time below 200ms
    rm_below = sum(rt < 0.2);
    rt(rt < 0.2) = NaN;
    rm_add_below = sum(isnan(rt)) - rm_correct;

    part = {participants{p}};
    rm_prop =  sum(isnan(rt))/192;
    remaining = 192 -  sum(isnan(rt));
    rm_table = table(part, rm_correct, rm_below, rm_add_below, rm_prop, remaining);
    writetable(rm_table, fullfile(part_path, [participants{p} '_ignored_trials.csv']))

    % Log transform and round to 16 digit for numerical stability (prevents some matrix inversion problems)
    logrt = round(log(rt*1000), 16);
    
    % Create an input matrix with contigencies and pain_intensity
    % Add ones at the end HGF uses last column as time index.
    u = [contingencies, pain, ones(length(contingencies), 1)];
    

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
    LMEs(end+1) = est.optim.LME;
    models{end+1} = est;
    labels{end+1} = 'hgf2';
    
    % Save trajectories for this model
    est.traj.ratings = rating;
    est.traj.pain = pain;
    est.traj.rt = rt;
    est.traj.choice = choice;
    est.traj.correct = correct;
    writetable(struct2table(est.traj), fullfile(part_path, [participants{p} '_HGF2_uncertainty_traj.csv']))
    
    % Save parameters for this model
    obs_table = struct2table(est.p_obs);
    obs_table.p = [];
    obs_table.ptrans =[];
    writetable(obs_table, fullfile(part_path, [participants{p} '_HGF2_uncertainty_obsparams.csv']))
    
    prc_table = table();
    prc_table.om2 = est.p_prc.om(2);
    prc_table.om3 = est.p_prc.om(3);
    writetable(prc_table, fullfile(part_path, [participants{p} '_HGF2_uncertainty_prcparams.csv']))
    
    
    %Make plots
    % Rescale Y between 0 and 1 to fit in plot
    est.y_original = est.y; % Keep original
    est.y = rescale(est.y);
    tapas_hgf_binary_plotTraj_nosd(est)
    sgtitle([participants{p} '- HGF2 without perceptual uncertainty'], 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'HGF2_traj.png'))
    tapas_fit_plotCorr(est)
    sgtitle([participants{p} '- HGF2  without perceptual uncertainty '], 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'HGF2_corr.png'))
    close all
    

    % Simulate and recover for this (winning) model
    rec_params = [];
    simulated_y = [];
    for i = 1:10
        sim = tapas_simModel(u, ...
            'tapas_ehgf_binary', ...
            est.p_prc.p,...
            'tapas_logrt_linear_binary_hgf2', ...
            est.p_obs.p, ...
            i); % different seed
        
        est_recover  = tapas_fitModel(sim.y, ...  
            u, ...  % Inputs are contingencies
            c, ...  % Perceptual model
            hgf_obs_model, ... % Response model
            opt_algo);
        rec_params = [rec_params; [est_recover.p_prc.om(2), est_recover.p_obs.p ]];
        simulated_y = [simulated_y , exp(sim.y)];
    end
    
    % Write recovered parameters and simulated data
    tab = array2table(rec_params, 'VariableNames', {'om2', 'b0', 'b1', 'b2', 'ze'});
    writetable(tab, fullfile(part_path, [participants{p} '_HGF2_recovered_params.csv']))
    writematrix(simulated_y, fullfile(part_path, [participants{p} '_HGF2_simulated_resp.csv']))
    
    
    %% HGF2PU
    hgf_obs_model = tapas_logrt_linear_binary_hgf2_config();

    % Get default priors
    c = tapas_ehgf_binary_pu_config();
    
    % Remove third level
    c.logkamu = [log(1), -Inf];
    c.logkasa = [     0,      0];

    % Set priors
    c.ommu(2) = -5
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
    LMEs(end+1) = est.optim.LME;
    models{end+1} = est;
    labels{end+1} = 'hgf2pu';
    
    % Save trajectories for this model

    est.traj.ratings = rating;
    est.traj.pain = pain;
    est.traj.rt = rt;
    est.traj.choice = choice;
    est.traj.correct = correct;
    writetable(struct2table(est.traj), fullfile(part_path, [participants{p} '_HGF2pu_uncertainty_traj.csv']))
    
    % Save parameters for this model
    obs_table = struct2table(est.p_obs);
    obs_table.p = [];
    obs_table.ptrans =[];
    writetable(obs_table, fullfile(part_path, [participants{p} '_HGF2pu_uncertainty_obsparams.csv']))

    prc_table = table();
    prc_table.om2 = est.p_prc.om(2);
    prc_table.om3 = est.p_prc.om(3);
    writetable(prc_table, fullfile(part_path, [participants{p} '_HGF2pu_uncertainty_prcparams.csv']))
    
    %Make plots
    % Rescale Y between 0 and 1 to fit in plot
    est.y_original = est.y; % Keep original
    est.y = rescale(est.y);
    
    tapas_hgf_binary_plotTraj_nosd(est)
    sgtitle([participants{p} '- HGF2 with perceptual uncertainty'], 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'HGF2pu_traj.png'))
    tapas_fit_plotCorr(est)
    sgtitle([participants{p} '- HGF2 with perceptual uncertainty'], 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'HGF2pu_corr.png'))
    close all
    
    simulated_y = [];
    for i = 1:10
        sim = tapas_simModel(u, ...
            'tapas_ehgf_binary_pu', ...
            est.p_prc.p,...
            'tapas_logrt_linear_binary_hgf2', ...
            est.p_obs.p, ...
            i); % different seed
        simulated_y = [simulated_y , exp(sim.y)];
    end
    writematrix(simulated_y, fullfile(part_path, [participants{p} '_HGF2pu_simulated_resp.csv']))
    
    
    %% HGF3
    hgf_obs_model = tapas_logrt_linear_binary_hgf3_config();

    % Get default priors
    c = tapas_ehgf_binary_config();
    
    % Set priors
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
    LMEs(end+1) = est.optim.LME;
    models{end+1} = est;
    labels{end+1} = 'hgf3';
    
    % Save trajectories for this model

    est.traj.ratings = rating;
    est.traj.pain = pain;
    est.traj.rt = rt;
    est.traj.choice = choice;
    est.traj.correct = correct;
    writetable(struct2table(est.traj), fullfile(part_path, [participants{p} '_HGF3_uncertainty_traj.csv']))
    
    % Save parameters for this model
    obs_table = struct2table(est.p_obs);
    obs_table.p = [];
    obs_table.ptrans =[];
    writetable(obs_table, fullfile(part_path, [participants{p} '_HGF3_uncertainty_obsparams.csv']))
    
    prc_table = table();
    prc_table.om2 = est.p_prc.om(2);
    prc_table.om3 = est.p_prc.om(3);
    writetable(prc_table, fullfile(part_path, [participants{p} '_HGF3_uncertainty_prcparams.csv']))
    
    
    %Make plots
    % Rescale Y between 0 and 1 to fit in plot
    est.y_original = est.y; % Keep original
    est.y = rescale(est.y);
    
    tapas_hgf_binary_plotTraj_nosd(est)
    sgtitle([participants{p} '- HGF3 without perceptual uncertainty'], 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'HGF3_traj.png'))
    tapas_fit_plotCorr(est)
    sgtitle([participants{p} '- HGF3 without perceptual uncertainty'], 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'HGF3_corr.png'))
    close all
    
    
    simulated_y = [];
    for i = 1:10
        sim = tapas_simModel(u, ...
            'tapas_ehgf_binary', ...
            est.p_prc.p,...
            'tapas_logrt_linear_binary_hgf3', ...
            est.p_obs.p, ...
            i); % different seed
        simulated_y = [simulated_y , exp(sim.y)];
    end
    writematrix(simulated_y, fullfile(part_path, [participants{p} '_HGF3_simulated_resp.csv']))
    
    
    %% HGF3 pu
    hgf_obs_model = tapas_logrt_linear_binary_hgf3_config();

    % Get default priors
    c = tapas_ehgf_binary_pu_config();
    
    % Set priors
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
    LMEs(end+1) = est.optim.LME;
    models{end+1} = est;
    labels{end+1} = 'hgf3pu';
    
    % Save trajectories for this model
    est.traj.ratings = rating;
    est.traj.pain = pain;
    est.traj.rt = rt;
    est.traj.choice = choice;
    est.traj.correct = correct;
    writetable(struct2table(est.traj), fullfile(part_path, [participants{p} '_HGF3pu_uncertainty_traj.csv']))
    
    % Save parameters for this model
    obs_table = struct2table(est.p_obs);
    obs_table.p = [];
    obs_table.ptrans =[];
    writetable(obs_table, fullfile(part_path, [participants{p} '_HGF3pu_uncertainty_obsparams.csv']))
    
    prc_table = table();
    prc_table.om2 = est.p_prc.om(2);
    prc_table.om3 = est.p_prc.om(3);
    prc_table.al = est.p_prc.al(1);
    writetable(prc_table, fullfile(part_path, [participants{p} '_HGF3pu_uncertainty_prcparams.csv']))
    
    
    %Make plots
    % Rescale Y between 0 and 1 to fit in plot
    est.y_original = est.y; % Keep original
    est.y = rescale(est.y);
    tapas_hgf_binary_plotTraj_nosd(est)
    sgtitle([participants{p} '- HGF3 with perceptual uncertainty'], 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'HGF3pu_traj.png'))
    tapas_fit_plotCorr(est)
    sgtitle([participants{p} '- HGF3 with perceptual uncertainty'], 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'HGF3pu_corr.png'))
    close all

    simulated_y = [];
    for i = 1:10
        sim = tapas_simModel(u, ...
            'tapas_ehgf_binary_pu', ...
            est.p_prc.p,...
            'tapas_logrt_linear_binary_hgf3', ...
            est.p_obs.p, ...
            i); % different seed
        simulated_y = [simulated_y , exp(sim.y)];
    end
    
    writematrix(simulated_y, fullfile(part_path, [participants{p} '_HGF3pu_simulated_resp.csv']))
    

    % Rescolar-Wagner
    rw_config = tapas_rw_binary_config();
    rw_obs = tapas_logrt_linear_binary_rw_config();

    % Set priors
    rw_config.logitv_0sa = 1;
    rw_config.priorsas = [rw_config.logitv_0sa,rw_config.logitalsa];

    % Fit model with perceptual
    est  = tapas_fitModel(logrt, ...  % Responses are log rt (missed trials == NaN)
        u, ...  % Inputs are contingencies
        rw_config, ...  % Perceptual model
        rw_obs, ... % Response model
        opt_algo);
    est.prc_model = 'rw';
    models{end+1} = est;
    LMEs(end+1) = est.optim.LME;
    labels{end+1} = 'rw';
    
    % Rescale Y between 0 and 1 to fit in plot
    est.traj.ratings = rating;
    est.traj.pain = pain;
    est.traj.rt = rt;
    est.traj.choice = choice;
    est.traj.correct = correct;
    
    % Save trajectories for this model
    writetable(struct2table(est.traj), fullfile(part_path, [participants{p} '_rw_traj.csv']))
    
    % Save parameters for this model
    obs_table = struct2table(est.p_obs);
    obs_table.p = [];
    obs_table.ptrans =[];
    writetable(obs_table, fullfile(part_path, [participants{p} '_rw_obsparams.csv']))
    
    prc_table = struct2table(est.p_prc);
    prc_table.p = [];
    prc_table.ptrans =[];
    writetable(prc_table, fullfile(part_path, [participants{p} '_rw_prcparams.csv']))
    
    figure;
    plot(est.traj.vhat)
    title([participants{p} '- Rescorla-Wagner trajectories'], 'FontSize', 25)
    legend('Expectation (vhat)', 'FontSize', 20)
    saveas(gcf, fullfile(part_path, 'RW_traj.png'))
    close all
    
    % Simulate data with this model for this participant
    simulated_y = [];
    for i = 1:10
        sim = tapas_simModel(u, ...
            'tapas_rw_binary', ...
            est.p_prc.p,...
            'tapas_logrt_linear_binary_rw', ...
            est.p_obs.p, ...
            i); % different seed
        simulated_y = [simulated_y , exp(sim.y)];
    end
    writematrix(simulated_y, fullfile(part_path, [participants{p} '_rw_simulated_resp.csv']))
    
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
    
    LMEs (end+1) = est.optim.LME;
    est.prc_model = 'ph';
    models{end+1} = est;
    labels{end+1} = 'ph';
    
    
    % Rescale Y between 0 and 1 to fit in plot
    est.traj.ratings = rating;
    est.traj.pain = pain;
    est.traj.rt = rt;
    est.traj.choice = choice;
    est.traj.correct = correct;
    
    % Save trajectories for this model
    writetable(struct2table(est.traj), fullfile(part_path, [participants{p} '_ph_traj.csv']))
    
    % Save parameters for this model
    obs_table = struct2table(est.p_obs);
    obs_table.p = [];
    obs_table.ptrans =[];
    writetable(obs_table, fullfile(part_path, [participants{p} '_ph_obsparams.csv']))
    
    prc_table = struct2table(est.p_prc);
    prc_table.p = [];
    prc_table.ptrans =[];
    writetable(prc_table, fullfile(part_path, [participants{p} '_ph_prcparams.csv']))
    
    
    figure;
    plot(est.traj.vhat)
    title( [participants{p} '- Pearce-Hall trajectories'], 'FontSize', 25)
    hold on
    plot(est.traj.al)
    legend('Expectation (vhat)','Associability (al)','FontSize', 20)
    saveas(gcf, fullfile(part_path, 'PH_traj.png'))
    close all
    
    
    simulated_y = [];
    for i = 1:10
        sim = tapas_simModel(u, ...
            'tapas_ph_binary', ...
            est.p_prc.p,...
            'tapas_logrt_linear_binary_ph', ...
            est.p_obs.p, ...
            i); % different seed
        simulated_y = [simulated_y , exp(sim.y)];
    end
    writematrix(simulated_y, fullfile(part_path, [participants{p} '_ph_simulated_resp.csv']))
    
    
    % Make LME figure
    labels = categorical(cellstr([labels{:}]));
    
    figure;
    bar(labels, LMEs);
    title( 'LMEs', 'FontSize', 25)
    saveas(gcf, fullfile(part_path, 'LMEs.png'))
    close all
    
    concatenated_labels = 'hgf2hgf2puhgf3hgf3purwph';  % Example concatenated string

    % Split the string into individual labels using a fixed length (e.g., 4 characters)
    labels = {'hgf2', 'hgf2pu', 'hgf3', 'hgf3pu', 'rw', 'ph'};  % Example, but you need to extract labels properly

    % Convert labels to categorical
    labels = categorical(labels);

    lme_part = LMEs;
    model_table = labels';
    tab = table(model_table, lme_part');
    writetable(tab, fullfile(part_path, 'LMEs.csv'))
    
    % Save LMEs
    if ~isreal(LMEs) % Make sure all ok
        stophere
    end
    
    parsave(fullfile(part_path,  'models.mat'), 'models')
    
    close all % Close all plots
end
diary off


%
function parsave(fname, x)
save(fname, 'x')
end

