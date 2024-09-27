% n2n_trainset_builder_STEP3.m - INTERRUPTABLE VERSION

% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
clear
close all
clc

%% Begin logging
ts = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
logname = ['step3_script_log_', ts, '.txt'];
diary(logname)

%% Set Operating Environment
% 1 = Use the paths in config.m that relate to my windows laptop
% 2 = Use the paths in config.m that relate to the Katana Cluster
opEnv = 1;

% Compute file retrievals in serial or parallel:
mode = 'parallel';
% Note: parallel requires MATLAB parallel computing toolbox

%% Load project configuration file

here = pwd;
run(fullfile(here, 'config.m'));
disp('Loaded N2N Config file.')

%% Build checkpoint file

checkpointFile = fullfile(here, 'n2n_trainset_builder_checkpoint.mat');

%% Check for existing checkpoint
if exist(checkpointFile, 'file')
    load(checkpointFile);
    disp('Loaded checkpoint. Resuming from last saved state.');
else
    currentStep = 1;
    disp('No checkpoint found. Starting from the beginning.');
end

%% Build the datastore

% % Debug test inputs
% isolated_detections_wav_path = 'D:\LSH_TEST';
% miniBatchSize = 100;
% sampleSize = 200;
% nTrainingPairs = 300;

if currentStep <= 1
    % Create AudioDatastore
    ads = batchAudioDatastore(isolated_detections_wav_path, ...
        'FileExtensions', '.wav', ...
        'MiniBatchSize', miniBatchSize, ...
        'LabelSource', 'fileNames', ...
        'MinSNR', minSNR);

    fprintf('Min. SNR setting of %d dB results in %d files in the datastore.\n', ...
        minSNR, length(ads.Files))
    
    % Save checkpoint
    currentStep = 2;
    save(checkpointFile, 'currentStep', 'ads');
end

%% Test Which Features To Use for (Dis)similarity Hashing

if currentStep <= 2
    fprintf('Running automated feature selection on %d signals...\n', sampleSize);

    % Compute window size
    windowLen = floor(stationarityThreshold * Fs); % Length of analysis window (samples)

    % Check inputs are valid
    assert(sampleSize < length(ads.Files), '"sampleSize" must be smaller than the number of files in the datastore path.')

    % Select features that are the best descriptors of signal dissimilarity.
    features = selectAudioFeatures(ads, windowLen, overlapPercent, nFFT, sampleSize, p);
    featuresToUse = features.featureNames;

    disp('Selected most salient features.')
    
    % Save checkpoint
    currentStep = 3;
    save(checkpointFile, 'currentStep', 'ads', 'features', 'featuresToUse');
end

%% Local (Dis)Similarity Hashing

if currentStep <= 3
    disp('Beginning Local Dissimilarity Hashing...')

    profile on

    % Run LDH Matching using selected features
    reset(ads)
    dissimilarPairs = LDHPairs_v2(ads, soiHiFreq, stationarityThreshold, ...
        overlapPercent, nFFT, fadeLen, featuresToUse, hashTables, bandSize);

    profile off

    % Get the profile information
    p = profile('info');

    % Generate a report and save it as PDF
    profsave(p, fullfile(pwd, 'LDHPairs_profile_report_ismember_simple'));
    
    % Save checkpoint
    currentStep = 4;
    save(checkpointFile, 'currentStep', 'ads', 'features', 'featuresToUse', 'dissimilarPairs');
end

%% Test & Sort Pairs by Similarity (Max of 2D Cross Correlation)

if currentStep <= 4
    nPairs = length(dissimilarPairs);
    similarity = zeros(nPairs, 1);
    rawMetrics = zeros(nPairs, 4);

    signalPairs = table('Size', [0, 3], ...
        'VariableTypes', {'string', 'string', 'double'}, ...
        'VariableNames', {'sig1', 'sig2', 'similarity'});

    % Get file paths and names for pairs & Test similarity
    for i = 1:nPairs
        % Search "ads.Files.index" for dissimilarPair ID's
        index1 = find([ads.Files.index] == dissimilarPairs(i, 1));
        index2 = find([ads.Files.index] == dissimilarPairs(i, 2));

        % Get paths to corresponding files
        sig1 = fullfile(ads.Files(index1).folder, ads.Files(index1).name);
        sig2 = fullfile(ads.Files(index2).folder, ads.Files(index2).name);

        % Load files in the pair
        [x1, ~] = audioread(sig1);
        [x2, ~] = audioread(sig2);

        % Only attempt to measre similarity if signals are valid
        if isempty(x1) || isempty(x2)
            similarity = NaN;  % Use NaN for empty signals
        elseif any(isnan(x1)) || any(isnan(x2))
            similarity = NaN;  % Use NaN for signals with any NaN values
        elseif all(x1 == 0) || all(x2 == 0)
            similarity = NaN;  % Use NaN for signals that are all zeros
        else
            % Both signals are non-empty, contain no NaNs, and are not all zeros
            similarity = signalSimilarity(x1, x2, Fs);
        end
        
        fprintf('Similarity of pair # %d is %d\n', i, similarity)

        % Put in table
        signalPairs = [signalPairs; {sig1, sig2, similarity}];
        
        % Save checkpoint every 100 pairs
        if mod(i, 100) == 0
            save(checkpointFile, 'currentStep', 'ads', 'features', 'featuresToUse', 'dissimilarPairs', 'signalPairs', 'i');
        end
    end

    % Sort similarity from least to most similar
    signalPairs = sortrows(signalPairs, 'similarity', 'ascend');
    
    % Save checkpoint
    currentStep = 5;
    save(checkpointFile, 'currentStep', 'ads', 'features', 'featuresToUse', 'dissimilarPairs', 'signalPairs');
end

%% Validation figure

if currentStep <= 5
    % Identify the most and least similar signal pairs
    [~, mostSimilarPair] = max(signalPairs.similarity(isfinite(signalPairs.similarity)));
    [~, leastSimilarPair] = min(signalPairs.similarity(isfinite(signalPairs.similarity)));

    % Get signals
    mostSim1 = audioread(signalPairs.sig1{mostSimilarPair});
    mostSim2 = audioread(signalPairs.sig2{mostSimilarPair});
    leastSim1 = audioread(signalPairs.sig1{leastSimilarPair});
    leastSim2 = audioread(signalPairs.sig2{leastSimilarPair});

    % Get Peak Correlation Value
    mostSimCCPeak = signalPairs.similarity(mostSimilarPair);
    leastSimCCPeak = signalPairs.similarity(leastSimilarPair);

    % Norm and DC filt
    mostSim1 = (mostSim1 - mean(mostSim1)) / (max(mostSim1) - mean(mostSim1));
    mostSim2 = (mostSim2 - mean(mostSim2)) / (max(mostSim2) - mean(mostSim2));
    leastSim1 = (leastSim1 - mean(leastSim1)) / (max(leastSim1) - mean(leastSim1));
    leastSim2 = (leastSim2 - mean(leastSim2)) / (max(leastSim2) - mean(leastSim2));

    % Compute Spectrograms
    windowLen = floor(stationarityThreshold * Fs);
    overlap = floor(windowLen * (overlapPercent / 100));

    [s_mostSim1, f, t] = spectrogram(mostSim1, windowLen, overlap, nFFT, Fs);
    [s_mostSim2, ~, ~] = spectrogram(mostSim2, windowLen, overlap, nFFT, Fs);
    [s_leastSim1, ~, ~] = spectrogram(leastSim1, windowLen, overlap, nFFT, Fs);
    [s_leastSim2, ~, ~] = spectrogram(leastSim2, windowLen, overlap, nFFT, Fs);

    [s_mostSimCross, ~, ~] = xspectrogram(mostSim1 , mostSim2, windowLen, overlap, nFFT, Fs);
    [s_leastSimCross, ~, ~] = xspectrogram(leastSim1 , leastSim2, windowLen, overlap, nFFT, Fs);

    % Extract Magnitude from Spectrograms
    s_mostSim1 = abs(s_mostSim1);
    s_mostSim2 = abs(s_mostSim2);
    s_leastSim1 = abs(s_leastSim1);
    s_leastSim2 = abs(s_leastSim2);

    s_mostSimCross = abs(s_mostSimCross);
    s_leastSimCross = abs(s_leastSimCross);

    % Normalize Spectrograms
    s_mostSim1 = s_mostSim1 ./ max(abs(s_mostSim1), [], 'all');
    s_mostSim2 = s_mostSim2 ./ max(abs(s_mostSim2), [], 'all');
    s_leastSim1 = s_leastSim1 ./ max(abs(s_leastSim1), [], 'all');
    s_leastSim2 = s_leastSim2 ./ max(abs(s_leastSim2), [], 'all');

    % % Compute deltas between pairs
    % mostSimDelta = abs((s_mostSim1) - (s_mostSim2));
    % leastSimDelta = abs((s_leastSim1) - (s_leastSim2));

    % Draw figure
    figure(1)
    tiledlayout(2,3)
    nexttile
    imagesc(t, f, mag2db(s_mostSim1))
    set(gca, "YDir", "normal")
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title(['Most Similar Pair - ccPeak = ', num2str(mostSimCCPeak), '  - Sig 1'])
    cb = colorbar;
    cb.Label.String = 'magnitude (dB, normalized)';

    nexttile
    imagesc(t, f, mag2db(s_mostSim2))
    set(gca, "YDir", "normal")
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title(['Most Similar Pair - ccPeak = ', num2str(mostSimCCPeak), '  - Sig 2'])
    cb = colorbar;
    cb.Label.String = 'magnitude (dB, normalized)';

    nexttile
    % imagesc(t, f, (mostSimDelta))
    imagesc(t, f, abs(s_mostSimCross))
    set(gca, "YDir", "normal")
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title('Most Similar Pair - abs(sig1-sig2)')
    cb = colorbar;
    cb.Label.String = 'magnitude (linear)';
    % if all(abs(mostSimDelta(:)) == 0)
    %     text(20, 60, 'Delta = 0')
    % end
    if all(abs(s_mostSimCross(:)) == 0)
        text(20, 60, 'Delta = 0')
    end

    nexttile
    imagesc(t, f, mag2db(s_leastSim1))
    set(gca, "YDir", "normal")
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title(['Least Similar Pair - ccPeak = ', num2str(leastSimCCPeak), '  - Sig 1'])
    cb = colorbar;
    cb.Label.String = 'magnitude (dB, normalized)';

    nexttile
    imagesc(t, f, mag2db(s_leastSim2))
    set(gca, "YDir", "normal")
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title(['Least Similar Pair - ccPeak = ', num2str(leastSimCCPeak), '  - Sig 2'])
    cb = colorbar;
    cb.Label.String = 'magnitude (dB, normalized)';

    nexttile
    % imagesc(t, f, (leastSimDelta))
    imagesc(t, f, abs(s_leastSimCross)) 
    set(gca, "YDir", "normal")
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title('Least Similar Pair - abs(sig1-sig2)')
    cb = colorbar;
    cb.Label.String = 'magnitude (linear)';
    % if all(abs(leastSimDelta(:)) == 0)
    %     text(20, 60, 'Delta = 0')
    % end
    if all(abs(s_leastSimCross(:)) == 0)
        text(20, 60, 'Delta = 0')
    end
    
    % Save checkpoint
    currentStep = 6;
    save(checkpointFile, 'currentStep', 'ads', 'features', 'featuresToUse', 'dissimilarPairs', 'signalPairs');
end

%% Save the training files out to their new locations

if currentStep <= 6
    for i = 1:nTrainingPairs

        % Define new file paths, with new names for rename operation
        destinationNewFileName_input = fullfile(n2n_train_inputs, ['train_input_', num2str(i), '.wav']);
        destinationNewFileName_target = fullfile(n2n_train_targets, ['train_target_', num2str(i), '.wav']);

        % Copy input & target files to new location with old names
        copyfile(signalPairs.sig1{i}, destinationNewFileName_input);
        copyfile(signalPairs.sig2{i}, destinationNewFileName_target);
    end

    % Save checkpoint
    currentStep = 7;
    save(checkpointFile, 'currentStep', 'ads', 'features', 'featuresToUse', 'dissimilarPairs', 'signalPairs');
end

%% Save the training dataset metadata

if currentStep <= 7
    save(fullfile(n2n_dataset_root, 'signalPairs.mat'), 'signalPairs', '-v7.3'); 
    clearvars signalPairs dissimilarPairs featuresToUse features
    
    % Final checkpoint (script completed)
    currentStep = 8;
    save(checkpointFile, 'currentStep');
end

% Clean up
if currentStep == 8
    delete(checkpointFile);
    disp('Script completed successfully. Checkpoint file deleted.');
end

% End logging
diary off

% OLD VERSION - NO CHECKPOINTS
% % n2n_trainset_builder_STEP3.m
% 
% % Ben Jancovich, 2024
% % Centre for Marine Science and Innovation
% % School of Biological, Earth and Environmental Sciences
% % University of New South Wales, Sydney, Australia
% %
% clear
% close all
% clc
% 
% %% Load project configuration file
% 
% here = pwd;
% run(fullfile(here, 'config.m'));
% disp('Loaded N2N Config file.')
% 
% %% Build the datastore
% 
% % % % Debug test inputs
% % isolated_detections_wav_path = 'D:\LSH_TEST';
% % miniBatchSize = 100;
% % sampleSize = 200;
% 
% % Create AudioDatastore
% ads = batchAudioDatastore(isolated_detections_wav_path, ...
%     'FileExtensions', '.wav', ...
%     'MiniBatchSize', miniBatchSize, ...
%     'LabelSource', 'fileNames', ...
%     'MinSNR', minSNR);
% 
% fprintf('Min. SNR setting of %d dB results in %d files in the datastore.\n', ...
%     minSNR, length(ads.Files))
% 
% %% Test Which Features To Use for (Dis)similarity Hashing
% 
% fprintf('Running automated feature selection on %d signals...\n', sampleSize);
% 
% % Compute window size
% windowLen = floor(stationarityThreshold * Fs); % Length of analysis window (samples)
% 
% % Check inputs are valid
% assert(sampleSize < length(ads.Files), '"sampleSize" must be smaller than the number of files in the datastore path.')
% 
% % Select features that are the best descriptors of signal dissimilarity.
% features = selectAudioFeatures(ads, windowLen, overlapPercent, nFFT, sampleSize, p);
% featuresToUse = features.featureNames;
% 
% disp('Selected most salient features.')
% 
% %% Local (Dis)Similarity Hashing
% 
% disp('Beginning Local Dissimilarity Hashing...')
% 
% profile on
% 
% % Run LDH Matching using selected features
% reset(ads)
% dissimilarPairs = LDHPairs_v2(ads, soiHiFreq, stationarityThreshold, ...
%     overlapPercent, nFFT, fadeLen, featuresToUse, hashTables, bandSize);
% 
% profile off
% 
% % Get the profile information
% p = profile('info');
% 
% % Generate a report and save it as PDF
% profsave(p, fullfile(pwd, 'LDHPairs_profile_report_ismember_simple'));
% 
% %% Test & Sort Pairs by Similarity (Max of 2D Cross Correlation)
% 
% nPairs = length(dissimilarPairs);
% similarity = zeros(nPairs, 1);
% rawMetrics = zeros(nPairs, 4);
% 
% signalPairs = table('Size', [0, 3], ...
%     'VariableTypes', {'string', 'string', 'double'}, ...
%     'VariableNames', {'sig1', 'sig2', 'similarity'});
% 
% % Get file paths and names for pairs & Test similarity
% for i = 1:nPairs
%     % Search "ads.Files.index" for dissimilarPair ID's
%     index1 = find([ads.Files.index] == dissimilarPairs(i, 1));
%     index2 = find([ads.Files.index] == dissimilarPairs(i, 2));
% 
%     % Get paths to corresponding files
%     sig1 = fullfile(ads.Files(index1).folder, ads.Files(index1).name);
%     sig2 = fullfile(ads.Files(index2).folder, ads.Files(index2).name);
% 
%     % Load files in the pair
%     [x1, ~] = audioread(sig1);
%     [x2, ~] = audioread(sig2);
% 
%     % Only attempt to measre similarity if signals are valid
%     if isempty(x1) || isempty(x2)
%         similarity = NaN;  % Use NaN for empty signals
%     elseif any(isnan(x1)) || any(isnan(x2))
%         similarity = NaN;  % Use NaN for signals with any NaN values
%     elseif all(x1 == 0) || all(x2 == 0)
%         similarity = NaN;  % Use NaN for signals that are all zeros
%     else
%         % Both signals are non-empty, contain no NaNs, and are not all zeros
%         similarity = signalSimilarity(x1, x2, Fs);
%     end
% 
%     fprintf('Similarity of pair # %d is %d\n', i, similarity)
% 
%     % Put in table
%     signalPairs = [signalPairs; {sig1, sig2, similarity}];
% end
% 
% % Sort similarity from least to most similar
% signalPairs = sortrows(signalPairs, 'similarity', 'ascend');
% 
% %% Validation figure
% 
% % Identify the most and least similar signal pairs
% [~, mostSimilarPair] = max(signalPairs.similarity(isfinite(signalPairs.similarity)));
% [~, leastSimilarPair] = min(signalPairs.similarity(isfinite(signalPairs.similarity)));
% 
% % Get signals
% mostSim1 = audioread(signalPairs.sig1{mostSimilarPair});
% mostSim2 = audioread(signalPairs.sig2{mostSimilarPair});
% leastSim1 = audioread(signalPairs.sig1{leastSimilarPair});
% leastSim2 = audioread(signalPairs.sig2{leastSimilarPair});
% 
% % Get Peak Correlation Value
% mostSimCCPeak = signalPairs.similarity(mostSimilarPair);
% leastSimCCPeak = signalPairs.similarity(leastSimilarPair);
% 
% % Norm and DC filt
% mostSim1 = (mostSim1 - mean(mostSim1)) / (max(mostSim1) - mean(mostSim1));
% mostSim2 = (mostSim2 - mean(mostSim2)) / (max(mostSim2) - mean(mostSim2));
% leastSim1 = (leastSim1 - mean(leastSim1)) / (max(leastSim1) - mean(leastSim1));
% leastSim2 = (leastSim2 - mean(leastSim2)) / (max(leastSim2) - mean(leastSim2));
% 
% % Compute Spectrograms
% windowLen = floor(stationarityThreshold * Fs);
% overlap = floor(windowLen * (overlapPercent / 100));
% 
% [s_mostSim1, f, t] = spectrogram(mostSim1, windowLen, overlap, nFFT, Fs);
% [s_mostSim2, ~, ~] = spectrogram(mostSim2, windowLen, overlap, nFFT, Fs);
% [s_leastSim1, ~, ~] = spectrogram(leastSim1, windowLen, overlap, nFFT, Fs);
% [s_leastSim2, ~, ~] = spectrogram(leastSim2, windowLen, overlap, nFFT, Fs);
% 
% [s_mostSimCross, ~, ~] = xspectrogram(mostSim1 , mostSim2, windowLen, overlap, nFFT, Fs);
% [s_leastSimCross, ~, ~] = xspectrogram(leastSim1 , leastSim2, windowLen, overlap, nFFT, Fs);
% 
% % Extract Magnitude from Spectrograms
% s_mostSim1 = abs(s_mostSim1);
% s_mostSim2 = abs(s_mostSim2);
% s_leastSim1 = abs(s_leastSim1);
% s_leastSim2 = abs(s_leastSim2);
% 
% s_mostSimCross = abs(s_mostSimCross);
% s_leastSimCross = abs(s_leastSimCross);
% 
% % Normalize Spectrograms
% s_mostSim1 = s_mostSim1 ./ max(abs(s_mostSim1), [], 'all');
% s_mostSim2 = s_mostSim2 ./ max(abs(s_mostSim2), [], 'all');
% s_leastSim1 = s_leastSim1 ./ max(abs(s_leastSim1), [], 'all');
% s_leastSim2 = s_leastSim2 ./ max(abs(s_leastSim2), [], 'all');
% 
% % % Compute deltas between pairs
% % mostSimDelta = abs((s_mostSim1) - (s_mostSim2));
% % leastSimDelta = abs((s_leastSim1) - (s_leastSim2));
% 
% % Draw figure
% figure(1)
% tiledlayout(2,3)
% nexttile
% imagesc(t, f, mag2db(s_mostSim1))
% set(gca, "YDir", "normal")
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title(['Most Similar Pair - ccPeak = ', num2str(mostSimCCPeak), '  - Sig 1'])
% cb = colorbar;
% cb.Label.String = 'magnitude (dB, normalized)';
% 
% nexttile
% imagesc(t, f, mag2db(s_mostSim2))
% set(gca, "YDir", "normal")
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title(['Most Similar Pair - ccPeak = ', num2str(mostSimCCPeak), '  - Sig 2'])
% cb = colorbar;
% cb.Label.String = 'magnitude (dB, normalized)';
% 
% nexttile
% % imagesc(t, f, (mostSimDelta))
% imagesc(t, f, abs(s_mostSimCross))
% set(gca, "YDir", "normal")
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title('Most Similar Pair - abs(sig1-sig2)')
% cb = colorbar;
% cb.Label.String = 'magnitude (linear)';
% % if all(abs(mostSimDelta(:)) == 0)
% %     text(20, 60, 'Delta = 0')
% % end
% if all(abs(s_mostSimCross(:)) == 0)
%     text(20, 60, 'Delta = 0')
% end
% 
% nexttile
% imagesc(t, f, mag2db(s_leastSim1))
% set(gca, "YDir", "normal")
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title(['Least Similar Pair - ccPeak = ', num2str(leastSimCCPeak), '  - Sig 1'])
% cb = colorbar;
% cb.Label.String = 'magnitude (dB, normalized)';
% 
% nexttile
% imagesc(t, f, mag2db(s_leastSim2))
% set(gca, "YDir", "normal")
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title(['Least Similar Pair - ccPeak = ', num2str(leastSimCCPeak), '  - Sig 2'])
% cb = colorbar;
% cb.Label.String = 'magnitude (dB, normalized)';
% 
% nexttile
% % imagesc(t, f, (leastSimDelta))
% imagesc(t, f, abs(s_leastSimCross)) 
% set(gca, "YDir", "normal")
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% title('Least Similar Pair - abs(sig1-sig2)')
% cb = colorbar;
% cb.Label.String = 'magnitude (linear)';
% % if all(abs(leastSimDelta(:)) == 0)
% %     text(20, 60, 'Delta = 0')
% % end
% if all(abs(s_leastSimCross(:)) == 0)
%     text(20, 60, 'Delta = 0')
% end
% 
% %% Save the training files out to their new locations
% 
% % for i = 1:nTrainingPairs
% % 
% %     % Define new file paths, with new names for rename operation
% %     destinationNewFileName_input = fullfile(n2n_train_inputs, ['train_input_', num2str(i), '.wav']);
% %     destinationNewFileName_target = fullfile(n2n_train_targets, ['train_target_', num2str(i), '.wav']);
% % 
% %     % Copy input & target files to new location with old names
% %     copyfile(signalPairs.sig1{i}, destinationNewFileName_input);
% %     copyfile(signalPairs.sig2{i}, destinationNewFileName_target);
% % end
% 
% %% Save the training dataset metadata
% % 
% % save(fullfile(n2n_dataset_root, 'signalPairs.mat'), 'signalPairs', '-v7.3'); 
% % clearvars signalPairs dissimilarPairs featuresToUse features
% % 
