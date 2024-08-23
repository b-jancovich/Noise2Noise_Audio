% n2n_trainset_testset_builder_STEP3.m

% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
clear
close all
clc

%% Build the datastore

% Create AudioDatastore
ads = batchAudioDatastore(isolated_detections_wav_path, ...
    'FileExtensions', '.wav', ...
    'MiniBatchSize', miniBatchSize, ...
    'LabelSource', 'fileNames');

%% Test Which Features To Use for (Dis)similarity Hashing

% Compute window size
Fs = ads.SampleRate;
windowLen = floor(stationarityThreshold * Fs); % Length of analysis window (samples)

% Select features that are the best descriptors of signal dissimilarity.
features = selectAudioFeatures(ads, windowLen, overlapPercent, nFFT, sampleSize, p);
featuresToUse = features.featureNames;

%% Local (Dis)Similarity Hashing

% Run LDH Matching using selected features
reset(ads)
dissimilarPairs = LDHPairs(ads, soiHiFreq, soiLoFreq, ...
    stationarityThreshold, overlapPercent, nFFT, fadeLen, featuresToUse);

%% Test similarity
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

    % Get measure of similarity for each pair
    [similarity, ~] = signalSimilarity(x1, x2, Fs);
    fprintf('Similarity of pair # %d is %d\n', i, similarity)

    % Put in table
    signalPairs = [signalPairs; {sig1, sig2, similarity}];
end

% Sort similarity from least to most similar
sortedT = sortrows(signalPairs, 'similarity', 'ascend');

%% Save the training files out to their new locations in a batch script

for i = 1:nTrainingPairs
    
    % Define new file paths, with new names for rename operation
    destinationNewFileName_input = fullfile(n2n_train_inputs, ['train_input_', num2str(i), '.wav']);
    destinationNewFileName_target = fullfile(n2n_train_targets, ['train_target_', num2str(i), '.wav']);

    % Copy input & target files to new location with old names
    copyfile(signalPairs.sig1{i}, destinationNewFileName_input);
    copyfile(signalPairs.sig2{i}, destinationNewFileName_target);
end

%% Save the training dataset metadata

save(fullfile(n2n_dataset_root, 'signalPairs.mat'), 'signalPairs', '-v7.3'); 

%% Prepare to build the testing dataset

% Load the noiseless detection from which to build the test dataset.
[noiseless_detection, ~] = audioread(noiseless_detection_path);

% Load the library of noise samples
load(noise_lib_path);

% Preprocess clean audio (Normalize and DC Subtract)
noiseless_detection = noiseless_detection ./ max(abs(noiseless_detection));
noiseless_detection = noiseless_detection - mean(noiseless_detection);

% Apply Fade-in and fade-out
window = hann(fadeLenNoiseless*2);
fadeIn = window(1:fadeLenNoiseless);
noiseless_detection(1:fadeLenNoiseless) = noiseless_detection(1:fadeLenNoiseless) .* fadeIn;
noiseless_detection(end-fadeLenNoiseless+1:end) = noiseless_detection(end-fadeLenNoiseless+1:end) .* flip(fadeIn);

% Adjust length of noiseless audio to match training data:
if length(noiseless_detection) > ads.SignalLength
    noiseless_detection = noiseless_detection(1:ads.SignalLength);
elseif length(noiseless_detection) < ads.SignalLength
    noiseless_detection = [noiseless_detection; zeros(ads.SignalLength-length(noiseless_detection), 1)];
end

% Initialise augmentation variables
sigma = (snrRange(2) - snrRange(1)) / 4;
pd = makedist('Normal', 'mu', snrMean, 'sigma', sigma);
trunc_pd = truncate(pd, snrRange(1), snrRange(2));
SNRs = random(trunc_pd, nTestingPairs, 1);
nYears = length([noiseLibrary.year]); % number of years of data in noise library
totalShift = shiftRate * nYears; % total change in call frequency over nYears (Hz)
endFreq = initialFreq - totalShift; % Final frequency after nYears (Hz)
stRealChange = 12 * log2(endFreq / initialFreq); % Monopolar range of random pitch shift (semitones)
stAugRangeBipolar = (stRealChange * 1.2) / 2; % Total range of random pitch shift in test set including some extra (Semitones)

% Note: stAugRangeBipolar includes and extra +/- 0.2 semitones of shift on
% top of what is actually seen in the real dataset to add extra generalizability 
% and add robustness to natural variances (doppler shifts, freak whales etc.).

%% Build Augmentation Object

disp("Building data augmentation object...")

% Set up augmentation object
augmenter = audioDataAugmenter('AugmentationParameterSource','random', ...
                            'NumAugmentations', nTestingPairs, ...
                            "AugmentationMode","sequential", ...
                            ... % Time Stretching
                            'TimeStretchProbability',0.5, ...
                            'SpeedupFactorRange', speedupFactorRange,...
                            ... % Pitch Shifting
                            "PitchShiftProbability", 0.5, ...
                            "SemitoneShiftRange", [stAugRangeBipolar, -stAugRangeBipolar], ...
                            ... % Time Shifting
                            "TimeShiftProbability", 0.5, ...
                            "TimeShiftRange", timeShiftRange,...
                            ... % Deactiviate Other Augmentations
                            "ApplyVolumeControl",false, ...
                            "ApplyAddNoise", false,...
                            "VolumeControlProbability", 0,...
                            "AddNoiseProbability", 0,...
                            "VolumeGainRange",  [0, 0],...
                            "SNRRange", [0, 0]);

% Define 2nd Order Butterworth Low Pass Filter & Add to Augmenter
lpfHandle = @(x, LowPassCutoff) filtfilt(butter(2, LowPassCutoff/(0.5*Fs), 'low'), 1, x);
addAugmentationMethod(augmenter, 'LowPassFilter', lpfHandle,...
    'AugmentationParameter', 'LowPassCutoff', ...
    'ParameterRange', lpfRange);

% Define 2nd Order Butterworth High Pass Filter & Add to Augmenter
hpfHandle = @(x, HighPassCutoff) filtfilt(butter(2, HighPassCutoff/(0.5*Fs), 'high'), 1, x);
addAugmentationMethod(augmenter, 'HighPassFilter', hpfHandle,...
    'AugmentationParameter', 'HighPassCutoff', ...
    'ParameterRange', hpfRange);

% Define Doppler Shifter & Add to Augmenter, then set probability and range
dopplerHandle = @(x, SourceVelocity) dopplerShift(x, Fs, SourceVelocity, c);
addAugmentationMethod(augmenter,'DopplerShift',dopplerHandle,...
    'AugmentationParameter', 'SourceVelocity', ...
    'ParameterRange', SourceVelocityRange);

%% Build Augmented Dataset

disp("Generating augmented clean signal dataset...")

% Run Clean Signal Augmentation
cleanSignals = augment(augmenter, noiseless_detection, Fs);

% Normalize all clean signals
for i = 1:nTestingPairs
    cleanSignals.Audio{i} = cleanSignals.Audio{i} ./ max(abs(cleanSignals.Audio{i}));

    % Adjust length of noiseless audio to match training data:
    if length(cleanSignals.Audio{i}) > ads.SignalLength
        cleanSignals.Audio{i} = cleanSignals.Audio{i}(1:ads.SignalLength);
    elseif length(cleanSignals.Audio{i}) < ads.SignalLength
        cleanSignals.Audio{i} = [cleanSignals.Audio{i}; zeros(ads.SignalLength-length(cleanSignals.Audio{i}), 1)];
    end
end

disp("Generating corrupted signal dataset...")

corruptedSignals = cell(nTestingPairs, 1);

% Generate synthetic data
for i = 1:nTestingPairs
    % Randomly select a noise vector year:
    randomYear = randi([min([noiseLibrary.Year]), max([noiseLibrary.Year])]);

    % Find indices of rows in noiseLibrary with matching year
    matchingYearIndices = find([noiseLibrary.Year] == randomYear);
    
    % Randomly select one of these rows
    noiseYearIdx = matchingYearIndices(randi(length(matchingYearIndices)));
    
    % Get the length of the audioData for that row
    len = length(noiseLibrary(noiseYearIdx).audioData);
    
    % Randomly select a starting index for the noise subsequence
    startIdx = randi(len - ads.SignalLength + 1);
    
    % Extract the noise subsequence
    signalNoise = noiseLibrary(noiseYearIdx).audioData(startIdx:startIdx + ads.SignalLength - 1);
    
    % Ensure the noise is a column vector
    signalNoise = signalNoise(:);

    % DC Filter the noise
    signalNoise = signalNoise - mean(signalNoise);

    % Extract this iterations's clean signal, ensuring it is a column vec
    signalClean = cleanSignals.Audio{i}(:);

    % Scale the signalNoise and signalClean to the specified SNR
    signalCorrupted = sigNoiseMixer(signalClean, signalNoise, SNRs(i), 1);
       
    % Normalize the corrupted signal
    corruptedSignals{i} = signalCorrupted ./ max(abs(signalCorrupted));
    disp(['Completed corrupted training sample # ', num2str(i)])
end

%% Save out data

disp("Saving training and testing datasets...")

for i = 1:nTesting
    % Filename for clean "groundtruth" Data
    fileNameTestGT = fullfile(n2n_test_groundTruth, ...
        ['testing_groundtruth_signal_', num2str(i),'.wav']);

    % Filename for corrupted "input" Data
    fileNameTestCorrupted = fullfile(n2n_test_inputs, ...
        ['test_input_signal_', num2str(i), '_SNR_', num2str(SNRs(i)), 'dB.wav']);

    % Save clean data to disk
    audiowrite(fileNameTestGT, cleanSignals.Audio{i}, Fs);

    % Save clean+noise data to disk
    audiowrite(fileNameTestCorrupted, corruptedSignals{i}, Fs);
end

%% Helper Functions:

function y = dopplerShift(x, fs, v, c)
    % Apply Doppler shift to audio signal
    %
    % Inputs:
    % x: input audio signal
    % fs: sampling frequency of x (Hz)
    % v: velocity of source (m/s), positive if moving towards receiver
    % c: speed of sound (m/s)
    %
    % Output:
    % y: Doppler-shifted audio signal
    
    % Calculate the Doppler shift factor
    alpha = c / (c - v);
    
    % Create time vectors
    t_original = (0:length(x)-1) / fs;
    t_shifted = t_original / alpha;
    
    % Interpolate the signal
    y = interp1(t_original, x, t_shifted, 'linear', 0);
    
    % Adjust the amplitude to conserve energy
    y = y * sqrt(alpha);
    
    % Ensure the output matches the dimension of the input
    if length(y) > length(x)
        y = y(1:length(x));
    elseif length(y) < length(x)
        y(end+1:length(x)) = 0;
    end
    
    % Ensure the output has the same orientation as the input
    if iscolumn(x)
        y = y(:);
    elseif isrow(x)
        y = y(:)';
    end

end