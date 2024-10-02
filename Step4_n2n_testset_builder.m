%% Step4_n2n_testset_builder.m - INTERRUPTABLE VERSION
%
% DESCRIPTION:
%   This script is the fourth step in building a noise-to-noise (N2N) training
%   dataset for audio processing. It generates a test dataset by augmenting
%   a clean whale song detection and mixing it with noise samples from the
%   noise library. The resulting dataset includes both clean (ground truth)
%   and corrupted (input) audio files for testing the N2N model.
%
%   This script uses a checkpoint system, allowing it to be interrupted
%   and resumed from the last completed step. The checkpoint file is
%   automatically managed by the script. Simply run it again if interrupted.
%   Script operations are logged in a text file that will be saved to the
%   current MATLAB directory.
%
% KEY FEATURES:
%   1. Loads a clean whale song detection and noise samples
%   2. Applies various audio augmentations to the clean signal
%   3. Mixes augmented clean signals with noise at specified SNR levels
%   4. Generates both clean (ground truth) and corrupted (input) test files
%   5. Supports interruptible execution through checkpointing
%   6. Implements realistic signal degradation over time
%   7. Applies Doppler shift to simulate moving sound sources
%
% DEPENDENCIES:
%   - MATLAB Audio Toolbox
%   - MATLAB Signal Processing Toolbox
%   - Custom functions: sigNoiseMixer, dopplerShift, Utilities repo
%
% SCRIPT WORKFLOW:
%   1. Initialization and Configuration
%      - Loads project configuration from config.m
%      - Sets up the audio datastore for the training set
%
%   2. Test Dataset Preparation
%      - Loads and preprocesses the clean whale song detection
%      - Prepares the noise library and augmentation parameters
%
%   3. Signal Augmentation
%      - Builds a data augmentation object with various transformations
%      - Applies augmentations to the clean signal
%
%   4. Test Dataset Generation
%      - Mixes augmented clean signals with noise at specified SNR levels
%      - Saves clean (ground truth) and corrupted (input) audio files
%
% USAGE:
%   1. Ensure all dependencies are installed and custom functions are in the MATLAB path
%   2. Set appropriate parameters in config.m
%   3. Run the script
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
%%
clear
close all
clc

%% Set environment
opEnv = 1;
% 1 = Use the paths in config.m that relate to my windows laptop
% 2 = Use the paths in config.m that relate to the Katana Cluster

%% Load project configuration file

here = pwd;
run(fullfile(here, 'config.m'));
disp('Loaded N2N Config file.')

%% Add utilities repo to path
[gitRootPath, ~, ~] = fileparts(here);
addpath(fullfile(gitRootPath, 'Utilities'));

%% Build the Training set datastore to get matching signal lengths & Fs

% Create AudioDatastore
ads = batchAudioDatastore(isolated_detections_wav_path, ...
    'FileExtensions', '.wav', ...
    'MiniBatchSize', miniBatchSize, ...
    'LabelSource', 'fileNames');

%% Prepare to build the testing dataset

% Load the noiseless detection from which to build the test dataset.
[noiseless_detection, ~] = audioread(noiseless_detection_path);

% Get the list of noise samples
noiseFileList = dir(fullfile(noise_lib_path, '*.wav'));

% Extract year from filename and add it as a new field to noiseFileList
for i = 1:length(noiseFileList)
    % Extract the date-time string from the filename
    dateTimeStr = regexp(noiseFileList(i).name, '\d{6}-\d{6}', 'match');
    
    if ~isempty(dateTimeStr)
        % Extract the first two digits of the date-time string
        yearStr = dateTimeStr{1}(1:2);
        
        % Convert to full year (assuming 20xx)
        fullYear = str2double(['20', yearStr]);
        
        % Add the year as a new field to the structure
        noiseFileList(i).year = fullYear;
    else
        % If no date-time string found, set year to NaN
        noiseFileList(i).year = NaN;
    end
end

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
nYears = length(unique([noiseFileList.year])); % number of years of data in noise library
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
augmenter = noiseDataAugmenter('AugmentationParameterSource','random', ...
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
    randomYear = randi([min([noiseFileList.year]), max([noiseFileList.year])]);

    % Find indices of rows in noiseLibrary with matching Year
    matchingYearIndices = find([noiseFileList.year] == randomYear);
    
    % Randomly select one of these rows
    noiseYearIdx = matchingYearIndices(randi(length(matchingYearIndices)));

    % Load the Audio for that row
    [noiseData, ~] = audioread(fullfile(noise_lib_path, noiseFileList(noiseYearIdx).name));
    
    % Randomly select a starting index for the noise subsequence
    startIdx = randi(length(noiseData) - ads.SignalLength + 1);
    
    % Extract the noise subsequence
    noiseData = noiseData(startIdx:startIdx + ads.SignalLength - 1);
    
    % Ensure the noise is a column vector
    noiseData = noiseData(:);

    % DC Filter the noise
    noiseData = noiseData - mean(noiseData);

    % Extract this iterations's clean signal, ensuring it is a column vec
    signalClean = cleanSignals.Audio{i}(:);

    % Scale the signalNoise and signalClean to the specified SNR
    signalCorrupted = sigNoiseMixer(signalClean, noiseData, SNRs(i), 1);
       
    % Normalize the corrupted signal
    signalCorrupted = signalCorrupted ./ max(abs(signalCorrupted));

    % Filename for clean "groundtruth" Data
    fileNameTestGT = fullfile(n2n_test_groundTruth, ...
        ['testing_groundtruth_signal_', num2str(i),'.wav']);

    % Filename for corrupted "input" Data
    fileNameTestCorrupted = fullfile(n2n_test_inputs, ...
        ['test_input_signal_', num2str(i), '_SNR_', num2str(SNRs(i)), 'dB.wav']);

    % Save clean data to disk
    audiowrite(fileNameTestGT, signalClean, Fs);

    % Save clean+noise data to disk
    audiowrite(fileNameTestCorrupted, signalCorrupted, Fs);
end

%% Helper Functions:

% Doppler shift moved to Utilities.