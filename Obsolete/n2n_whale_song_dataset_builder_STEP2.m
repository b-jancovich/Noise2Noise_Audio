% Noise2Noise Dataset Builder - STEP 2
% Searches the detection results and retrieves the time-datestamps for the
% best detections, by calculating a combined score from SNR and SINR.
%
% Ben Jancovich, 2024
% Modified by Claude, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
clear
close all
clc

%% Set parameters
soiHiFreq = 60;
soiLoFreq = 25;
fadeLen = 0.5;
nTrainingPairs = 10000;
numHashFunctions = 500;
numBands = 50;

%% Set up paths and load configuration
here = pwd;
gitRoot = here(1:regexp(here, 'Git', 'end'));
localisationPath = fullfile(gitRoot, "localisation_and_source_level_est");
run(fullfile(localisationPath, 'config.m'));

%% Set input data paths
rawAudioPath = "D:\Diego Garcia South";
detectionsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\Manue_Chagos_RawData\DGS";
wav_subdir_prefix = 'DiegoGarcia';
cleanAudioSavePath = 'C:\Users\z5439673\Git\localisation_and_source_level_est\2_DATA';

%% Set output datapath
n2n_dataset_root = 'C:\Users\z5439673\OneDrive - UNSW\H0419778\Noise2Noise_Audio';
n2n_dataset_inputs = fullfile(n2n_dataset_root, 'Inputs');
n2n_dataset_targets = fullfile(n2n_dataset_root, 'Targets'); 

%% Load Checkpoint

% Find files matching the pattern
savePattern = fullfile(n2n_dataset_root, '*_all_valid_detections_n=*.mat');
files = dir(savePattern);

if isempty(files)
    error('No matching files found.');
elseif isscalar(files)
    selectedFile = files(1);
else
    % Multiple files found, display them and ask user to select
    fprintf('Multiple matching files found:\n');
    for i = 1:length(files)
        fprintf('%d: %s\n', i, files(i).name);
    end
    
    % Ask user to select a file
    while true
        selection = input('Enter the number of the file you want to load: ');
        if isnumeric(selection) && selection > 0 && selection <= length(files)
            selectedFile = files(selection);
            break;
        else
            fprintf('Invalid selection. Please enter a number between 1 and %d.\n', length(files));
        end
    end
end

% Construct the full path to the selected file
% Use only selectedFile.name here, as selectedFile.folder already contains the full path
saveName = fullfile(selectedFile.folder, selectedFile.name);

% Load checkpoint matfile object
checkpoint = matfile(saveName);

fprintf('Using data in file: %s\n', saveName);

% Clear variables except those needed
clearvars -except checkpoint soiHiFreq soiLoFreq fadeLen nTrainingPairs numHashFunctions numBands Fs n2n_dataset_root n2n_dataset_inputs n2n_dataset_targets

%% Preprocess audio 
% The "preprocessedAudio" is used only to pair dissimilar signals.

% Get the size of the detectionAudio
nSignals = size(checkpoint, 'detectionsAll');
measuresig = checkpoint.detectionsAll(1, :);
nSamps = size(measuresig.detectionAudio{1,1}, 1);

% Build a window function to fade-in and fade-out the signals
windowSamps = 2 * fadeLen * Fs;
if mod(windowSamps, 2) == 0
    windowSamps = windowSamps + 1;
end
window = hann(windowSamps);
onesToAdd = nSamps - windowSamps;
windowFull = [window(1:floor(windowSamps/2+1)); ones(onesToAdd-1, 1); flipud(window(1:floor(windowSamps/2+1)))];

% Build a Band Stop filter to remove the signal of interest
Wn(1) = soiLoFreq / (Fs / 2);
Wn(2) = soiHiFreq / (Fs / 2);
[b, a] = butter(8, Wn, 'stop');

% Preallocate cell array for preprocessedAudio
preprocessedAudio = cell(nSignals, 1);

for i = 1:nSignals
    audio = detectionsAll.detectionAudio{i,1};
    audio = audio ./ max(abs(audio));
    audio = audio .* windowFull;
    audio = audio - mean(audio);
    
    if any(isinf(audio)) || all(audio == 0) || any(isnan(audio))
        preprocessedAudio{i} = NaN(size(audio));
    else
        preprocessedAudio{i} = filtfilt(b, a, audio);
    end
end

% Remove empty/NaN cells (Failed preprocessing)
validIndices = ~cellfun(@(x) isempty(x) || any(isnan(x)), preprocessedAudio);
preprocessedAudio = preprocessedAudio(validIndices);

%% Generate Uncorrelated Signal Pairs

% Use Local Similarity Hashing in the Wavelet Domain to group dissimilar
% signals:
[signalPairs, signalPairIndices] = waveletLSH(...
    cell2mat(preprocessedAudio'), nTrainingPairs, ...
    numHashFunctions, numBands);

nPairs = length(signalPairs);
similarity = zeros(nPairs, 1);
rawMetrics = zeros(nPairs, 4);
for i = 1:nPairs
    % Get measure of similarity for each pair
    [similarity(i), rawMetrics(i,:)] = signalSimilarity(signalPairs{i, 1}, signalPairs{i, 2}, Fs);
end

% Sort similarity from least to most similar
[~, sortedSimilarityIndices] = sort(similarity, "ascend");

% Save out signals
for i = 1:nTrainingPairs
    % Retrieve this pair's audio data
    targetIndex = signalPairIndices(sortedSimilarityIndices(i), 1);
    inputIndex = signalPairIndices(sortedSimilarityIndices(i), 2);
    
    target = detectionsAll.detectionAudio{targetIndex, 1};
    input = detectionsAll.detectionAudio{inputIndex, 1};
    
    % Define this pair's file paths and names
    filename_target = fullfile(n2n_dataset_targets, ['train_target_', num2str(i), '.wav']);
    filename_input = fullfile(n2n_dataset_inputs, ['train_input_', num2str(i), '.wav']);
    
    % Save out wav files
    audiowrite(filename_target, target, Fs);
    audiowrite(filename_input, input, Fs);
end