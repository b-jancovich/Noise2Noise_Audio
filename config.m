% Noise2Noise Master Configuration File

%% Step 1 - Whale Song Retriever 

% Init Parameters
minutesSeparation = 10; % Minimum separation between detections (minutes)
call_duration = 35;     % Duration of the target whale call (seconds)
buffer_duration = 3;    % Time to add to the start and end of each detection (seconds)
Fs = 250;               % Sampling frequency of recordings (Hz)
wav_dateformat_serial = 'yymmdd-HHMMSS';

% Paths
% Path to complete CTBTO dataset:
rawAudioPath = "D:\Diego Garcia South";
% Path to detection MAT Files:
detectionsPath = "C:\Users\z5439673\OneDrive - UNSW\H0419778\Manue_Chagos_RawData\DGS";
% String for CTBT Location Folder:
wav_subdir_prefix = 'DiegoGarcia';
% Output path for isolated detection wavs
isolated_detections_wav_path = 'D:\Isolated_Detection_Wavs';

%% Step 2 - Noise Library Builder

% Init Parameters
interCallInterval = 195; % Duration of silence between song repetitions (seconds)
safetyFactor = 6; % Safety factor in calculating time separation between detections

% Paths
noise_lib_path = "D:\DGS_noise_library";

%% Step 3 - n2n_trainset_testset_builder

% Init Parameters
% Dissimilarity Matching Variables
soiHiFreq = 60; % the highest freq in the signal of interest (Hz)
soiLoFreq = 25; % the lowest freq in the signal of interest (Hz)
fadeLen = 0.5; % Duration of fades at the start/end of audio file (s)
nTrainingPairs = 15000; % Number of input-target pairs to build for training
miniBatchSize = 5000; % Number of files to process in each LSH Batch 
%                     (lower this if out-of-memory errors occur in LDHPairs)

% Audio Feature Extraction variables
p = 95; % Amount of variance in signals to be explained by selected features (%)
overlapPercent = 75; % Overlap between analysis windows (%)
nFFT = 1024; % Number of FFT points
sampleSize = 1500; % Number of audio files to randomly sample for feature selection.
stationarityThreshold = 0.2; % The longest duration over which the signals  
%                           in the recordings can be considered to be "stationary" (s)

% NOTE: Window size is equal to floor(stationarity_threshold * Fs).

% Test Dataset Synthesis Variables
nTestingPairs = 2000; % Number of input-target pairs to build for testing
fadeLenNoiseless = 512; % Length of the fade to apply to the start & end of the clean recording (samples)
timeShiftRange = [-4, 20]; % Min and Max limits of random time shift in test signals (seconds)
snrRange = [-60, 40]; % Min and Max limits of random SNR in test signals (dB)
snrMean = -0.9; % The mean SNR of the test signals (dB)
initialFreq = 35; % Freq of chagos call fundamental (Hz)
shiftRate = 0.33; % Rate of change in blue whale song (Hz/year)
speedupFactorRange = [0.8, 1.2]; % Min/Max audio speedup factor range
lpfRange = [38, 50]; % Min/Max Cutoff Frequency Range for random LPF (Hz)
hpfRange = [10, 32]; % Min/Max Cutoff Frequency Range for random HPF (Hz)
SourceVelocityRange = [1, 50]; % Min/Max velocity of source for doppler shift (m/s)
c = 1500; % Approximate propagation velocity in water (m/s)

% Paths
% Output path for isolated detection wavs
isolated_detections_wav_path = 'D:\Isolated_Detection_Wavs';

noise_lib_path = "D:\DGS_noise_library";

% Path to noiseless copy of signal:
noiseless_detection_path = "C:\Users\z5439673\Git\localisation_and_source_level_est\2_DATA\H08S1_02-Aug-2015_23-26-43_Chagos_Song_SPECTRAL_DENOISE_RX.wav";

% Output data paths:
n2n_dataset_root = 'C:\Users\z5439673\OneDrive - UNSW\H0419778\Noise2Noise_Audio';
n2n_train_inputs = fullfile(n2n_dataset_root, 'train_inputs');
n2n_train_targets = fullfile(n2n_dataset_root, 'train_targets'); 
n2n_test_inputs = fullfile(n2n_dataset_root, 'test_inputs');
n2n_test_groundTruth = fullfile(n2n_dataset_root, 'test_groundtruth'); 

