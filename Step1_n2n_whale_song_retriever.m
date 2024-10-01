%% Step1_n2n_whale_song_retriever.m - INTERRUPTABLE VERSION
%
% DESCRIPTION:
%   This script is the first step in building a noise-to-noise (N2N) training
%   dataset for audio processing. It retrieves whale song detections from
%   continuous audio data, trims them to the region of interest, and saves
%   them as individual WAV files.
% 
%   This script uses a checkpoint system, allowing it to be interrupted 
%   and resumed from the last completed step. The checkpoint file is 
%   automatically managed by the script. Simply run it again if interrupted. 
%   Script operations are logged in a text file that will be saved to the 
%   current MATLAB directory.
%
% KEY FEATURES:
%   1. Loads and processes detection data from MAT files
%   2. Filters detections based on time separation and signal quality
%   3. Retrieves audio segments for each valid detection
%   4. Supports both serial and parallel processing modes
%   5. Implements a checkpoint system for interruptible execution
%   6. Handles audio segments that span multiple files
%   7. Saves processed audio segments and metadata
%
% DEPENDENCIES:
%   - MATLAB Parallel Computing Toolbox (for parallel processing mode)
%   - Custom functions: findROI_wavWriter, find_closest_wav, makeFilenameSafe
%
% SCRIPT WORKFLOW:
%   1. Initialization and Configuration
%      - Loads project configuration from config.m
%      - Sets up logging and determines operating environment
%
%   2. Detection Data Processing
%      - Loads detection data from MAT files
%      - Filters and sorts detections based on time and signal quality
%
%   3. Audio Segment Retrieval and Processing
%      - Retrieves audio segments for each valid detection
%      - Trims audio to the region of interest
%      - Handles detections that span multiple audio files
%
%   4. Audio Saving and Metadata Generation
%      - Saves processed audio segments as individual WAV files
%      - Generates and saves metadata for the processed detections
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
clear
close all
clc

% Begin logging
ts = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
logname = ['step1_script_log_', ts, '.txt'];
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

%% Add Paths

[gitRoot, ~, ~] = fileparts(here);
utilPath = fullfile(gitRoot, 'Utilities');
addpath(utilPath);

%% Load and process detections

detectionsFiles = dir(fullfile(detectionsPath, '*.mat'));

% Calculate expected number of samples
expected_num_samples = (call_duration + (buffer_duration*2)) * Fs + 1;

% Preallocate cell array for faster concatenation
tempTables = cell(length(detectionsFiles), 1);

for i = 1:length(detectionsFiles)
    data = load(fullfile(detectionsPath, detectionsFiles(i).name), 'detections');
    tempTables{i} = array2table(data.detections, 'VariableNames', ...
        {'Year', 'JulianDay', 'Month', 'Week', 'Time', 'SNR', 'SINR', 'SNRClass'});
end

% Concatenate all tables at once
detectionsAll = vertcat(tempTables{:});

% Make sure we have some detections
assert(~isempty(detectionsAll), ['No detections found.', newline, ...
    'Check operating environment setting.', newline ...
    'Check file paths in "config.m".'])

% Remove invalid detections
validDetection = ~(isnan(detectionsAll.SNR) | isnan(detectionsAll.SINR) | ...
    isinf(detectionsAll.SNR) | isinf(detectionsAll.SINR));

detectionsAll = detectionsAll(validDetection, :);

% Convert serial time to datetime and sort
detectionsAll.datetime_Readable = datetime(detectionsAll.Time, 'ConvertFrom', 'datenum');
detectionsAll.serial_datetime_time = (detectionsAll.Time - datenum('1970-01-01')) * 86400;
[~, sortIdx] = sort(detectionsAll.serial_datetime_time);
detectionsAll = detectionsAll(sortIdx, :);

% Filter based on time separation
minTimeDiff = minutesSeparation * 60;  % Convert minutes to seconds
validDetections = false(height(detectionsAll), 1);
validDetections(1) = true;
lastValidTime = detectionsAll.serial_datetime_time(1);

for i = 2:height(detectionsAll)
    if detectionsAll.serial_datetime_time(i) - lastValidTime >= minTimeDiff
        validDetections(i) = true;
        lastValidTime = detectionsAll.serial_datetime_time(i);
    end
end

detectionsAll = detectionsAll(validDetections, :);

%% Process audio segments and save as separate WAV files

switch mode
    case 'serial'
        % Pre-allocate arrays
        nDetections = height(detectionsAll);
        wav_subdir_paths = strings(nDetections, 1);
        detection_datestrings = strings(nDetections, 1);
        detectionsAll.audioFilename = strings(nDetections, 1);

        % Pre-compute common values
        for i = 1:nDetections
            % wav_subdir_paths(i) = fullfile(rawAudioPath, strcat(wav_subdir_prefix, num2str(detectionsAll.Year(i))), 'wav/');
            wav_subdir_paths(i) = fullfile(rawAudioPath, [wav_subdir_prefix, num2str(detectionsAll.Year(i))], 'wav/');
            detection_datestrings(i) = datestr(detectionsAll.Time(i), wav_dateformat_serial);
        end

        % Cache directory listings outside the loop
        wav_files_cache = containers.Map();

        for i = 1:nDetections
            if ~isKey(wav_files_cache, wav_subdir_paths(i))
                wav_files_cache(wav_subdir_paths(i)) = dir(fullfile(wav_subdir_paths(i), '*.wav'));
            end
        end

        % Find the largest iteration number from existing files
        start_iteration = findLatestFile(isolated_detections_wav_path) + 1;

        disp(['Starting from iteration: ', num2str(start_iteration)]);

        for i = start_iteration:nDetections
            wavs_filelist = wav_files_cache(wav_subdir_paths(i));  % Use the cached list
            wav_filename = find_closest_wav(wavs_filelist, char(detection_datestrings(i)));
            wav_fileName= fullfile(wav_subdir_paths(i), wav_filename);

            % Retrieve audio file, trim to region of interest, save trimmed file:
            [successFlag, detectionsAll.audioFilename(i)] = findROI_wavWriter(...
                wav_fileName, wavs_filelist, isolated_detections_wav_path, detection_datestrings(i), ...
                i, detectionsAll.datetime_Readable(i), detectionsAll.SNR(i), ...
                expected_num_samples, call_duration, buffer_duration);
        end

        % Remove rows where audio wasn't saved
        detectionsAll = detectionsAll(detectionsAll.audioFilename ~= "", :);      
        
    case 'parallel'
        % Pre-allocate arrays
        nDetections = height(detectionsAll);
        wav_subdir_paths = strings(nDetections, 1);
        detection_datestrings = strings(nDetections, 1);
        detectionsAll.audioFilename = strings(nDetections, 1);
        
        % Extract necessary data from detectionsAll
        datetime_readable = detectionsAll.datetime_Readable;
        SNR = detectionsAll.SNR;
        
        % Pre-compute common values
        for i = 1:nDetections
            wav_subdir_paths(i) = fullfile(rawAudioPath, [wav_subdir_prefix, num2str(detectionsAll.Year(i))], 'wav/');
            detection_datestrings(i) = string(datestr(detectionsAll.Time(i), wav_dateformat_serial));
        end
        
        % Get unique paths and map detections to them
        [unique_paths, ~, idx_paths] = unique(wav_subdir_paths);
        
        % Cache directory listings for unique paths
        wav_files_per_path = cell(length(unique_paths), 1);
        for k = 1:length(unique_paths)
            wav_files_per_path{k} = dir(fullfile(unique_paths(k), '*.wav'));
        end
        
        % Map detections to their corresponding file lists
        wav_files_list = wav_files_per_path(idx_paths);
        
        % Find the largest iteration number from existing files
        start_iteration = findLatestFile(isolated_detections_wav_path) + 1;
        
        disp(['Starting from iteration: ', num2str(start_iteration)]);
        
        % Pre-allocate audioFilename array
        audioFilename = strings(nDetections, 1);

        % Number of detections to process
        numFutures = nDetections - start_iteration + 1;
        
        % Initialize futures array
        futures = parallel.FevalFuture.empty(numFutures, 0);

        % Initialize error logging
        errorLog = cell(numFutures, 1);

        tic
        ticBytes(gcp);

        % Submit parfeval requests
        for idx = 1:numFutures
            i = idx + start_iteration - 1;
            futures(idx) = parfeval(@processDetectionWithErrorHandling, 2, i, wav_files_list{i}, detection_datestrings(i), ...
                wav_subdir_paths(i), datetime_readable(i), SNR(i), expected_num_samples, ...
                call_duration, buffer_duration, isolated_detections_wav_path);
        end

        % Collect the results
        for idx = 1:numFutures
            try
                [completedIdx, audioFilename_i, error_info] = fetchNext(futures);
                i = completedIdx + start_iteration - 1;
                audioFilename(i) = audioFilename_i;
                
                if ~isempty(error_info)
                    errorLog{idx} = error_info;
                    fprintf('Warning: Error in iteration %d: %s\n', i, error_info.message);
                end
                
                % Print progress
                if mod(idx, 100) == 0
                    fprintf('Processed %d/%d iterations\n', idx, numFutures);
                end
            catch ME
                fprintf('Error fetching result for iteration %d: %s\n', idx, ME.message);
                errorLog{idx} = struct('message', ME.message, 'stack', ME.stack);
            end
        end

        tocBytes(gcp)
        toc
     
        % Assign the audio filenames back to detectionsAll
        detectionsAll.audioFilename = audioFilename;
        
        % Remove rows where audio wasn't saved
        detectionsAll = detectionsAll(detectionsAll.audioFilename ~= "", :);

        % Print summary of errors
        errorCount = sum(~cellfun(@isempty, errorLog));
        fprintf('Total errors encountered: %d\n', errorCount);
        
        if errorCount > 0
            fprintf('Error summary:\n');
            for idx = 1:length(errorLog)
                if ~isempty(errorLog{idx})
                    fprintf('Iteration %d: %s\n', idx + start_iteration - 1, errorLog{idx}.message);
                end
            end
        end
end

%% Save Metadata

nDetections = height(detectionsAll);
now = char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-SS'));
saveName = fullfile(isolated_detections_wav_path, 'isolated_detections_metadata.mat');

% Check if the file exists
if exist(saveName, 'file') == 2

    % File exists, load existing data
    existingData = load(saveName);

    % Append new data to existing data
    if isfield(existingData, 'detectionsAll')
        combinedDetections = [existingData.detectionsAll; detectionsAll];
    else
        combinedDetections = detectionsAll;
    end

    % Save combined data
    save(saveName, 'combinedDetections', '-v7.3');
    disp(['Appended metadata to existing file: ' saveName]);
else

    % File doesn't exist, create new file
    save(saveName, 'detectionsAll', '-v7.3');
    disp(['Created new metadata file: ' saveName]);
end

% End logging
diary off

%% Helper Functions

% Helper function for error handling
function [audioFilename, error_info] = processDetectionWithErrorHandling(varargin)
    error_info = [];
    try
        audioFilename = processDetection(varargin{:});
    catch ME
        audioFilename = "";
        error_info = struct('message', ME.message, 'stack', ME.stack);
    end
end

function latestFile = findLatestFile(folderPath)
    % Uses the iteration number in the wav filenames to get the index of the
    % last detection successfully saved to disk.
    
    % Get all WAV files in the folder
    files = dir(fullfile(folderPath, 'detectionAudio_*.wav'));
    
    % If no files found, return 0
    if isempty(files)
        latestFile = 0;
        return;
    end
    
    % Extract iteration numbers from filenames
    iterations = zeros(length(files), 1);
    for i = 1:length(files)
        % Extract the number after 'detectionAudio_'
        parts = strsplit(files(i).name, '_');
        iterations(i) = str2double(parts{2});
    end
    
    % The largest iteration number is the latest retrieved file
    latestFile = max(iterations);
end

function audioFilename_i = processDetection(i, wavs_filelist, detection_datestring, ...
    wav_subdir_path, datetime_readable_i, SNR_i, expected_num_samples, ...
    call_duration, buffer_duration, isolated_detections_wav_path)
    % Processes a single detection and returns the audio filename.

    % Find the closest WAV file
    wav_filename = find_closest_wav(wavs_filelist, char(detection_datestring));
    wav_fileName = fullfile(wav_subdir_path, wav_filename);

    % Retrieve audio file, trim to region of interest, save trimmed file
    [successFlag, audioFilename_i] = findROI_wavWriter(...
        wav_fileName, wavs_filelist, isolated_detections_wav_path, detection_datestring, ...
        i, datetime_readable_i, SNR_i, ...
        expected_num_samples, call_duration, buffer_duration);

    if ~successFlag
        audioFilename_i = "";  % Return empty string if failed
    end
end
