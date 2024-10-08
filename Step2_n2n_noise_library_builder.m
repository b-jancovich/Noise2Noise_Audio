%% Step2_n2n_noise_library_builder.m - INTERRUPTABLE VERSION
%
% DESCRIPTION:
%   This script is the second step in building a noise-to-noise (N2N) training
%   dataset for audio processing. It extracts subsequences of audio that do
%   not contain the signal of interest (whale songs) from continuous audio data.
%   These "noise-only" segments are then saved as individual WAV files to create
%   a noise library.
%
%   This script uses a checkpoint system, allowing it to be interrupted
%   and resumed from the last completed step. The checkpoint file is
%   automatically managed by the script. Simply run it again if interrupted.
%   Script operations are logged in a text file that will be saved to the
%   current MATLAB directory.
%
% KEY FEATURES:
%   1. Loads and processes detection data from MAT files
%   2. Identifies periods of time without whale songs based on detection data
%   3. Extracts audio segments for noise-only periods
%   4. Supports both serial and parallel processing modes
%   5. Implements a checkpoint system for interruptible execution
%   6. Handles audio segments that span multiple files
%   7. Saves processed noise-only audio segments
%   8. Generates spectrograms for random samples of the noise library
%
% DEPENDENCIES:
%   - MATLAB Parallel Computing Toolbox (for parallel processing mode)
%   - Custom functions: assembleROIAudio, find_closest_wav
%
% SCRIPT WORKFLOW:
%   1. Initialization and Configuration
%      - Loads project configuration from config.m
%      - Sets up logging and determines operating environment
%
%   2. Detection Data Processing
%      - Loads detection data from MAT files
%      - Filters detections based on time separation and signal quality
%      - Identifies periods of time without whale songs
%
%   3. Noise Audio Segment Retrieval and Processing
%      - Retrieves audio segments for each identified noise-only period
%      - Handles audio segments that span multiple files
%
%   4. Noise Library Creation
%      - Saves processed noise-only audio segments as individual WAV files
%      - Generates spectrograms for a random sample of noise segments
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

% Begin logging
ts = char(datetime("now", "Format", "dd-MMM-uuuu_HH-mm-ss"));
logname = ['step2_script_log_', ts, '.txt'];
diary(logname)

%% Set Operating Environment
% 1 = Use the paths in config.m that relate to my windows laptop
% 2 = Use the paths in config.m that relate to the Katana Cluster
opEnv = 1;

% Compute file retrievals in serial or parallel:
mode = 'parallel'; % or 'serial'
% Note: parallel requires MATLAB parallel computing toolbox

%% Load project configuration file

here = pwd;
run(fullfile(here, 'config.m'));
disp('Loaded N2N Config file.')

%% Add Paths

[gitRoot, ~, ~] = fileparts(here);
utilPath = fullfile(gitRoot, 'Utilities');
addpath(utilPath);

%% Load detections, Clean Up and Build serial_datenum Dates

detectionsFiles = dir(fullfile(detectionsPath, '*.mat'));

if isempty(detectionsFiles)
    error('No detection files found - Check "detectionsPath".')
end

% Calculate expected number of samples
expected_num_samples = (call_duration + (buffer_duration*2)) * Fs + 1;

% Preallocate cell array for faster concatenation
tempTables = cell(length(detectionsFiles), 1);

for i = 1:length(detectionsFiles)
    data = load(fullfile(detectionsPath, detectionsFiles(i).name), 'detections');
    tempTables{i} = array2table(data.detections, 'VariableNames', ...
        {'Year', 'JulianDay', 'Month', 'Week', 'serial_datenum', 'SNR', 'SINR', 'SNRClass'});
    disp(['Loaded Detections File: ', detectionsFiles(i).name])
end

% Concatenate all tables at once
detectionsAll = vertcat(tempTables{:});

% Remove invalid detections (missing SNR, SINR)
validDetection = ~(isnan(detectionsAll.SNR) | isnan(detectionsAll.SINR) | ...
    isinf(detectionsAll.SNR) | isinf(detectionsAll.SINR));
detectionsAll = detectionsAll(validDetection, :);

% DetectionsAll field "Time" is in MATLAB's serial "datenum" format
% (fractional days since January 01, 0000)
% Convert serial time to serial_datenum Time and sort detections
detectionsAll.datetime_Readable = datetime(detectionsAll.serial_datenum, 'ConvertFrom', 'datenum');

%% Filter list of detections by defining a minimim time separtation

% Calculate the minimum separation between detection timestamps to qualify
% a period of time as "song-free".
maxSongLengthSeconds = (call_duration * safetyFactor) + (buffer_duration * safetyFactor);
maxSongLengthDays = maxSongLengthSeconds / 86400;
minimumSongSeparationSeconds = maxSongLengthSeconds + interCallInterval * safetyFactor;
minimumSongSeparationDays = minimumSongSeparationSeconds / 86400;
minimumSongSeparationSamps = minimumSongSeparationSeconds * Fs;

% Sort detectionsAll by serial_datenum
detectionsAll = sortrows(detectionsAll, 'serial_datenum');

% Calculate time differences between consecutive rows (in seconds)
timeDiffsSeconds = diff(detectionsAll.serial_datenum) * 86400;

% Find indices where the time difference is greater than or equal to
% minimumSongSeparationSeonds
validIndices = find(timeDiffsSeconds >= minimumSongSeparationSeconds);

% Create a logical array for filtering
filterMask = false(height(detectionsAll), 1);
filterMask(validIndices) = true;
filterMask(validIndices + 1) = true;

% Apply the filter to detectionsAll
filteredDetections = detectionsAll(filterMask, :);

% Display the results
disp(['Number of raw detections: ', num2str(height(detectionsAll))]);
disp('Filtering detections by temporal separation...');

% Clear unused vars
clearvars detectionsAll detectionsFiles data tempTables

%% Get time indices of song-free periods

noiseLibrary = struct("Year", [], "startTime_serial_datenum", [], "endTime_serial_datenum", [], ...
    "separation2Next_Minutes", []);
nNoiseOnlySequences = height(filteredDetections);
nIdx = 1;
for i = 1:nNoiseOnlySequences-1
    % Get current and next detections's start and end times
    currentDetectionStart = filteredDetections{i,"serial_datenum"};
    currentDetectionEnd = currentDetectionStart + maxSongLengthDays;
    nextDetectionStart = filteredDetections{i+1,"serial_datenum"} - maxSongLengthDays;

    % Calculate duration between end of this detection and start of
    % next detection (days):
    currentToNextSeparation_days = nextDetectionStart - currentDetectionEnd;

    % If separation between detections is big enough, record this
    % time period in the library as a noise sample:
    if currentToNextSeparation_days > minimumSongSeparationDays

        % Record the year of the noise sample
        noiseLibrary(nIdx).Year = filteredDetections{i, "Year"};

        % Record the start timestamp of noise-only period
        noiseLibrary(nIdx).startTime_serial_datenum = currentDetectionEnd;

        % Record the time separation
        noiseLibrary(nIdx).separation2Next_Minutes = currentToNextSeparation_days/60;

        % Record the End timestamp of noise-only period
        noiseLibrary(nIdx).endTime_serial_datenum = nextDetectionStart;

        % Increment counter
        nIdx = nIdx + 1;
    end
end

disp(['Number of song-free time periods identified: ', num2str(nIdx)]);

%% Get the Audio corresponing to these song-free periods

% Count Detections
nNoiseOnlySequences = length(noiseLibrary);

% Pre-compute paths and start time strings
for i = 1:nNoiseOnlySequences
    % Get wav subdirectory paths
    noiseLibrary(i).wavSubDirPath = fullfile(rawAudioPath, [wav_subdir_prefix, num2str(noiseLibrary(i).Year)], 'wav/');

    % Format the datenum as a string
    noiseLibrary(i).startTimeDatestrings = datestr(noiseLibrary(i).startTime_serial_datenum, wav_dateformat_serial);
end

% Cache directory listings outside the main audio retrieval loop
wav_files_cache = containers.Map();

for i = 1:nNoiseOnlySequences
    if ~isKey(wav_files_cache, noiseLibrary(i).wavSubDirPath)
        wav_files_cache(noiseLibrary(i).wavSubDirPath) = dir(fullfile(noiseLibrary(i).wavSubDirPath, '*.wav'));
    end
end

switch mode
    case 'serial'
        % Find the latest processed noise file
        latestdatenum = findLatestNoiseFile(noise_lib_path);

        % Get then wavs and return the ROI
        for i = 1:nNoiseOnlySequences
            % Skip already processed files
            if ~isempty(latestdatenum) && datenum(noiseLibrary(i).startTimeDatestrings, 'yymmdd-HHMMSS') <= latestdatenum
                continue;
            end

            wavs_filelist = wav_files_cache(noiseLibrary(i).wavSubDirPath);  % Use the cached list
            wav_filename = find_closest_wav(wavs_filelist, char(noiseLibrary(i).startTimeDatestrings));

            if isempty(wavs_filelist)
                error('No wav files found - Check wav file paths and that storage volume is mounted.')
            end

            % Retrieve audio file, trim/append to region of interest, write to struct:
            [audioData, ~, successFlag] = assembleROIAudio(...
                wavs_filelist, noiseLibrary(i).wavSubDirPath, noiseLibrary(i).startTime_serial_datenum, noiseLibrary(i).endTime_serial_datenum);

            if successFlag == true
                fileName = ['DGS_noise_', noiseLibrary(i).startTimeDatestrings, '.wav'];
                fullNamePath = fullfile(noise_lib_path, fileName);
                audiowrite(fullNamePath, audioData, Fs);
            end
        end

    case 'parallel'
        % Find the latest processed noise file
        latestdatenum = findLatestNoiseFile(noise_lib_path);

        % Load checkpoint if it exists
        checkpointFile = 'noise_library_checkpoint.mat';
        if exist(checkpointFile, 'file')
            load(checkpointFile, 'lastProcessedIndex');
            disp(['Resuming from checkpoint: ' num2str(lastProcessedIndex)]);
        else
            lastProcessedIndex = 0;
        end

        % Number of noise sequences to process
        numFutures = nNoiseOnlySequences;

        % Initialize error logging
        errorLog = cell(numFutures, 1);

        % Process in batches
        batchSize = 500;
        batchNum = 1;
        for batchStart = lastProcessedIndex+1:batchSize:numFutures
            batchEnd = min(batchStart + batchSize - 1, numFutures);

            % Initialize futures array for this batch
            futures = parallel.FevalFuture.empty(batchEnd - batchStart + 1, 0);

            % Submit parfeval requests for this batch
            for i = batchStart:batchEnd
                % Skip already processed files
                if ~isempty(latestdatenum) && datenum(noiseLibrary(i).startTimeDatestrings, 'yymmdd-HHMMSS') <= latestdatenum
                    continue;
                end

                wavs_filelist = wav_files_cache(noiseLibrary(i).wavSubDirPath);
                futures(i-batchStart+1) = parfeval(@processNoiseSequence, 2, i, wavs_filelist, ...
                    noiseLibrary(i).wavSubDirPath, noiseLibrary(i).startTime_serial_datenum, ...
                    noiseLibrary(i).endTime_serial_datenum, noiseLibrary(i).startTimeDatestrings, ...
                    noise_lib_path, Fs);
            end

            % Collect the results for this batch
            for i = 1:length(futures)
                try
                    [completedIdx, successFlag, error_info] = fetchNext(futures);

                    if ~isempty(error_info)
                        errorLog{completedIdx} = error_info;
                        fprintf('Warning: Error in sequence %d: %s\n', completedIdx, error_info.message);
                    end

                    % Print progress
                    if mod(completedIdx, 50) == 0
                        fprintf('Processed %d noise sequences of batch %d. Batch size %d.\n', completedIdx, batchNum, batchSize);
                    end

                    % Save checkpoint every 500 processed items
                    if mod(completedIdx, 500) == 0
                        lastProcessedIndex = completedIdx;
                        save(checkpointFile, 'lastProcessedIndex');
                    end
                catch ME
                    fprintf('Error fetching result for sequence %d: %s\n', batchStart+i-1, ME.message);
                    errorLog{batchStart+i-1} = struct('message', ME.message, 'stack', ME.stack);
                end
            end

            % Clear futures to free up memory
            clear futures;
            fprintf('Completed batch %d.\n', batchNum)
            batchNum = batchNum + 1;
        end

        % Print summary of errors
        errorCount = sum(~cellfun(@isempty, errorLog));
        fprintf('Total errors encountered: %d\n', errorCount);

        if errorCount > 0
            fprintf('Error summary:\n');
            for i = 1:length(errorLog)
                if ~isempty(errorLog{i})
                    fprintf('Sequence %d: %s\n', i, errorLog{i}.message);
                end
            end
        end

        % Remove checkpoint file after successful completion
        if exist(checkpointFile, 'file')
            delete(checkpointFile);
        end
end
disp('Noise library construction complete.')

switch mode
    case 'parallel'
        disp('Shutting down parallel Pool. End Process.')
        
        % Shut down parallel pool
        delete(gcp)
    case 'serial'
        disp('End Process.')
end

% End logging
diary off

%% Helper Functions

function [audioData, Fs, successFlag] = assembleROIAudio(wavs_filelist, read_folder, ROI_start_serial_datenum, ROI_end_serial_datenum)
% Assembles audio data for a given region of interest (ROI) from multiple WAV files
% NOTE: IT IS ASSUMED ALL FILES ARE THE SAME BIT DEPTH & SAMPLE RATE

% Inputs:
%   wavs_filelist: struct array containing information about WAV files
%   read_folder: string, path to the folder containing WAV files
%   ROI_start_serial_datenum: double, start time of ROI in serial_datenum format
%   ROI_end_serial_datenum: double, end time of ROI in serial_datenum format
%
% Outputs:
%   audioData: vector of doubles, assembled audio data for the ROI
%   Fs: double, sampling frequency of the audio data
%   successFlag: boolean, indicates whether the operation was successful

% Initialize variables
successFlag = true;
audioData = [];
wav_datestring_format = 'yyMMdd-HHmmss';
max_attempts = 2;
retry_delay = 2; % seconds  
% Initialize variables for size tracking
MAX_FILE_SIZE_GB = 0.5;
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 1e+9;  % 2GB in bytes
current_size_bytes = 0;

% Convert datenum times to datenum
ROI_start_datetime = datetime(ROI_start_serial_datenum, 'ConvertFrom', 'datenum');
ROI_end_datetime = datetime(ROI_end_serial_datenum, 'ConvertFrom', 'datenum');

% Select the first file to get attributes
testAttributesFile = fullfile(wavs_filelist(1).folder, wavs_filelist(1).name);
% Read audio sample rate
[~, Fs] = audioread(testAttributesFile);
% Read wav file header
fid = fopen(testAttributesFile, 'r');
% Seek to the position where bit depth is stored (position 35-36)
fseek(fid, 34, 'bof');
% Read 2 bytes for bit depth
bit_depth = fread(fid, 1, 'uint16');
% Close the file
fclose(fid);

% Find relevant WAV files
relevant_files = [];
for i = 1:length(wavs_filelist)
    % Get file start time:
    file_start_str = extractAfter(wavs_filelist(i).name, "H08S1_");
    file_start_datetime = datetime(file_start_str(1:end-4), 'InputFormat', wav_datestring_format);

    % Calculate the number of bytes per sample
    bytes_per_sample = bit_depth / 8;

    % Calculate the number of samples (subtract 44 bytes for WAV header)
    num_samples = (wavs_filelist(i).bytes - 44) / bytes_per_sample;

    % Calculate the file duration in seconds
    file_duration = num_samples / Fs;

    % Calculate file end time
    file_end_datetime = file_start_datetime + seconds(file_duration);

    % If the file start and end are within the ROI start and end, mark as
    % relevant:
    if (file_start_datetime <= ROI_end_datetime && file_end_datetime >= ROI_start_datetime)
        relevant_files = [relevant_files; wavs_filelist(i)];
    end
end

% Sort relevant files by start time
[~, sort_idx] = sort({relevant_files.name});
relevant_files = relevant_files(sort_idx);

% Read and assemble audio data
for i = 1:length(relevant_files)
    file_path = fullfile(read_folder, relevant_files(i).name);

    for attempt = 1:max_attempts
        try
            [audio, Fs] = audioread(file_path);
            break;
        catch
            if attempt < max_attempts
                warning('Failed to read audio file %s. Retrying in %d seconds... (Attempt %d of %d)', ...
                    file_path, retry_delay, attempt, max_attempts);
                pause(retry_delay);
            else
                warning('Failed to read audio file %s after %d attempts. Skipping.', ...
                    file_path, max_attempts);
                successFlag = false;
                return;
            end
        end
    end

    % Determine start and end samples for this file
    file_start_str = extractAfter(relevant_files(i).name, "H08S1_");
    file_start_datetime = datetime(file_start_str(1:end-4), 'InputFormat', wav_datestring_format);
    file_duration = seconds(length(audio) / Fs);
    file_end_datetime = file_start_datetime + file_duration;

    % Calculate start and end samples as integers
    start_sample = max(1, round(seconds(ROI_start_datetime - file_start_datetime) * Fs) + 1);
    end_sample = min(length(audio), round(seconds(ROI_end_datetime - file_start_datetime) * Fs) + 1);

    % Calculate the size of the audio segment in bytes
    segment_size_bytes = (end_sample - start_sample + 1) * bytes_per_sample;

    % Check if adding this segment would exceed the size limit
    if current_size_bytes + segment_size_bytes > MAX_FILE_SIZE_BYTES
        warning('Reached 2GB limit. Truncating audio data.');
        break;
    end

    % Append relevant audio data
    if start_sample <= length(audio) && end_sample >= 1
        segment = audio(start_sample:end_sample);
        audioData = [audioData; segment];
        current_size_bytes = current_size_bytes + segment_size_bytes;
    end

    % Check if we've reached the ROI end
    if file_end_datetime >= ROI_end_datetime
        break;
    end

end

% Check if we have assembled any audio data
if isempty(audioData)
    warning('No audio data found for the specified ROI.');
    successFlag = false;
    return;
end

successFlag = true;

% Remove DC offset
audioData = audioData - mean(audioData);

end

function [successFlag, error_info] = processNoiseSequence(i, wavs_filelist, ...
    wavSubDirPath, startTime_serial_datenum, endTime_serial_datenum, startTimeDatestrings, ...
    noise_lib_path, Fs)

error_info = [];

try
    if isempty(wavs_filelist)
        error('No wav files found - Check wav file paths and that storage volume is mounted.')
    end

    % Retrieve audio file, trim/append to region of interest
    [audioData, ~, retrievalSuccessFlag] = assembleROIAudio(...
        wavs_filelist, wavSubDirPath, startTime_serial_datenum, endTime_serial_datenum);

    if retrievalSuccessFlag
        fileName = ['DGS_noise_', startTimeDatestrings, '.wav'];
        fullNamePath = fullfile(noise_lib_path, fileName);
        audiowrite(fullNamePath, audioData, Fs);
        successFlag = true;
        fprintf('Wrote File %d to disk at %s.\n', i, fullNamePath)
    else
        error('Failed to retrieve audio data for noise sequence %d', i);
    end
catch ME
    successFlag = false;
    error_info = struct('message', ME.message, 'stack', ME.stack);
end
end

function latestDateTime = findLatestNoiseFile(folderPath)
% Find the latest processed noise file based on the date-time in the filename

% Get all WAV files in the folder
files = dir(fullfile(folderPath, 'DGS_noise_*.wav'));

% If no files found, return empty
if isempty(files)
    latestDateTime = [];
    return;
end

% Extract date-times from filenames
dateTimes = zeros(length(files), 1);
for i = 1:length(files)
    % Extract the date-time after 'DGS_noise_'
    parts = strsplit(files(i).name, '_');
    dateTimes(i) = datenum(parts{3}(1:end-4), 'yyMMdd-HHmmSS');
end

% The largest date-time is the latest processed file
latestDateTime = max(dateTimes);
end