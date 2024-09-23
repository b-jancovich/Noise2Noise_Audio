% Noise Library Builder
% Step 2 of n2n train-test dataset builder 
%
% Extracts subsequences of audio that do not contain the signal of interest
% Based on the original n2n_whale_song_dataset_builder_STEP1.m script
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia

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

%% Load detections, Clean Up and Build POSIX Dates

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
        {'Year', 'JulianDay', 'Month', 'Week', 'Time', 'SNR', 'SINR', 'SNRClass'});
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
% Convert serial time to Posix Time and sort detections
detectionsAll.datetime_Readable = datetime(detectionsAll.Time, 'ConvertFrom', 'datenum');
detectionsAll.posix_time = (detectionsAll.Time - datenum('1970-01-01')) * 86400;

%% Filter list of detections by defining a minimim time separtation

% Calculate the minimum separation between detection timestamps to qualify
% a period of time as "song-free". Posix time is in seconds.
maxSongLength = (call_duration * safetyFactor) + (buffer_duration * safetyFactor);
minimumSongSeparationPosix = maxSongLength + interCallInterval * safetyFactor; 
minimumSongSeparationSamps = minimumSongSeparationPosix * Fs; 

% Sort detectionsAll by posix_time
detectionsAll = sortrows(detectionsAll, 'posix_time');

% Calculate time differences between consecutive rows
timeDiffs = diff(detectionsAll.posix_time);

% Find indices where the time difference is greater than or equal to minimumSongSeparationPosix
validIndices = find(timeDiffs >= minimumSongSeparationPosix);

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

noiseLibrary = struct("Year", [], "startTimePosix", [], "endTimePosix", [], ...
    "separation2Next_Minutes", []);
nNoiseOnlySequences = height(filteredDetections);
nIdx = 1;
for i = 1:nNoiseOnlySequences-1
        % Get current and next detections's start and end times
        currentDetectionStart = filteredDetections{i,"posix_time"};
        currentDetectionEnd = currentDetectionStart + maxSongLength;
        nextDetectionStart = filteredDetections{i+1,"posix_time"} - maxSongLength;

        % Calculate seconds between end of this detection and start of next detection:
        currentToNextSeparationPosix = nextDetectionStart - currentDetectionEnd;

        % If separation between detections is big enough, record this 
        % time period in the library as a noise sample:
        if currentToNextSeparationPosix > minimumSongSeparationPosix

            % Record the year of the noise sample
            noiseLibrary(nIdx).Year = filteredDetections{i, "Year"};

            % Record the start timestamp of noise-only period
            noiseLibrary(nIdx).startTimePosix = currentDetectionEnd;

            % Record the time separation
            noiseLibrary(nIdx).separation2Next_Minutes = currentToNextSeparationPosix/60;

            % Record the End timestamp of noise-only period
            noiseLibrary(nIdx).endTimePosix = nextDetectionStart;

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
        
    % Convert start time of song-free period from POSIX time to MATLAB datenum
    datenum_time = noiseLibrary(i).startTimePosix / 86400 + datenum('1970-01-01');
    
    % Format the datenum as a string
    noiseLibrary(i).startTimeDatestrings = datestr(datenum_time, wav_dateformat_serial);
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
        latestDateTime = findLatestNoiseFile(noise_lib_path);
        
        % Start execution timing
        tic
        % Get then wavs and return the ROI
        for i = 1:nNoiseOnlySequences
            % Skip already processed files
            if ~isempty(latestDateTime) && datenum(noiseLibrary(i).startTimeDatestrings, 'yymmdd-HHMMSS') <= latestDateTime
                continue;
            end
        
            wavs_filelist = wav_files_cache(noiseLibrary(i).wavSubDirPath);  % Use the cached list
            wav_filename = find_closest_wav(wavs_filelist, char(noiseLibrary(i).startTimeDatestrings));
        
            if isempty(wavs_filelist)
                error('No wav files found - Check wav file paths and that storage volume is mounted.')
            end
        
            % Retrieve audio file, trim/append to region of interest, write to struct:
            [audioData, ~, successFlag] = assembleROIAudio(...
                wavs_filelist, noiseLibrary(i).wavSubDirPath, noiseLibrary(i).startTimePosix, noiseLibrary(i).endTimePosix);
        
            if successFlag == true
                fileName = ['DGS_noise_', noiseLibrary(i).startTimeDatestrings, '.wav'];
                fullNamePath = fullfile(noise_lib_path, fileName);
                audiowrite(fullNamePath, audioData, Fs);
            end
        end
        % Stop execution timing
        toc
    case 'parallel'
        % Find the latest processed noise file
        latestDateTime = findLatestNoiseFile(noise_lib_path);
        
        tic
        ticBytes(gcp);
        
        % Number of noise sequences to process
        numFutures = nNoiseOnlySequences;
        
        % Initialize futures array
        futures = parallel.FevalFuture.empty(numFutures, 0);
        
        % Initialize error logging
        errorLog = cell(numFutures, 1);
        
        % Submit parfeval requests
        for i = 1:numFutures
            % Skip already processed files
            if ~isempty(latestDateTime) && datenum(noiseLibrary(i).startTimeDatestrings, 'yymmdd-HHMMSS') <= latestDateTime
                continue;
            end
        
            wavs_filelist = wav_files_cache(noiseLibrary(i).wavSubDirPath);
            futures(i) = parfeval(@processNoiseSequence, 2, i, wavs_filelist, ...
                noiseLibrary(i).wavSubDirPath, noiseLibrary(i).startTimePosix, ...
                noiseLibrary(i).endTimePosix, noiseLibrary(i).startTimeDatestrings, ...
                noise_lib_path, Fs);
        end

        % Collect the results
        for i = 1:numFutures
            try
                [completedIdx, successFlag, error_info] = fetchNext(futures);
                
                if ~isempty(error_info)
                    errorLog{completedIdx} = error_info;
                    fprintf('Warning: Error in sequence %d: %s\n', completedIdx, error_info.message);
                end
                
                % Print progress
                if mod(i, 100) == 0
                    fprintf('Processed %d/%d noise sequences\n', i, numFutures);
                end
            catch ME
                fprintf('Error fetching result for sequence %d: %s\n', i, ME.message);
                errorLog{i} = struct('message', ME.message, 'stack', ME.stack);
            end
        end

        tocBytes(gcp)
        toc

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
       
end

% Randomly select 10 rows from noiseLibrary and plot the spectrogram of the audioData
nToPlot = 10;
sampleIndices = randperm(length(noiseLibrary), nToPlot);

for i = 1:nToPlot
    idx = sampleIndices(i);
    audio = noiseLibrary(idx).audioData;
    
    figure(i)
    spectrogram(audio, 256, floor(256*0.9), 1024, Fs, 'yaxis');
    title(sprintf('Sample %d (Year: %d)', i, noiseLibrary(idx).Year));
    sgtitle('Spectrograms of 10 Random Noise Samples')
end

% End logging
diary off

%% Helper Functions

function [audioData, Fs, successFlag] = assembleROIAudio(wavs_filelist, read_folder, ROI_start_posix, ROI_end_posix)
% Assembles audio data for a given region of interest (ROI) from multiple WAV files
%
% Inputs:
%   wavs_filelist: struct array containing information about WAV files
%   read_folder: string, path to the folder containing WAV files
%   ROI_start_posix: double, start time of ROI in POSIX format
%   ROI_end_posix: double, end time of ROI in POSIX format
%
% Outputs:
%   audioData: vector of doubles, assembled audio data for the ROI
%   Fs: double, sampling frequency of the audio data
%   successFlag: boolean, indicates whether the operation was successful

% Initialize variables
successFlag = true;
audioData = [];
Fs = NaN;
wav_dateformat_datetime = 'yyMMdd-HHmmss';
max_attempts = 2;
retry_delay = 2; % seconds

% Convert POSIX times to datetime
ROI_start_datetime = datetime(ROI_start_posix, 'ConvertFrom', 'posixtime');
ROI_end_datetime = datetime(ROI_end_posix, 'ConvertFrom', 'posixtime');

% Find relevant WAV files
relevant_files = [];
for i = 1:length(wavs_filelist)
    file_start_str = extractAfter(wavs_filelist(i).name, "H08S1_");
    file_start_datetime = datetime(file_start_str(1:end-4), 'InputFormat', wav_dateformat_datetime);
    file_duration = seconds(wavs_filelist(i).bytes / (2 * 100)); % Assuming 16-bit samples at 100 Hz
    file_end_datetime = file_start_datetime + file_duration;
    
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
    file_start_datetime = datetime(file_start_str(1:end-4), 'InputFormat', wav_dateformat_datetime);
    file_duration = seconds(length(audio) / Fs);
    file_end_datetime = file_start_datetime + file_duration;
    
    % Calculate start and end samples as integers
    start_sample = max(1, round(seconds(ROI_start_datetime - file_start_datetime) * Fs) + 1);
    end_sample = min(length(audio), round(seconds(ROI_end_datetime - file_start_datetime) * Fs) + 1);
    
    % Append relevant audio data
    if start_sample <= length(audio) && end_sample >= 1
        audioData = [audioData; audio(start_sample:end_sample)];
    end
end

% Check if we have assembled any audio data
if isempty(audioData)
    warning('No audio data found for the specified ROI.');
    successFlag = false;
    return;
end

% Remove DC offset
audioData = audioData - mean(audioData);

end

function filename = find_closest_wav(filelist, target_date_time)
    % Convert the target date-time stamp to posixtime
    target_posix = convert_to_posix(target_date_time);
    
    % Initialize the closest posixtime and filename
    closest_posix = -inf;  % Changed from inf to -inf
    filename = '';
    
    % Iterate over the files
    for i = 1:length(filelist)
        % Extract the date-time stamp from the file name
        file_datetime_str = regexp(filelist(i).name, '\d+', 'match');
        
        % If there's no date-time stamp, skip this file
        if isempty(file_datetime_str)
            continue;
        end
        
        % Convert the date-time stamp to posixtime
        file_posix = convert_to_posix([file_datetime_str{1,3}, '-', file_datetime_str{1,4}]);
        
        % If the file's datetime is after the target, skip this file
        if file_posix > target_posix
            continue;
        end
        
        % If the file's datetime is closer to the target than the current closest, update the closest
        if file_posix > closest_posix  % Changed comparison logic
            closest_posix = file_posix;
            filename = filelist(i).name;
        end
    end
end

function [successFlag, error_info] = processNoiseSequence(i, wavs_filelist, ...
    wavSubDirPath, startTimePosix, endTimePosix, startTimeDatestrings, ...
    noise_lib_path, Fs)
    
    error_info = [];
    
    try
        if isempty(wavs_filelist)
            error('No wav files found - Check wav file paths and that storage volume is mounted.')
        end
        
        % Retrieve audio file, trim/append to region of interest
        [audioData, ~, retrievalSuccessFlag] = assembleROIAudio(...
            wavs_filelist, wavSubDirPath, startTimePosix, endTimePosix);
        
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