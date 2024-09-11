% Whale Song Retriever (Auto-restart version)
% Step 1 of n2n train-test dataset builder
%
% Searches the detection results and retrieves the time-datestamps for all
% detections that are separated by >= "minutesSeparation", then retrieves
% the corresponding wav files from the continuous audio data and trims each
% one to the detection. Resulting detections are saved out as wav files. If
% a detection runs over the end of a wav file into the next one, files are
% checked for continuity, then concatenated.
%
% Ben Jancovich, 2024
% Centre for Marine Science and Innovation
% School of Biological, Earth and Environmental Sciences
% University of New South Wales, Sydney, Australia
%
clear
close all
clc

%% Load project configuration file

here = pwd;
run(here, 'config.m');
disp('Loaded N2N Config file.')

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

% Remove invalid detections
validDetection = ~(isnan(detectionsAll.SNR) | isnan(detectionsAll.SINR) | ...
    isinf(detectionsAll.SNR) | isinf(detectionsAll.SINR));

detectionsAll = detectionsAll(validDetection, :);

% Convert serial time to datetime and sort
detectionsAll.datetime_Readable = datetime(detectionsAll.Time, 'ConvertFrom', 'datenum');
detectionsAll.posix_time = (detectionsAll.Time - datenum('1970-01-01')) * 86400;
[~, sortIdx] = sort(detectionsAll.posix_time);
detectionsAll = detectionsAll(sortIdx, :);

% Filter based on time separation
minTimeDiff = minutesSeparation * 60;  % Convert minutes to seconds
validDetections = false(height(detectionsAll), 1);
validDetections(1) = true;
lastValidTime = detectionsAll.posix_time(1);

for i = 2:height(detectionsAll)
    if detectionsAll.posix_time(i) - lastValidTime >= minTimeDiff
        validDetections(i) = true;
        lastValidTime = detectionsAll.posix_time(i);
    end
end

detectionsAll = detectionsAll(validDetections, :);

%% Process audio segments and save as separate WAV files

% Pre-allocate arrays
nDetections = height(detectionsAll);
wav_subdir_paths = strings(nDetections, 1);
detection_datestrings = strings(nDetections, 1);
detectionsAll.audioFilename = strings(nDetections, 1);

% Pre-compute common values
for i = 1:nDetections
    % wav_subdir_paths(i) = fullfile(rawAudioPath, strcat(wav_subdir_prefix, num2str(detectionsAll.Year(i))), 'wav/');
    wav_subdir_paths(i) = fullfile(rawAudioPath, [wav_subdir_prefix, num2str(detectionsAll.Year(i))], 'wav/');
    detection_datestrings(i) = string(datestr(detectionsAll.Time(i), wav_dateformat_serial));
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

%% Helper Functions

function safeName = makeFilenameSafe(filename)
% Takes a string and makes it save for use as a filename in windows
% filesystems.

% Replace characters that are not safe for filenames
safeName = regexprep(filename, '[<>:"/\|?*]', '_');

% Replace spaces with underscores
safeName = strrep(safeName, ' ', '_');

% Ensure the filename is not too long (max 255 characters)
if length(safeName) > 255
    safeName = safeName(1:255);
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

function [successFlag, fileNameOut] = findROI_wavWriter(...
    read_fileName, wavs_filelist, write_path, ROI_datestring,...
    iteration, datetime_readable, SNR, expected_num_samples, ...
    call_duration, buffer_duration)
% Takes the read path for a CTBTO wav file, and the detection datestring
% for the detection in that wav, then trims the wav to the time region of
% interest and saves it out as a new wav file.

% Init
successFlag = true;
wav_dateformat_serial = 'yymmdd-HHMMSS';
wav_dateformat_datetime = 'yyMMdd-HHmmss';
max_attempts = 2;
retry_delay = 2; % seconds
fileNameOut = NaN;
silence_threshold = 1e-7; % RMS threshold for silence

% Read Audio
for attempt = 1:max_attempts
    try
        [audiodata, Fs] = audioread(read_fileName);
        successFlag = true;
        break; % Exit the loop if successful
    catch
        if attempt < max_attempts
            warning('Failed to read audio file for detection %d. Retrying in %d seconds... (Attempt %d of %d)', ...
                iteration, retry_delay, attempt, max_attempts);
            disp(read_fileName)
            pause(retry_delay);
        else
            warning('Failed to read audio file for detection %d after %d attempts. Skipping.', ...
                iteration, max_attempts);
            disp(read_fileName)
            return
        end
    end
end

% Get file start and end times from file name
[read_folder, fileName, fileExt] = fileparts(read_fileName);
file_start_str = extractAfter(fileName, "H08S1_");
file_start_datetime = datetime(file_start_str, 'InputFormat', wav_dateformat_datetime);
wav_duration = numel(audiodata) / Fs;
file_end_datetime = file_start_datetime + seconds(wav_duration);
file_start_serial = datenum(file_start_datetime);
file_end_serial = datenum(file_end_datetime);
timevector_serial = linspace(file_start_serial, file_end_serial, numel(audiodata));
detection_datetime_serial = datenum(char(ROI_datestring), wav_dateformat_serial);
[~, dettime_idx] = min(abs(timevector_serial - detection_datetime_serial));

% Region of interest start is 1, or detection time minus buffer, whichever is larger.
ROIstart = max(1, dettime_idx - round(buffer_duration * Fs));

% Region of interest end is the length of the file, or the detection
% time plus the call duration, plus the buffer duration, whichever is smaller.
ROIend = min(length(audiodata), dettime_idx + round((call_duration + buffer_duration) * Fs));

% Handle case where detection audio is split across two files
if ROIstart >= 1 && ...
        ROIend <= length(audiodata) && ...
        length(audiodata) == expected_num_samples
    detectionAudio = audiodata(ROIstart:ROIend);
else
    detectionAudio = audiodata(ROIstart:ROIend);
    while length(detectionAudio) < expected_num_samples
        fprintf('Retrieved wav is too short (%d). Attempting to append next file...\n', length(detectionAudio))

        % Find the index of the current file
        current_file_index = find(strcmp({wavs_filelist.name}, strcat(fileName, fileExt)), 1);

        if current_file_index < length(wavs_filelist)
            next_filename = wavs_filelist(current_file_index + 1).name;
            next_filepath = fullfile(read_folder, next_filename);

            % Extract date information from filenames
            current_datestring = extractAfter(fileName, '_');
            current_date = datetime(current_datestring, 'InputFormat', wav_dateformat_datetime);
            next_date = datetime(next_filename(7:end-4), 'InputFormat', wav_dateformat_datetime);

            % Read the next file
            try
                [next_audiodata, ~] = audioread(next_filepath);
            catch
                warning('Failed to read the next audio file. Skipping detection %d.', iteration);
                successFlag = false;
                return;
            end

            % Calculate expected time difference
            expected_time_diff = seconds(numel(audiodata) / Fs);
            actual_time_diff = next_date - current_date;

            % Check if files are continuous (allow for small tolerance, e.g., 0.5 seconds)
            if abs(expected_time_diff - actual_time_diff) > seconds(0.5)
                warning('Files are not continuous. Skipping detection %d.', iteration);
                successFlag = false;
                return;
            end

            % Append the files
            detectionAudio = [detectionAudio; next_audiodata];
            fprintf('Successfully appended next file. Length is now %d.\n', length(detectionAudio))
        else
            warning('This is the last file in the directory. Cannot append next file. Skipping detection %d.', iteration);
            successFlag = false;
            return;
        end
    end
end

% If we have extra samples, trim the end
if numel(detectionAudio) > expected_num_samples
    detectionAudio = detectionAudio(1:expected_num_samples);
end

% Remove DC offset
detectionAudio = detectionAudio - mean(detectionAudio);

% Calculate RMS
rms_level = sqrt(mean(detectionAudio.^2));

% Check if the signal is silent
if rms_level < silence_threshold
    warning('Detection %d is silent (RMS: %.9f). Skipping.', iteration, rms_level);
    successFlag = false;
    return;
end

% Create filename for the WAV file
wav_filename = sprintf('detectionAudio_%d_%s_%f.wav', iteration, ...
    char(datetime_readable), SNR);
wav_filename = makeFilenameSafe(wav_filename);

% Full path for the new WAV file
full_wav_path = fullfile(write_path, wav_filename);

for attempt = 1:max_attempts
    try
        % Save the WAV file
        audiowrite(full_wav_path, detectionAudio, Fs);

        % If we've made it here, the file was saved successfully
        fileNameOut = string(wav_filename);
        disp(['Saved detection ', num2str(iteration), ' as ', wav_filename, ' (RMS: ', num2str(rms_level), ')']);
        break
    catch
        if attempt < max_attempts
            warning('Failed to save WAV file for detection %d. Retrying in %d seconds... (Attempt %d of %d)', ...
                iteration, retry_delay, attempt, max_attempts);
            pause(retry_delay);
        else
            warning('Failed to save WAV file for detection %d after %d attempts. Skipping.', ...
                iteration, max_attempts);
            successFlag = false;
            return;  % Return to the calling script or function
        end
    end
end
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
