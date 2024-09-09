classdef batchAudioDatastore < matlab.io.Datastore & matlab.io.datastore.MiniBatchable
    %BATCHAUDIODATASTORE Custom datastore for handling audio files in batches
    %   The batchAudioDatastore class provides functionality to read audio files,
    %   optionally extract labels from filenames, and supports batch processing.
    %
    %   batchAudioDatastore Properties:
    %       Files           - Structure array containing information about the audio files
    %       MiniBatchSize   - Number of audio files to read in each batch
    %       CurrentFileIndex - Index of the current file being processed
    %       Order           - Randomized order of file indices for shuffled reading
    %       SampleRate      - Sample rate of the audio files (assumes all files have the same rate)
    %       SignalLength    - Length (in samples) of each audio signal (assumes all files have the same length)
    %       Labels          - Cell array of labels extracted from filenames (if enabled)
    %       NumObservations - Total number of audio files in the datastore
    %       MinSNR          - Minimum Signal-to-Noise Ratio for file filtering
    %
    %   batchAudioDatastore Methods:
    %       batchAudioDatastore - Constructor for creating a batchAudioDatastore object
    %       read         - Read the next batch of audio files from the datastore
    %       readSingle   - Read a single audio file from the datastore
    %       hasdata      - Check if there are more files to read in the datastore
    %       reset        - Reset the datastore to the beginning and reshuffle the reading order
    %       progress     - Get the fraction of files that have been read from the datastore
    %       filterFilesBySNR - Filter files based on naming convention and minimum SNR
    %
    %   Example:
    %       % Create a datastore with labels from filenames and minimum SNR
    %       ds = batchAudioDatastore('path/to/audio/files', 'LabelSource', 'fileNames', 'MiniBatchSize', 64, 'MinSNR', -0.5);
    %
    %       % Read a batch of files
    %       [data, info] = read(ds);
    %       audioData = data;
    %       labels = info.Labels;
    %
    %       % Read files in a loop
    %       reset(ds);
    %       while hasdata(ds)
    %           [data, info] = read(ds);
    %           % Process data here
    %       end
    %
    %       % Read a single file
    %       [data, info] = readSingle(ds);
    %       singleAudioFile = data;
    %       fileLabel = info.Labels;
    %
    %   See also audioDatastore, datastore, matlab.io.datastore.MiniBatchable

    properties
        Files
        MiniBatchSize
        CurrentFileIndex
        Order
        SampleRate
        SignalLength
        Labels
        MinSNR
    end

    properties(SetAccess = protected)
        NumObservations
    end

    methods
        function ds = batchAudioDatastore(folder, varargin)
            % Parse input arguments
            p = inputParser;
            addRequired(p, 'folder', @(x) ischar(x) || isstring(x));
            addParameter(p, 'FileExtensions', '.wav', @(x) ischar(x) || isstring(x) || iscellstr(x));
            addParameter(p, 'MiniBatchSize', 1, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'LabelSource', 'none', @(x) ischar(x) || isstring(x));
            addParameter(p, 'MinSNR', -Inf, @(x) isnumeric(x)); % New parameter for minimum SNR
            parse(p, folder, varargin{:});

            % Set properties
            exts = p.Results.FileExtensions;
            if ischar(exts) || isstring(exts)
                exts = {char(exts)};
            end

            files = [];
            for i = 1:length(exts)
                files = [files; dir(fullfile(folder, ['*' exts{i}]))];
            end

            if isempty(files)
                error('No files found with the specified extensions in the provided folder.');
            end

            ds.MiniBatchSize = p.Results.MiniBatchSize;
            ds.MinSNR = p.Results.MinSNR; % Set the MinSNR property

            % Filter files based on naming convention and MinSNR
            validFiles = ds.filterFilesBySNR(files);

            if isempty(validFiles)
                error('No files meet the specified SNR criterion or follow the required naming convention.');
            end

            % Write the detection ID for this file to the files struct
            for i = 1:length(validFiles)
                index = extractBetween(validFiles(i).name, 'detectionAudio_', '_');
                validFiles(i).index = sscanf(sprintf(' %s', index{1}),'%f',[1,Inf]);
            end

            % Don't read too many files:
            if numel(validFiles) > 1000
                nFiles2Check = 200;
                fileIDX = randi(numel(validFiles), 1, nFiles2Check);
            else
                nFiles2Check = numel(validFiles);
                fileIDX = linspace(1, numel(validFiles), numel(validFiles));
            end

            % Preallocate and gather file info in a single pass
            fileLengths = zeros(nFiles2Check, 1);
            SampleRate = zeros(nFiles2Check, 1);

            % Get SampleRate & Length for a random selection of files
            for i = 1:nFiles2Check
                idx = fileIDX(i);
                filePath = fullfile(validFiles(idx).folder, validFiles(idx).name);
                info = audioinfo(filePath);
                fileLengths(i) = info.TotalSamples;
                SampleRate(i) = info.SampleRate;
            end

            % Determine most common length and SampleRate
            ds.SignalLength = mode(fileLengths);
            ds.SampleRate = mode(SampleRate);
            ds.Files = validFiles;
            ds.NumObservations = numel(ds.Files);
            ds.Order = randperm(ds.NumObservations);
            ds.CurrentFileIndex = 1;
            
            % Handle labels
            labelSource = validatestring(p.Results.LabelSource, {'none', 'fileNames'});
            if strcmpi(labelSource, 'fileNames')
                ds.Labels = cellfun(@(x) extractLabel(x), {validFiles.name}, 'UniformOutput', false)';
            else
                ds.Labels = {};
            end
        end

        function tf = hasdata(ds)
            tf = ds.CurrentFileIndex <= ds.NumObservations;
        end

        function [data, info] = read(ds)
            if ~ds.hasdata()
                error('No more data to read. Use reset to restart.');
            end

            idxEnd = min(ds.CurrentFileIndex + ds.MiniBatchSize - 1, ds.NumObservations);
            idx = ds.CurrentFileIndex:idxEnd;
            files = ds.Files(ds.Order(idx));
            audioData = zeros(ds.SignalLength, numel(files));
            fileNames = cell(numel(files), 1);
            labels = cell(numel(files), 1);

            for i = 1:numel(files)
                filePath = fullfile(files(i).folder, files(i).name);
                [thisAudio, thisSampleRate] = audioread(filePath);
                
                % Ensure signal matches expected length and SampleRate
                if length(thisAudio) == ds.SignalLength && thisSampleRate == ds.SampleRate
                    audioData(:, i) = thisAudio;
                    fileNames{i} = files(i).name;
                    if ~isempty(ds.Labels)
                        labels{i} = ds.Labels{ds.Order(idx(i))};
                    end
                else
                    audioData(:, i) = NaN;
                    fileNames{i} = NaN;
                    labels{i} = NaN;
                    disp('Incorrect number of samples or SampleRate for this file. Skipping.')
                end
            end

            ds.CurrentFileIndex = idxEnd + 1;

            data = audioData;
            info = struct('FileNames', {fileNames}, 'SampleRate', ds.SampleRate, 'Labels', {labels});
        end

        function [data, info] = readSingle(ds)
            if ~ds.hasdata()
                error('No more data to read. Use reset to restart.');
            end

            fileIdx = ds.CurrentFileIndex;
            ds.CurrentFileIndex = ds.CurrentFileIndex + 1;  % Increment index for next call
            file = ds.Files(ds.Order(fileIdx));
            filePath = fullfile(file.folder, file.name);

            % Read single audio file
            [thisAudio, thisSampleRate] = audioread(filePath);

            % Ensure signal matches expected length and SampleRate
            if length(thisAudio) == ds.SignalLength && thisSampleRate == ds.SampleRate
                audioData = thisAudio;
                fileName = file.name;
                if ~isempty(ds.Labels)
                    label = ds.Labels{ds.Order(fileIdx)};
                else
                    label = [];
                end
            else
                audioData = NaN;
                fileName = NaN;
                label = NaN;
                disp('Incorrect number of samples or SampleRate for this file. Skipping.')
            end

            data = audioData;
            info = struct('FileNames', {fileName}, 'SampleRate', ds.SampleRate, 'Labels', {label});
        end

        function reset(ds)
            ds.CurrentFileIndex = 1;
            ds.Order = randperm(ds.NumObservations);  % Reshuffle data
        end

        function frac = progress(ds)
            frac = (ds.CurrentFileIndex - 1) / ds.NumObservations;
        end

        % Setter for MiniBatchSize to ensure it's always a positive integer
        function set.MiniBatchSize(ds, value)
            if ~isnumeric(value) || value <= 0 || mod(value,1) ~= 0
                error('MiniBatchSize must be a positive integer.');
            end
            ds.MiniBatchSize = value;
        end

        % Method to filter files based on SNR
        function validFiles = filterFilesBySNR(ds, files)
            validFiles = struct('folder', {}, 'name', {}, 'date', {}, 'bytes', {}, 'isdir', {}, 'datenum', {});
            validIdx = 1;
            for i = 1:numel(files)
                [~, name, ~] = fileparts(files(i).name);
                parts = strsplit(name, '_');
                if numel(parts) == 7 && strcmp(parts{1}, 'detectionAudio')
                    try
                        snr = str2double(parts{end});
                        if ~isnan(snr) && snr >= ds.MinSNR
                            validFiles(validIdx) = files(i);
                            validIdx = validIdx + 1;
                        end
                    catch
                        % Skip files that don't match the expected format
                        continue;
                    end
                end
            end
        end

        % Setter for MinSNR to ensure it's always a numeric value
        function set.MinSNR(ds, value)
            if ~isnumeric(value)
                error('MinSNR must be a numeric value.');
            end
            ds.MinSNR = value;
        end
    end
end

function label = extractLabel(filename)
    % Extract label from filename
    parts = strsplit(filename, '_');
    if numel(parts) >= 6
        label = strjoin(parts(2:7), '_');
        % Remove the file extension if present
        label = regexprep(label, '\.wav$', '');
    else
        label = '';
    end
end