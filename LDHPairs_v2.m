function dissimilarPairs = LDHPairs_v2(ads, soiHiFreq, ...
    stationarityThreshold, overlapPercent, nFFT, fadeLen, featuresToUse,...
    hashTables, bandSize)
%% LDHPairs_v2 Function Documentation
%
% Function Signature:
%   function dissimilarPairs = LDHPairs_v2(ads, soiHiFreq, stationarityThreshold,
%                               overlapPercent, nFFT, fadeLen, featuresToUse, 
%                               hashTables, bandSize)
%
% Description:
%   The LDHPairs_v2 function implements a Locality-Sensitive Hashing (LSH) based algorithm
%   to find dissimilar pairs of audio signals in a large dataset. It processes audio signals
%   in batches, extracts features, generates LSH signatures, and uses these to identify
%   dissimilar pairs efficiently.
%
% Parameters:
%   ads                   - Custom audio datastore object containing the dataset
%                           of audio signals - Must be created with
%                           "batchAudioDatastore.m"
%   soiHiFreq             - Upper bandwidth limit of the Signal of Interest (Hz) 
%   stationarityThreshold - Max. duration over which the signal can be considered stationary (s)
%   overlapPercent        - Percentage of overlap between adjacent windows during feature extraction (%)
%   nFFT                  - Number of FFT points used in spectral analysis
%   fadeLen               - Length of the fade applied at the beginning and end of each audio signal (s) 
%   featuresToUse         - Cell array of strings specifying which audio
%                           features to extract (The output from function
%                           "selectAudioFeatures.m")
%   hashTables            - Number of hash tables to use in the LSH algorithm
%   bandSize              - Size of each band in the LSH signature
%
% Returns:
%   dissimilarPairs       - Matrix where each row contains the file indices of a pair of dissimilar audio signals
%
% Algorithm Overview:
%   1. Initialization: Set up analysis parameters, create filters, and initialize hash tables.
%   2. Batch Processing: Process audio signals in batches:
%      a. Preprocess each audio signal (normalization, windowing, filtering)
%      b. Extract specified audio features
%      c. Generate LSH signatures from extracted features
%      d. Update hash tables with the signatures
%   3. Dissimilarity Matching: Analyze hash tables to find dissimilar pairs of audio signals
%
% Key Components:
%
% 1. Audio Preprocessing (preprocessAudio function)
%    - Normalizes the audio signal
%    - Applies a fade window to reduce edge effects
%    - Removes DC offset
%    - Applies a high-pass filter so that the signal of interest is NOT
%    considered in the measures of dissimilarity.
%
% 2. Feature Extraction (extractFeatures function)
%    - Uses MATLAB's audioFeatureExtractor to compute various audio features
%    - Configurable to extract different sets of features (e.g., GTCC, ERB, pitch)
%
% 3. LSH Signature Generation (generateLSHSignature function)
%    - Implements a random projection method to create binary signatures
%    - Uses median of projections as the threshold for binarization
%
% 4. Hash Table Update (updateHashTables function)
%    - Divides the LSH signature into bands
%    - Updates multiple hash tables based on the band hashes
%
% 5. Dissimilar Pair Finding (findDissimilarPairs function)
%    - Analyzes hash tables to identify potentially dissimilar pairs
%    - Uses a min-heap to efficiently select the most dissimilar pairs
%    - Ensures each audio signal is paired only once
%
% Usage Example:
%   ads = audioDatastore('path/to/audio/files');
%   soiHiFreq = 100;  % Hz
%   stationarityThreshold = 0.1;  % seconds
%   overlapPercent = 50;
%   nFFT = 1024;
%   fadeLen = 0.01;  % seconds
%   featuresToUse = {'mfcc', 'spectralCentroid', 'spectralFlux'};
%   hashTables = 10;
%   bandSize = 4;
%   
%   dissimilarPairs = LDHPairs_v2(ads, soiHiFreq, stationarityThreshold, overlapPercent, ...
%                                 nFFT, fadeLen, featuresToUse, hashTables, bandSize);
%
% Note:
%   This function requires the MATLAB Audio Toolbox for feature extraction.
%   Ensure all dependencies are available before running the function.
%
% See also: audioFeatureExtractor, audioDatastore, selectAudioFeatures, batchAudioDatastore

    % Get the size of the dataset and signal info
    nSignals = ads.NumObservations;

    % Read a single audio file to get signal info
    [sig0, sig0_info] = readSingle(ads);
    nSamples = size(sig0, 1);
    Fs = sig0_info.SampleRate;
    miniBatchSize = ads.MiniBatchSize;
    reset(ads)

    % Set analysis window size
    windowLen = floor(stationarityThreshold * Fs);

    % Precompute window and filter
    windowFull = createFadeWindow(nSamples, fadeLen, Fs);
    [b, a] = createHighPassFilter(soiHiFreq, Fs);

    % Initialize hash tables
    allHashTables = initializeHashTables(hashTables);

    % Set number of hashes
    numHashes = hashTables * bandSize;

    % Compute number of batches
    numBatches = ceil(nSignals / miniBatchSize);

    % Store file IDs
    fileIDs = [ads.Files.index];

    % Extract features, convert to signatures, and build hash tables
    batchCount = 0;
    while hasdata(ads)
        % Process batch
        processBatch(ads, windowFull, b, a, Fs, windowLen, overlapPercent,...
            nFFT, numHashes, allHashTables, hashTables, bandSize, featuresToUse);
        
        % Update Counter
        batchCount = batchCount+1;
        fprintf('Processed batch %d of %d\n', batchCount, numBatches);
    end

    disp('Pre-processing, Feature Extraction and Hashing completed.')
    disp('Starting dissimilarity Matching....')

    % Run dissimilar pair matching
    dissimilarPairs = findDissimilarPairs(allHashTables, nSignals, fileIDs);
end

%% createFadeWindow
function windowFull = createFadeWindow(nSamples, fadeLen, Fs)
% Function Signature:
%   function windowFull = createFadeWindow(nSamples, fadeLen, Fs)
%
% Description:
%   Creates a window function with faded edges to be applied to the audio signal.
%
% Parameters:
%   nSamples - Total number of samples in the audio signal
%   fadeLen  - Length of the fade (in seconds) to be applied at the edges
%   Fs       - Sampling frequency of the audio signal
%
% Returns:
%   windowFull - Window function with faded edges and ones in the middle

    windowSamps = 2 * fadeLen * Fs;
    if mod(windowSamps, 2) == 0
        windowSamps = windowSamps + 1;
    end
    window = hann(windowSamps);
    onesToAdd = nSamples - windowSamps;
    windowFull = [window(1:floor(windowSamps/2+1)); ones(onesToAdd-1, 1); flipud(window(1:floor(windowSamps/2+1)))];
end

%% createHighPassFilter
function [b, a] = createHighPassFilter(soiHiFreq, Fs)
% Function Signature:
%   function [b, a] = createHighPassFilter(soiHiFreq, Fs)
%
% Description:
%   Creates coefficients for a high-pass Butterworth filter.
%
% Parameters:
%   soiHiFreq - Cutoff frequency for the high-pass filter (in Hz)
%   Fs        - Sampling frequency of the audio signal
%
% Returns:
%   b - Numerator coefficients of the filter
%   a - Denominator coefficients of the filter

    Wn = soiHiFreq / (Fs / 2);
    [b, a] = butter(8, Wn, 'High');
end

%% initializeHashTables
function allHashTables = initializeHashTables(hashTables)
% Function Signature:
%   function allHashTables = initializeHashTables(hashTables)
%
% Description:
%   Initializes the hash tables used in the LSH algorithm.
%
% Parameters:
%   hashTables - Number of hash tables to create
%
% Returns:
%   allHashTables - Cell array of hash tables (MATLAB containers.Map objects)

    allHashTables = cell(1, hashTables);
    for i = 1:hashTables
        allHashTables{i} = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    end
end

%% processBatch
function processBatch(ads, windowFull, b, a, Fs, windowLen, overlapPercent, ...
    nFFT, numHashes, allHashTables, hashTables, bandSize, featuresToUse)
% Function Signature:
%   function processBatch(ads, windowFull, b, a, Fs, windowLen, overlapPercent, nFFT, numHashes, allHashTables, hashTables, bandSize, featuresToUse)
%
% Description:
%   Processes a batch of audio signals, extracting features and updating hash tables.
%
% Parameters:
%   ads             - Audio datastore object
%   windowFull      - Window function to apply to each signal
%   b, a            - Filter coefficients for high-pass filter
%   Fs              - Sampling frequency
%   windowLen       - Length of the analysis window (in samples)
%   overlapPercent  - Overlap percentage for feature extraction
%   nFFT            - Number of FFT points
%   numHashes       - Total number of hash functions
%   allHashTables   - Cell array of hash tables
%   hashTables      - Number of hash tables
%   bandSize        - Size of each band in the LSH signature
%   featuresToUse   - Cell array of feature names to extract

    [xbatch, ~] = read(ads);

    % Each column is a signal
    for i = 1:size(xbatch, 2)
        x = xbatch(:, i);
        y = preprocessAudio(x, windowFull, b, a);

        % If preprocess returns zeros, skip this file.
        if sum(y) == 0
            continue
        end
        features = extractFeatures(y, Fs, windowLen, overlapPercent, nFFT, featuresToUse);
        signature = generateLSHSignature(features, numHashes);
        fileIndex = ads.CurrentFileIndex + i - ads.MiniBatchSize - 1;
        updateHashTables(signature, fileIndex, allHashTables, hashTables, bandSize);
    end
end

%% preprocessAudio
function y = preprocessAudio(x, windowFull, b, a)
% Function Signature:
%   function y = preprocessAudio(x, windowFull, b, a)
%
% Description:
%   Preprocesses an audio signal by normalizing, windowing, removing DC offset, and filtering.
%
% Parameters:
%   x          - Input audio signal
%   windowFull - Window function to apply to the signal
%   b, a       - Filter coefficients for high-pass filter
%
% Returns:
%   y - Preprocessed audio signal

    % Check for NaN or Inf values in the input
    if any(~isfinite(x))
        warning('Input signal contains NaN or Inf values. Replacing with zeros.');
        x(~isfinite(x)) = 0;
    end
    
    % if signal is silent, do not proceed
    if rms(x) < 1e-7
        warning('Signal is silent. Returning zeros.');
        y = zeros(size(x));
        return;
    end
    
    % Normalize, apply window, remove DC offset, and filter
    y = (x ./ max(abs(x))) .* windowFull - mean(x);
    y = filtfilt(b, a, y);
end

%% extractFeatures
function features = extractFeatures(y, Fs, windowLen, overlapPercent, nFFT, featuresToUse)
    overlapLength = floor(windowLen * (overlapPercent/100));
% Function Signature:
%   function features = extractFeatures(y, Fs, windowLen, overlapPercent, nFFT, featuresToUse)
%
% Description:
%   Extracts audio features from a preprocessed audio signal.
%
% Parameters:
%   y               - Preprocessed audio signal
%   Fs              - Sampling frequency
%   windowLen       - Length of the analysis window (in samples)
%   overlapPercent  - Overlap percentage for feature extraction
%   nFFT            - Number of FFT points
%   featuresToUse   - Cell array of feature names to extract
%
% Returns:
%   features - Matrix of extracted features

    afe = audioFeatureExtractor('SampleRate', Fs, ...
        'Window', hamming(windowLen, 'periodic'), ...
        'OverlapLength', overlapLength, ...
        'FFTLength', nFFT);

    % Set lowest possible low frequency limit for pitch extractor.
    lfLim = 1;
    while Fs/lfLim >= windowLen
        lfLim = lfLim + 1;
    end

    % Set extractor parameters
    setExtractorParameters(afe,"gtcc",NumCoeffs=13);
    setExtractorParameters(afe,"erb", NumBands=13);
    setExtractorParameters(afe,"pitch",Range=[lfLim, Fs/2]);

    % Get all available feature extractor names
    afe_info = info(afe, "all");
    featureSwitches = fields(afe_info);
    
    % Set only the specified features to true
    for i = 1:length(featureSwitches)
        if ismember(featureSwitches{i}, featuresToUse)
            afe.set(featureSwitches{i}, true);
        else
            afe.set(featureSwitches{i}, false);
        end
    end

    features = extract(afe, y);
end

%% generateLSHSignature
function signature = generateLSHSignature(features, numHashes)
% Function Signature:
%   function signature = generateLSHSignature(features, numHashes)
%
% Description:
%   Generates a Locality-Sensitive Hashing (LSH) signature from extracted features.
%
% Parameters:
%   features  - Matrix of extracted audio features
%   numHashes - Number of hash functions to use
%
% Returns:
%   signature - Binary LSH signature
%
% Algorithm:
%   1. Generate random projection vectors
%   2. Project features onto random vectors
%   3. Create binary signature based on median of projections

    [numFeatures, ~] = size(features);
    signature = false(1, numHashes);
    randomVectors = randn(numFeatures, numHashes);
    projections = features' * randomVectors;
    signature = median(projections, 1) > 0;
end

%% updateHashTables
function updateHashTables(signature, fileIndex, allHashTables, hashTables, bandSize)
% Function Signature:
%   function updateHashTables(signature, fileIndex, allHashTables, hashTables, bandSize)
%
% Description:
%   Updates the LSH hash tables with the signature of a processed audio file.
%
% Parameters:
%   signature     - Binary LSH signature of the audio file
%   fileIndex     - Index of the current audio file
%   allHashTables - Cell array of hash tables
%   hashTables    - Number of hash tables
%   bandSize      - Size of each band in the LSH signature
%
% Algorithm:
%   1. Divide the signature into bands
%   2. For each band:
%      a. Compute band hash
%      b. Update corresponding hash table

    for j = 1:hashTables
        bandStart = (j-1)*bandSize + 1;
        bandEnd = j*bandSize;
        band = signature(bandStart:bandEnd);
        bandHash = sum(band);

        if ~isKey(allHashTables{j}, bandHash)
            allHashTables{j}(bandHash) = fileIndex;
        else
            allHashTables{j}(bandHash) = [allHashTables{j}(bandHash), fileIndex];
        end
    end
    
    if mod(fileIndex, 1000) == 0
        disp(['Processed file index: ', num2str(fileIndex)]);
    end
end

%% findDissimilarPairs
function dissimilarPairs = findDissimilarPairs(allHashTables, nSignals, fileIDs)
% Function Signature:
%   function dissimilarPairs = findDissimilarPairs(allHashTables, nSignals, fileIDs)
%
% Description:
%   Analyzes hash tables to find pairs of dissimilar audio signals.
%
% Parameters:
%   allHashTables - Cell array of hash tables
%   nSignals      - Total number of audio signals processed
%   fileIDs       - Array of file IDs corresponding to the processed signals
%
% Returns:
%   dissimilarPairs - Matrix where each row contains indices of a dissimilar pair
%
% Algorithm:
%   1. Initialize dissimilarity counts and min-heap
%   2. Process hash tables to update dissimilarity counts
%   3. Convert dissimilarity counts to a heap
%   4. Select most dissimilar pairs while ensuring each signal is used only once
%
% Note:
%   This function uses Java's PriorityQueue for efficient heap operations.

    numHashTables = length(allHashTables);
    disp(['Number of hash tables: ', num2str(numHashTables)]);

    % Initialize a min-heap to store the most dissimilar pairs
    dissimilarityHeap = java.util.PriorityQueue();

    % Set dissimilarityCounts type according to numHashTables
    if numHashTables <= 127
        countType = 'int8';
    elseif numHashTables <= 32767
        countType = 'int16';
    else
        countType = 'int32';
    end

    % Process hash tables and update dissimilarity counts
    dissimilarityCounts = containers.Map('KeyType', 'char', 'ValueType', countType);

    for i = 1:numHashTables
        disp(['Processing hash table ', num2str(i), ' of ', num2str(numHashTables)]);
        table = allHashTables{i};
        keys = cell2mat(table.keys);

        for j = 1:length(keys)
            bucket = table(keys(j));
            if length(bucket) > 1
                combinations = nchoosek(bucket, 2);
                for k = 1:size(combinations, 1)
                    pair = sort([combinations(k,1), combinations(k,2)]);
                    pairKey = sprintf('%d,%d', pair(1), pair(2));
                    if isKey(dissimilarityCounts, pairKey)
                        dissimilarityCounts(pairKey) = dissimilarityCounts(pairKey) + 1;
                    else
                        dissimilarityCounts(pairKey) = 1;
                    end
                end
            end
            fprintf('Completed pairing from bucket %d\n', j)
        end
    end

    clearvars bucket combinations

    disp('Finished processing hash tables. Beginning pair selection...');

    % Convert dissimilarity counts to a heap
    keys = dissimilarityCounts.keys;
    for i = 1:length(keys)
        pairKey = keys{i};
        count = dissimilarityCounts(pairKey);
        dissimilarityHeap.add(java.lang.Double(count));
    end

    maxPairs = floor(nSignals / 2);
    dissimilarPairs = zeros(maxPairs, 2, 'uint32');
    pairsCount = 0;
    availableSignals = true(nSignals, 1);

    while ~dissimilarityHeap.isEmpty() && pairsCount < maxPairs
        minCount = dissimilarityHeap.poll();

        % Find the pair with this count
        for i = 1:length(keys)
            pairKey = keys{i};
            if dissimilarityCounts(pairKey) == minCount
                pair = sscanf(pairKey, '%d,%d');
                idx1 = pair(1);
                idx2 = pair(2);

                if availableSignals(idx1) && availableSignals(idx2)
                    pairsCount = pairsCount + 1;
                    dissimilarPairs(pairsCount, :) = [fileIDs(idx1), fileIDs(idx2)];
                    availableSignals([idx1, idx2]) = false;

                    % Parallel Version
                    % Remove all pairs containing idx1 or idx2 from the heap
                    keysToRemove = cell(1, length(keys));
                    parfor j = 1:length(keys)
                        pairToCheck = sscanf(keys{j}, '%d,%d');
                        if any(ismember_simple(pairToCheck, [idx1, idx2]))
                            keysToRemove{j} = keys{j};
                        else
                            keysToRemove{j} = '';
                        end
                    end

                    % Remove empty cells from keysToRemove
                    keysToRemove = keysToRemove(~cellfun('isempty', keysToRemove));

                    for j = 1:length(keysToRemove)
                        dissimilarityCounts.remove(keysToRemove{j});
                    end
                    keys = dissimilarityCounts.keys;
                    % End parallel version

                    % % % Serial Version
                    % % % Remove all pairs containing idx1 or idx2 from the heap
                    % % keysToRemove = {};
                    % % for j = 1:length(keys)
                    % %     pairToCheck = sscanf(keys{j}, '%d,%d');
                    % %     if any(ismember_simple(pairToCheck, [idx1, idx2]))
                    % %         keysToRemove{end+1} = keys{j};
                    % %     end
                    % % end
                    % % for j = 1:length(keysToRemove)
                    % %     dissimilarityCounts.remove(keysToRemove{j});
                    % % end
                    % % keys = dissimilarityCounts.keys;
                    % % % End serial version

                    if mod(pairsCount, 1000) == 0
                        disp(['Paired signal #', num2str(pairsCount), ' of ', num2str(maxPairs)]);
                    end
                    break;
                end
            end
        end
    end

    dissimilarPairs = dissimilarPairs(1:pairsCount, :);
    disp(['Total pairs found: ', num2str(pairsCount)]);
    disp('Finished findDissimilarPairs function');
end

%% ismember_simple
function tf = ismember_simple(a, s)
    % Optimized ismember function for specific input sizes & types
    % a: [2x1] vector of type double
    % s: [1x2] vector of type double
    % Returns: [2x1] logical vector

    % Preallocate output
    tf = false(2, 1);
    
    % Direct comparisons
    tf(1) = (a(1) == s(1)) || (a(1) == s(2));
    tf(2) = (a(2) == s(1)) || (a(2) == s(2));
end