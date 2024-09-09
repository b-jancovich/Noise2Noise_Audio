function dissimilarPairs = LDHPairs_v2(ads, soiHiFreq, ...
    stationarityThreshold, overlapPercent, nFFT, fadeLen, featuresToUse)
    
    % Hashing Constants
    hashTables = 20;
    bandSize = 10;

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

    % Find dissimilar pairs
    dissimilarPairs = findDissimilarPairs(allHashTables, nSignals, fileIDs);
end

function windowFull = createFadeWindow(nSamples, fadeLen, Fs)
    windowSamps = 2 * fadeLen * Fs;
    if mod(windowSamps, 2) == 0
        windowSamps = windowSamps + 1;
    end
    window = hann(windowSamps);
    onesToAdd = nSamples - windowSamps;
    windowFull = [window(1:floor(windowSamps/2+1)); ones(onesToAdd-1, 1); flipud(window(1:floor(windowSamps/2+1)))];
end

function [b, a] = createHighPassFilter(soiHiFreq, Fs)
    Wn = soiHiFreq / (Fs / 2);
    [b, a] = butter(8, Wn, 'High');
end

function allHashTables = initializeHashTables(hashTables)
    allHashTables = cell(1, hashTables);
    for i = 1:hashTables
        allHashTables{i} = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    end
end

function processBatch(ads, windowFull, b, a, Fs, windowLen, overlapPercent, ...
    nFFT, numHashes, allHashTables, hashTables, bandSize, featuresToUse)

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

function y = preprocessAudio(x, windowFull, b, a)
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

function features = extractFeatures(y, Fs, windowLen, overlapPercent, nFFT, featuresToUse)
    overlapLength = floor(windowLen * (overlapPercent/100));

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

function signature = generateLSHSignature(features, numHashes)
    [numFeatures, ~] = size(features);
    signature = false(1, numHashes);

    randomVectors = randn(numFeatures, numHashes);
    projections = features' * randomVectors;
    signature = median(projections, 1) > 0;
end

function updateHashTables(signature, fileIndex, allHashTables, hashTables, bandSize)
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

function dissimilarPairs = findDissimilarPairs(allHashTables, nSignals, fileIDs)
    
    numHashTables = length(allHashTables);
    disp(['Number of hash tables: ', num2str(numHashTables)]);
    
    % Initialize a min-heap to store the most dissimilar pairs
    dissimilarityHeap = java.util.PriorityQueue();
    
    % Process hash tables and update dissimilarity counts
    dissimilarityCounts = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    
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
        end
    end

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
                    
                    % Remove all pairs containing idx1 or idx2 from the heap
                    keysToRemove = {};
                    for j = 1:length(keys)
                        pairToCheck = sscanf(keys{j}, '%d,%d');
                        if any(ismember(pairToCheck, [idx1, idx2]))
                            keysToRemove{end+1} = keys{j};
                        end
                    end
                    for j = 1:length(keysToRemove)
                        dissimilarityCounts.remove(keysToRemove{j});
                    end
                    keys = dissimilarityCounts.keys;
                    
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

% This version works but is very slow due to sparse array indexing operations:
% function dissimilarPairs = findDissimilarPairs(allHashTables, nSignals, fileIDs)
%     disp('Starting findDissimilarPairs function');
% 
%     dissimilarityMatrix = sparse(nSignals, nSignals);
%     disp(['Initialized dissimilarityMatrix with size: ', num2str(size(dissimilarityMatrix))]);
% 
%     % Process hash tables
%     numHashTables = length(allHashTables);
%     disp(['Number of hash tables: ', num2str(numHashTables)]);
% 
%     for i = 1:numHashTables
%         disp(['Processing hash table ', num2str(i), ' of ', num2str(numHashTables)]);
%         table = allHashTables{i};
%         keys = cell2mat(table.keys);
%         disp(['Number of keys in table ', num2str(i), ': ', num2str(length(keys))]);
% 
%         for j = 1:length(keys)
%             bucket = table(keys(j));
%             disp(['Table ', num2str(i), ', Key ', num2str(j), ': Bucket size = ', num2str(length(bucket))]);
% 
%             % Choose combination method based on how many signals are in the bucket. 
%             if length(bucket) > 1
%                 if length(bucket) <= 1000
%                     combinations = nchoosek(bucket, 2);
%                 else
%                     numCombinations = min(1000, nchoosek(length(bucket), 2));
%                     combinations = datasample(bucket, 2*numCombinations, 'Replace', true);
%                     combinations = reshape(combinations, [], 2);
%                 end
%                 disp(['Number of combinations: ', num2str(size(combinations, 1))]);
% 
%                 for k = 1:size(combinations, 1)
%                     idx1 = combinations(k,1);
%                     idx2 = combinations(k,2);
%                     dissimilarityMatrix(idx1, idx2) = dissimilarityMatrix(idx1, idx2) + 1;
%                     dissimilarityMatrix(idx2, idx1) = dissimilarityMatrix(idx2, idx1) + 1;
%                 end
%             else
%                 disp('Bucket length <= 1. No matrix update.');
%             end
%         end
%         disp(['Finished processing hash table ', num2str(i)]);
%     end
% 
%     disp('Finished updating hash tables.');
%     disp('Beginning pair selection...');
% 
%     % Set diagonal to maximum value to ensure we don't pair a signal with itself
%     dissimilarityMatrix(1:nSignals+1:end) = Inf;
% 
%     maxPairs = floor(nSignals / 2);
%     dissimilarPairs = zeros(maxPairs, 2, 'uint32');
%     pairsCount = 0;
%     availableSignals = true(nSignals, 1);
% 
%     while sum(availableSignals) >= 2
%         [row, col, val] = find(dissimilarityMatrix(availableSignals, availableSignals));
% 
%         if isempty(val)
%             disp('No more valid pairs found. Breaking loop.');
%             break;
%         end
% 
%         % Debug Print:
%         % disp(['Number of non-inf values: ', num2str(length(val))]);
%         % disp(['Min value: ', num2str(min(val)), ', Max value: ', num2str(max(val))]);
%         % disp(['Max non-inf value: ', num2str(max(val(isfinite(val))))]);
% 
%         [~, minIdx] = min(val);
% 
%         % Debug Print:
%         % disp(['Selected minVal: ', num2str(minVal), ', at index: ', num2str(minIdx)]);
% 
%         availableIndices = find(availableSignals);
%         idx1 = availableIndices(row(minIdx));
%         idx2 = availableIndices(col(minIdx));
% 
%         % Debug Print:
%         % disp(['Selected idx1: ', num2str(idx1), ', idx2: ', num2str(idx2)]);
%         % disp(['availableIndices: ', num2str(length(availableIndices))]);
% 
%         if idx1 == idx2
%             disp(['Skipping self-pair: ', num2str(idx1)]);
%             disp(['Row: ', num2str(row(minIdx)), ', Col: ', num2str(col(minIdx))]);
% 
%             % Mark this signal as unavailable
%             availableSignals(idx1) = false;
%             availableSignals(idx2) = false;
%             dissimilarityMatrix(idx1, :) = Inf;
%             dissimilarityMatrix(:, idx1) = Inf;
%             disp(['Marked signal ', num2str(idx1), ' as unavailable']);
%             continue;
%         end
% 
%         pairsCount = pairsCount + 1;
%         dissimilarPairs(pairsCount, :) = [fileIDs(idx1), fileIDs(idx2)];
% 
%         availableSignals(idx1) = false;
%         availableSignals(idx2) = false;
% 
%         dissimilarityMatrix(idx1, :) = Inf;
%         dissimilarityMatrix(:, idx1) = Inf;
%         dissimilarityMatrix(idx2, :) = Inf;
%         dissimilarityMatrix(:, idx2) = Inf;
% 
%         % Display progress every 1000 signals paired
%         if mod(pairsCount, 1000) == 0
%             disp(['Paired signal #', num2str(pairsCount), ' of ', num2str(nSignals)]);
%         end
% 
%         % break loop if we have enough 
%         if pairsCount >= maxPairs
%             disp('Reached maximum number of pairs. Breaking loop.');
%             break;
%         end
% 
%     end
% 
%     dissimilarPairs = dissimilarPairs(1:pairsCount, :);
% 
%     disp(['Total pairs found: ', num2str(pairsCount)]);
%     disp('Finished findDissimilarPairs function');
% end