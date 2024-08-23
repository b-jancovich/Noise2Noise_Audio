function [signalPairs, signalPairIndices] = waveletLSHBatch(audioDatastore, nTrainingPairs, numHashFunctions, numBands, nBatchSize, soiLoFreq, soiHiFreq, fadeLen, tempDir)
    % waveletLSHBatch: Perform Wavelet Local Sensitivity Hashing with batch processing
    % This version is optimized for very large datasets
    %
    % Inputs:
    %   audioDatastore - datastore containing paths to audio files
    %   nTrainingPairs - number of training pairs to generate
    %   numHashFunctions - number of hash functions to use
    %   numBands - number of bands for LSH
    %   nBatchSize - number of audio files to process in each batch
    %   soiLoFreq - lowest frequency in the signal of interest
    %   soiHiFreq - highest frequency in the signal of interest
    %   fadeLen - length of fade in/out in seconds
    %   tempDir - directory to store temporary files
    %
    % Outputs:
    %   signalPairs - cell array of dissimilar signal pairs
    %   signalPairIndices - indices of the paired signals

    % Initialize variables
    nFiles = length(audioDatastore.Files);
    nBatches = ceil(nFiles / nBatchSize);

    fprintf('Starting Wavelet LSH Batch Processing\n');
    fprintf('Total number of files: %d\n', nFiles);
    fprintf('Number of batches: %d\n', nBatches);

    % Read the first file to get the number of samples and Fs
    fprintf('Reading first file to determine audio properties...\n');
    [audio, Fs] = read(audioDatastore);
    reset(audioDatastore);
    nSamps = length(audio);
    fprintf('Audio sample rate: %d Hz\n', Fs);
    fprintf('Number of samples per file: %d\n', nSamps);

    % Build a window function to fade-in and fade-out the signals
    fprintf('Building fade window...\n');
    windowSamps = 2 * fadeLen * Fs;
    if mod(windowSamps, 2) == 0
        windowSamps = windowSamps + 1;
    end
    window = hann(windowSamps);
    onesToAdd = nSamps - windowSamps;
    windowFull = [window(1:floor(windowSamps/2+1)); ones(onesToAdd-1, 1); flipud(window(1:floor(windowSamps/2+1)))];

    % Build a Band Stop filter to remove the signal of interest
    fprintf('Creating Band Stop filter...\n');
    Wn(1) = soiLoFreq / (Fs / 2);
    Wn(2) = soiHiFreq / (Fs / 2);
    [b, a] = butter(8, Wn, 'stop');

    % Define preprocessing function
    preprocessAudio = @(x) filtfilt(b, a, (x ./ max(abs(x))) .* windowFull - mean(x));

    % Process audio files in batches
    fprintf('Starting batch processing...\n');
    featureFiles = cell(1, nBatches);
    for batchIdx = 1:nBatches
        startIdx = (batchIdx - 1) * nBatchSize + 1;
        endIdx = min(batchIdx * nBatchSize, nFiles);
        
        fprintf('Processing batch %d of %d (files %d to %d)\n', batchIdx, nBatches, startIdx, endIdx);
        
        % Read batch of audio files
        batchFiles = audioDatastore.Files(startIdx:endIdx);
        batchAudio = cellfun(@(x) audioread(x), batchFiles, 'UniformOutput', false);
        
        % Preprocess batch audio and extract features
        fprintf('Preprocessing and extracting features for batch %d...\n', batchIdx);
        batchFeatures = preprocessAndExtractFeatures(batchAudio, preprocessAudio);
        
        % Save features to disk
        featureFile = fullfile(tempDir, sprintf('features_batch_%d.mat', batchIdx));
        save(featureFile, 'batchFeatures', '-v7.3');
        featureFiles{batchIdx} = featureFile;
        
        fprintf('Finished processing batch %d of %d\n', batchIdx, nBatches);
    end

    % Create a datastore for the feature files
    featureDS = arrayDatastore(featureFiles, 'ReadFcn', @(x) getfield(load(x), 'batchFeatures'));

    % Perform LSH on all features
    fprintf('Beginning Hashing...\n');
    hashTable = minhashLSH(featureDS, numHashFunctions, nFiles, tempDir);
    fprintf('Hashing Completed.\n');

    % Find dissimilar pairs
    fprintf('Beginning Dissimilarity Matching...\n');
    [signalPairs, signalPairIndices] = findDissimilarPairs(hashTable, numBands, nTrainingPairs, featureDS, nFiles);
    fprintf('Dissimilarity Matching Completed.\n');
    
    fprintf('Wavelet LSH Batch Processing Completed.\n');
    fprintf('Number of dissimilar pairs found: %d\n', size(signalPairs, 1));

    % Clean up temporary files
    cellfun(@delete, featureFiles);
    delete(fullfile(tempDir, 'hashTable_*.mat'));
end

function features = preprocessAndExtractFeatures(audioBatch, preprocessAudio)
    waveletType = 'db4';
    numSignals = length(audioBatch);
    features = false(2700, numSignals);  % Preallocate based on previous truncation

    fprintf('Preprocessing and extracting features for %d signals\n', numSignals);
    for i = 1:numSignals
        if mod(i, 100) == 0 || i == numSignals
            fprintf('Processing signal %d of %d\n', i, numSignals);
        end
        
        audio = audioBatch{i};
        
        % Preprocess the audio
        preprocessedAudio = preprocessAudio(audio);
        
        % Compute maximum wavelet level
        waveletLevel = wmaxlev(length(preprocessedAudio), waveletType);

        % Wavelet decomposition
        [coefficients, ~] = wavedec(preprocessedAudio, waveletLevel, waveletType);

        % Quantize coefficients
        threshold = mean(abs(coefficients));
        binaryFeatures = coefficients > threshold;

        % Truncate the binary features to remove the frequency range of zero-energy
        features(:, i) = binaryFeatures(2700:end);
    end
    fprintf('Feature extraction completed for all signals in batch\n');
    features = sparse(features);  % Convert to sparse array
end

function hashTable = minhashLSH(featureDS, numHashFunctions, numSignals, tempDir)
    fprintf('Performing MinHash LSH for %d signals with %d hash functions\n', numSignals, numHashFunctions);

    % Generate random hash functions
    a = randi([1 2^32-1], numHashFunctions, 1, 'uint32');
    b = randi([1 2^32-1], numHashFunctions, 1, 'uint32');

    % Initialize hash table
    hashTable = zeros(numHashFunctions, numSignals, 'uint32');
    signalIdx = 1;

    while hasdata(featureDS)
        batchFeatures = read(featureDS);
        batchSize = size(batchFeatures, 2);
        
        for i = 1:batchSize
            if mod(signalIdx, 1000) == 0 || signalIdx == numSignals
                fprintf('Computing hash for signal %d of %d\n', signalIdx, numSignals);
            end

            nonZeroIndices = find(batchFeatures(:,i));
            if ~isempty(nonZeroIndices)
                for j = 1:numHashFunctions
                    hashValues = mod(a(j) * uint32(nonZeroIndices) + b(j), 2^32-1);
                    hashTable(j, signalIdx) = min(hashValues);
                end
            end
            signalIdx = signalIdx + 1;
        end

        % Save hash table to disk in chunks
        if mod(signalIdx, 10000) == 0 || signalIdx > numSignals
            chunkFile = fullfile(tempDir, sprintf('hashTable_%d.mat', signalIdx));
            save(chunkFile, 'hashTable', '-v7.3');
            hashTable = zeros(numHashFunctions, numSignals, 'uint32');  % Reset for next chunk
        end
    end
    fprintf('MinHash LSH completed for all signals\n');
end

function [dissimilarPairs, dissimilarIndices] = findDissimilarPairs(hashTable, numBands, numPairs, featureDS, numSignals)
    [numHashFunctions, ~] = size(hashTable);
    bandSize = numHashFunctions / numBands;

    fprintf('Finding dissimilar pairs using %d bands\n', numBands);

    % Use a min-heap to keep track of top dissimilar pairs
    heapSize = min(numPairs * 3, nchoosek(numSignals, 2));
    dissimilarHeap = zeros(heapSize, 3); % [similarity, index1, index2]
    heapCount = 0;

    % Process each band
    for i = 1:numBands
        if mod(i, 10) == 0 || i == numBands
            fprintf('Processing band %d of %d\n', i, numBands);
        end
        
        bandStart = (i-1)*bandSize + 1;
        bandEnd = i*bandSize;
        bandHashes = hashTable(bandStart:bandEnd, :);

        % Find signals that hash to the same bucket in this band
        [~, ~, bandGroups] = unique(bandHashes', 'rows');

        % Update similarity count
        for group = 1:max(bandGroups)
            members = find(bandGroups == group);
            if length(members) > 1
                for j = 1:length(members)
                    for k = j+1:length(members)
                        updateDissimilarHeap(members(j), members(k), 1);
                    end
                end
            end
        end
    end

    fprintf('Converting heap to sorted array...\n');
    % Convert heap to sorted array
    [~, sortOrder] = sort(dissimilarHeap(1:heapCount, 1));
    sortedPairs = dissimilarHeap(sortOrder, :);

    fprintf('Removing duplicate pairs and ensuring uniqueness...\n');
    % Remove duplicate pairs and ensure uniqueness
    uniquePairs = unique(sortedPairs(:, 2:3), 'rows', 'stable');
    dissimilarIndices = uniquePairs(1:min(numPairs, size(uniquePairs, 1)), :);

    fprintf('Creating cell array of dissimilar feature pairs...\n');
    % Create cell array of dissimilar feature pairs
    dissimilarPairs = cell(size(dissimilarIndices, 1), 2);
    reset(featureDS);
    for i = 1:size(dissimilarIndices, 1)
        batchIdx1 = ceil(dissimilarIndices(i, 1) / nBatchSize);
        batchIdx2 = ceil(dissimilarIndices(i, 2) / nBatchSize);
        
        if batchIdx1 == batchIdx2
            batchFeatures = read(featureDS);
            dissimilarPairs{i, 1} = batchFeatures(:, mod(dissimilarIndices(i, 1) - 1, nBatchSize) + 1);
            dissimilarPairs{i, 2} = batchFeatures(:, mod(dissimilarIndices(i, 2) - 1, nBatchSize) + 1);
        else
            batchFeatures1 = read(featureDS);
            dissimilarPairs{i, 1} = batchFeatures1(:, mod(dissimilarIndices(i, 1) - 1, nBatchSize) + 1);
            
            % Skip to the correct batch for the second feature
            for j = 1:(batchIdx2 - batchIdx1 - 1)
                read(featureDS);
            end
            batchFeatures2 = read(featureDS);
            dissimilarPairs{i, 2} = batchFeatures2(:, mod(dissimilarIndices(i, 2) - 1, nBatchSize) + 1);
        end
        
        if mod(i, 100) == 0 || i == size(dissimilarIndices, 1)
            fprintf('Processed %d of %d dissimilar pairs\n', i, size(dissimilarIndices, 1));
        end
    end

    fprintf('Dissimilar pair finding completed. Found %d pairs.\n', size(dissimilarPairs, 1));

    % Nested function to update the dissimilar pairs heap
    function updateDissimilarHeap(index1, index2, similarityIncrease)
        if index1 > index2
            [index1, index2] = deal(index2, index1);
        end
        
        % Check if pair already exists in heap
        existingIndex = find(dissimilarHeap(1:heapCount, 2) == index1 & dissimilarHeap(1:heapCount, 3) == index2, 1);
        
        if ~isempty(existingIndex)
            % Update existing pair
            dissimilarHeap(existingIndex, 1) = dissimilarHeap(existingIndex, 1) + similarityIncrease;
            siftDown(existingIndex);
        else
            % Add new pair if heap is not full
            if heapCount < heapSize
                heapCount = heapCount + 1;
                dissimilarHeap(heapCount, :) = [similarityIncrease, index1, index2];
                siftUp(heapCount);
            elseif similarityIncrease < dissimilarHeap(1, 1)
                % Replace root if new pair is more dissimilar
                dissimilarHeap(1, :) = [similarityIncrease, index1, index2];
                siftDown(1);
            end
        end
    end

    % Heap operations
    function siftUp(index)
        while index > 1
            parentIndex = floor(index / 2);
            if dissimilarHeap(index, 1) < dissimilarHeap(parentIndex, 1)
                dissimilarHeap([index, parentIndex], :) = dissimilarHeap([parentIndex, index], :);
                index = parentIndex;
            else
                break;
            end
        end
    end

    function siftDown(index)
        while 2 * index <= heapCount
            childIndex = 2 * index;
            if childIndex < heapCount && dissimilarHeap(childIndex + 1, 1) < dissimilarHeap(childIndex, 1)
                childIndex = childIndex + 1;
            end
            if dissimilarHeap(childIndex, 1) < dissimilarHeap(index, 1)
                dissimilarHeap([index, childIndex], :) = dissimilarHeap([childIndex, index], :);
                index = childIndex;
            else
                break;
            end
        end
    end
end