function [dissimilarPairs, dissimilarIndices] = waveletLSH(audioData, numPairs, numHashFunctions, numBands)
%
% This function uses wavelet decomposition and Locality Sensitive Hashing 
% (LSH) to efficiently find dissimilar pairs of signals in a large set of 
% audio signals. It uses the 'db4' wavelet with 'n' levels of decomposition 
% on each audio signal. It then quantizes the wavelet coefficients to 
% create binary features and applies a MinHash to the binary features 
% before using the banding technique of LSH to efficiently group similar 
% signals. To pair signals with maximal dissimilarity, the function then 
% counts the number of times each pair of signals is hashed to the same 
% bucket across all bands. Pairs with lower counts are considered more 
% dissimilar.
%
% Inputs:
% audioData = Matrix of audio signals that is [nSamples x nSignals]
% numPairs = Number of dissimilar pairs to return
% numHashFunctions = Number of hash functions to use in LSH
% numBands = Number of bands for LSH
%
% Output:
% dissimilarPairs = Nx2 cell array of pairs of dissimilar audio signals
% dissimilarIndices = Nx2 array of indices of dissimilar audio signals
%                   relating to their column number locations in audioData.

    [numSamples, numSignals] = size(audioData);
    maxPairs = floor(numSignals / 2);

    if numPairs > maxPairs
        warning('Requested numPairs (%d) is more than half the number of signals (%d). Setting numPairs to %d.', numPairs, numSignals, maxPairs);
        numPairs = maxPairs;
    end

    %% Wavelet decomposition
    waveletType = 'db4';  

    % Compute maximum wavelet level
    waveletLevel = wmaxlev(numSamples, waveletType);

    % Preallocate memory for efficiency
    coefficients = cell(1, numSignals);

    % Parallel processing for wavelet decomposition
    for i = 1:numSignals
        [coefficients{i}, ~] = wavedec(audioData(:, i), waveletLevel, waveletType);
        disp(['Computed Wavelet Decomposition for signal # ', num2str(i)])
    end

    %% Feature extraction

    % Combine all coefficients into a single matrix
    allCoefficients = cell2mat(coefficients);

    % Quantize coefficients
    threshold = mean(abs(allCoefficients(:)));
    binaryFeatures = allCoefficients > threshold;

    % Truncate the binary features to remove the frequency range of zero-energy
    % imagesc(binaryFeatures)
    binaryFeatures = binaryFeatures(2700:end, :);

    % MinHash and LSH
    disp('Beginning Hashing...')
    hashTable = minhashLSH(binaryFeatures, numHashFunctions);
    disp('Hashing Completed.')

    disp('Beginning Dissimilarity Matching...')
    % Find dissimilar pairs
    [dissimilarPairs, dissimilarIndices] = findDissimilarPairs(hashTable, numBands, numPairs, audioData);
    disp('Dissimilarity Matching Completed.')
end

%% Hashing

function hashTable = minhashLSH(binaryFeatures, numHashFunctions)
    [~, numSignals] = size(binaryFeatures);

    % Compute MinHash signatures
    hashTable = zeros(numHashFunctions, numSignals, 'uint32');

    parfor i = 1:numSignals
        % Generate random hash functions for each parallel worker
        a = randi([1 2^32-1], numHashFunctions, 1, 'uint32');
        b = randi([1 2^32-1], numHashFunctions, 1, 'uint32');

        nonZeroIndices = find(binaryFeatures(:,i));
        if isempty(nonZeroIndices)
            continue;
        end
        for j = 1:numHashFunctions
            hashValues = mod(a(j) * uint32(nonZeroIndices) + b(j), 2^32-1);
            hashTable(j,i) = min(hashValues);
        end
        disp(['Computed Hashes for signal # ', num2str(i)])
    end
end

%% Dissimilar Pair Selection
function [dissimilarPairs, dissimilarIndices] = findDissimilarPairs(hashTable, numBands, numPairs, audioData)
    [numHashFunctions, numSignals] = size(hashTable);
    bandSize = numHashFunctions / numBands;

    % Use a min-heap to keep track of top dissimilar pairs
    % We'll store more pairs than requested to ensure we have enough after filtering
    heapSize = min(numPairs * 3, nchoosek(numSignals, 2));
    dissimilarHeap = zeros(heapSize, 3); % [similarity, index1, index2]
    heapCount = 0;

    disp('Starting to process bands...');
    % Process each band
    for i = 1:numBands
        if mod(i, max(1, floor(numBands/10))) == 0
            disp(['Processing band ' num2str(i) ' of ' num2str(numBands)]);
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
    disp('Finished processing all bands.');

    disp('Sorting and filtering dissimilar pairs...');
    % Convert heap to sorted array
    [~, sortOrder] = sort(dissimilarHeap(1:heapCount, 1));
    sortedPairs = dissimilarHeap(sortOrder, :);

    % Remove duplicate pairs and ensure uniqueness
    uniquePairs = unique(sortedPairs(:, 2:3), 'rows', 'stable');
    dissimilarIndices = uniquePairs(1:min(numPairs, size(uniquePairs, 1)), :);

    disp('Creating cell array of dissimilar audio pairs...');
    % Create cell array of dissimilar audio pairs
    dissimilarPairs = cell(size(dissimilarIndices, 1), 2);
    for i = 1:size(dissimilarIndices, 1)
        if mod(i, max(1, floor(size(dissimilarIndices, 1)/10))) == 0
            disp(['Creating pair ' num2str(i) ' of ' num2str(size(dissimilarIndices, 1))]);
        end
        dissimilarPairs{i, 1} = audioData(:, dissimilarIndices(i, 1));
        dissimilarPairs{i, 2} = audioData(:, dissimilarIndices(i, 2));
    end
    disp('Finished creating dissimilar pairs.');

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