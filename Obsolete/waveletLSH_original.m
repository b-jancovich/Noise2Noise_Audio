function [dissimilarPairs, dissimilarIndices] = waveletLSH_original(audioData, numPairs, numHashFunctions, numBands)
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
    hashTable = minhashLSH(binaryFeatures, numHashFunctions);

    % Find dissimilar pairs
    [dissimilarPairs, dissimilarIndices] = findDissimilarPairs(hashTable, numBands, numPairs, audioData);
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
    end
end

%% Pairing based on Dissimilarity ("Process all signals at once" version) 

function [dissimilarPairs, dissimilarIndices] = findDissimilarPairs(hashTable, numBands, numPairs, audioData)
    [numHashFunctions, numSignals] = size(hashTable);
    bandSize = numHashFunctions / numBands;

    % Initialize similarity count matrix
    similarityCount = zeros(numSignals);

    % LSH banding technique
    for i = 1:numBands
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
                        similarityCount(members(j), members(k)) = similarityCount(members(j), members(k)) + 1;
                        similarityCount(members(k), members(j)) = similarityCount(members(k), members(j)) + 1;
                    end
                end
            end
        end
    end

    % Find the least similar pairs
    [~, sortedIndices] = sort(similarityCount(:));
    [rows, cols] = ind2sub(size(similarityCount), sortedIndices);

    % Remove duplicate pairs and self-pairs
    validPairs = rows < cols;
    dissimilarIndices = [rows(validPairs), cols(validPairs)];

    % Ensure unique pairs
    dissimilarIndices = unique(dissimilarIndices, 'rows', 'stable');

    % Select the required number of pairs
    dissimilarIndices = dissimilarIndices(1:min(numPairs, size(dissimilarIndices, 1)), :);

    % Create cell array of dissimilar audio pairs
    dissimilarPairs = cell(size(dissimilarIndices, 1), 2);
    for i = 1:size(dissimilarIndices, 1)
        dissimilarPairs{i, 1} = audioData(:, dissimilarIndices(i, 1));
        dissimilarPairs{i, 2} = audioData(:, dissimilarIndices(i, 2));
    end
end