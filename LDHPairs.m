function dissimilarPairs = LDHPairs(ads, soiHiFreq, soiLoFreq, ...
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
    [b, a] = createBandstopFilter(soiLoFreq, soiHiFreq, Fs);

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
            nFFT, numHashes, allHashTables, hashTables, bandSize, fileIDs,...
            featuresToUse);
        
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

function [b, a] = createBandstopFilter(soiLoFreq, soiHiFreq, Fs)
    Wn = [soiLoFreq, soiHiFreq] / (Fs / 2);
    [b, a] = butter(8, Wn, 'stop');
end

function allHashTables = initializeHashTables(hashTables)
    allHashTables = cell(1, hashTables);
    for i = 1:hashTables
        allHashTables{i} = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    end
end

function processBatch(ads, windowFull, b, a, Fs, windowLen, overlapPercent, ...
    nFFT, numHashes, allHashTables, hashTables, bandSize, fileIDs, featuresToUse)

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
        fileID = fileIDs(fileIndex);
        updateHashTables(signature, fileID, allHashTables, hashTables, bandSize);
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

function updateHashTables(signature, fileID, allHashTables, hashTables, bandSize)
    for j = 1:hashTables
        bandStart = (j-1)*bandSize + 1;
        bandEnd = j*bandSize;
        band = signature(bandStart:bandEnd);
        bandHash = sum(band);

        if ~isKey(allHashTables{j}, bandHash)
            allHashTables{j}(bandHash) = fileID;
        else
            allHashTables{j}(bandHash) = [allHashTables{j}(bandHash), fileID];
        end
    end
end

function dissimilarPairs = findDissimilarPairs(allHashTables, nSignals, fileIDs)
    dissimilarityMatrix = sparse(nSignals, nSignals);

    for i = 1:length(allHashTables)
        table = allHashTables{i};
        keys = cell2mat(table.keys);

        for j = 1:length(keys)
            bucket = table(keys(j));
            if length(bucket) > 1
                combinations = nchoosek(bucket, 2);
                for k = 1:size(combinations, 1)
                    idx1 = find(fileIDs == combinations(k,1));
                    idx2 = find(fileIDs == combinations(k,2));
                    dissimilarityMatrix(idx1, idx2) = dissimilarityMatrix(idx1, idx2) + 1;
                    dissimilarityMatrix(idx2, idx1) = dissimilarityMatrix(idx2, idx1) + 1;
                end
            end
        end
        fprintf('Updated dissimilarity matrix with hash table %d of %d\n', i, length(allHashTables))
    end

    disp('Finished updating hash tables.')
    disp('Beginning matrix sorting...')

    % Convert similarity to dissimilarity
    dissimilarityMatrix = length(allHashTables) - dissimilarityMatrix;
    
    % Set diagonal to -1 to ensure we don't pair a signal with itself
    dissimilarityMatrix(logical(eye(size(dissimilarityMatrix)))) = -1;
    
    dissimilarPairs = [];
    availableSignals = true(1, nSignals);
    pairsCount = 1;
    while sum(availableSignals) >= 2
        [maxVal, maxIdx] = max(dissimilarityMatrix(:));
        if maxVal <= 0
            break;
        end
        
        [row, col] = ind2sub(size(dissimilarityMatrix), maxIdx);
        
        dissimilarPairs = [dissimilarPairs; fileIDs(row), fileIDs(col)];
        availableSignals(row) = false;
        availableSignals(col) = false;
        
        % Set all entries for these signals to -1
        dissimilarityMatrix(row, :) = -1;
        dissimilarityMatrix(:, row) = -1;
        dissimilarityMatrix(col, :) = -1;
        dissimilarityMatrix(:, col) = -1;

        fprintf('Generated pair# %d).\n', pairsCount)
        pairsCount = pairsCount+1;
    end
end