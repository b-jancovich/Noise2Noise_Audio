function featuresOut = selectAudioFeatures(ads, windowLen, overlapPercent, nFFT, sampleSize, p)
    % Which Features to use?
    % This function is designed to test which audio features best explain the
    % variance between audio files in a dataset. The strength of each feature's 
    % explanatory power is measured by Minimum Redundancy Maximum Relevance (MRMR).
    %
    % Inputs:
    %   ads            - Audio datastore object created with batchAudioDatastore()
    %   windowLen      - Length of analysis window in samples 
    %   overlapPercent - Percentage of overlap between windows 
    %   nFFT           - Number of FFT points
    %   sampleSize     - Number of audio files to sample from the datastore 
    %
    % Outputs:
    %   features       - Struct containing the following fields:
    %     .names           - Cell array of feature names that explain 90% of the difference
    %     .scores          - Vector of feature scores for the selected features
    %     .numFeatures     - Number of features that explain 90% of the difference
    %
    % Ben Jancovich, 2024 (modified)
    % Centre for Marine Science and Innovation
    % School of Biological, Earth and Environmental Sciences
    % University of New South Wales, Sydney, Australia
    %
    %% Begin:

    % Set read size on AudioDatastore
    ads.MiniBatchSize = sampleSize;

    % Read a single audio file to get signal info
    [~, sig0_info] = readSingle(ads);
    Fs = sig0_info.SampleRate;
    reset(ads)

    % Set up Audio Feature Extractor
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

    % Set extractor to get all features
    afe_info = info(afe, "all");
    featureSwitches = fields(afe_info);
    cellfun(@(x)afe.set(x,true),featureSwitches)

    % Get some random signals from the ADS & extract features
    randomIndices = randperm(ads.NumObservations, sampleSize);
    sampleLabels = ads.Labels(randomIndices);
    audioData = cell(sampleSize, 1);
    features = cell(sampleSize, 1);
    % Feature Extraction:
    for i = 1:sampleSize
        % Randomly index a file in the datastore
        ads.CurrentFileIndex = randomIndices(i);
        % Read that single audio file from datastore 
        [audioData{i}, ~] = ads.readSingle();
        % Extract features from file
        features{i} = extract(afe, audioData{i});
        % Normalize the features by their mean and standard deviation
        features{i} = (features{i} - mean(features{i},1))./std(features{i},[],1);
        fprintf('Extracted features for randomly selected signal # %d\n', i)
    end

    % Get feature names and column counts
    outputMap = info(afe);

    % Replicate file-level labels for one-to-one correspondence with features
    N = cellfun(@(x)size(x, 1), features);
    T = repelem(sampleLabels, N);

    % Vertically concatenate features into a single matrix
    X = cat(1,features{:});

    % Use fscmrmr to rank features using the minimum-redundancy/maximum-
    % relevance (MRMR) algorithm. The MRMR is a sequential algorithm that 
    % finds an optimal set of features that is mutually and maximally 
    % dissimilar and can represent the response variable effectively.

    % Set random number generator for reproducibility
    disp('Running MRMR Algorithm...')
    rng("default")
    [~, featureSelectionScores] = fscmrmr(X,T);
    disp('MRMR Complete...')

    % Extract feature names & arrange according to number of columns
    % produced by each feature
    featureNames = uniqueFeatureName(outputMap);

    % Group features with the same base name and sum their scores
    [uniqueBaseNames, ~, baseNameIndices] = unique(regexprep(featureNames, '\d+$', ''));
    groupedScores = accumarray(baseNameIndices, featureSelectionScores);

    % Calculate total score
    totalScore = sum(groupedScores);
    
    % Normalize the scores to get percentages
    normalizedScores = groupedScores / totalScore;
    
    % Sort the scores in descending order (keep track of indices)
    [sortedScores, sortedIdx] = sort(normalizedScores, 'descend');
    
    % Compute cumulative sum of sorted scores
    cumulativeScores = cumsum(sortedScores);
    
    % Find the minimum number of features m to explain p% of the variance
    m = find(cumulativeScores >= p/100, 1, 'first');
    
    % The indices of the selected features
    selectedFeaturesIdx = sortedIdx(1:m);
    
    % The selected features and their scores
    selectedFeaturesScores = sortedScores(1:m);
    
    % Get the names of the selected features
    selectedFeatureNames = uniqueBaseNames(selectedFeaturesIdx);

    % Plot the scores of the selected features on a bar plot in descending
    % score order
    figure;
    bar(selectedFeaturesScores);
    title('Selected Audio Features Scores (Grouped)');
    xlabel('Feature');
    ylabel('Feature Score');
    xticks(1:m);
    xticklabels(selectedFeatureNames);
    xtickangle(45);
    grid on;
    
    % Add percentage labels on top of each bar
    text(1:length(selectedFeaturesScores), selectedFeaturesScores, ...
         num2str(selectedFeaturesScores*100,'%.1f%%'), ...
         'VerticalAlignment','bottom', 'HorizontalAlignment','center');
   
    % Build output struct:
    featuresOut = struct();
    featuresOut.featureNames = selectedFeatureNames;
    featuresOut.featureScores = selectedFeaturesScores;
end

function c = uniqueFeatureName(afeInfo)
    %UNIQUEFEATURENAME Create unique feature names
    %c = uniqueFeatureName(featureInfo) creates a unique set of feature names
    %for each element of each feature described in the afeInfo struct. The
    %afeInfo struct is returned by the info object function of
    %audioFeatureExtractor.
    a = repelem(fields(afeInfo),structfun(@numel,afeInfo));
    b = matlab.lang.makeUniqueStrings(a);
    d = find(endsWith(b,"_1"));
    c = strrep(b,"_","");
    c(d-1) = strcat(c(d-1),"0");
end