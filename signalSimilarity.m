function similarity = signalSimilarity(sig1, sig2, Fs, varargin)
    % This function measures similarity between audio signals containing 
    % time-shifted events mixed with other sounds, using multiple metrics:
    % 1. 2D Cross-Correlation of spectrograms
    % 2. Mel-Frequency Cepstral Coefficients (MFCC) comparison
    % 3. Spectral Flux correlation
    % 4. RMS energy envelope correlation
    % It uses Dynamic Time Warping to account for time offset in similar events.
    %
    % Inputs:
    %   sig1, sig2 - Signals to measure. Assumed to be identical dimensions and
    %                pre-processed (i.e. DC Filtered, Normalised, windowed start/end)
    %   Fs         - Sampling frequency (Hz)
    %   varargin   - Optional parameter-value pairs:
    %                'WindowSize' - STFT window size (default: 250)
    %                'Overlap'    - STFT window overlap (default: 200)
    %
    % Outputs:
    %   similarity - The maximum value of the 2D cross correlation of the
    %   two signal's spectrograms
    %
    % 2024 Ben Jancovich
    % University of New South Wales

    %% Input validation
    if ~isequal(size(sig1), size(sig2))
        error('Input signals must have the same dimensions.');
    end
    if ~isscalar(Fs) || Fs <= 0
        error('Sampling frequency must be a positive scalar.');
    end

    %% Parse optional inputs
    p = inputParser;
    addParameter(p, 'WindowSize', 250, @(x) isscalar(x) && x > 0);
    addParameter(p, 'Overlap', 200, @(x) isscalar(x) && x >= 0);
    parse(p, varargin{:});

    nWindow = p.Results.WindowSize;
    noverlap = p.Results.Overlap;

    %% Define constants
    nfft = 512;
 
    %% Normalize and DC filter
    sig1 = (sig1 - mean(sig1)) / (max(sig1) - mean(sig1));
    sig2 = (sig2 - mean(sig2)) / (max(sig2) - mean(sig2));

    %% Compute spectrograms
    [S1, ~, ~] = spectrogram(sig1, nWindow, noverlap, nfft, Fs, 'yaxis');
    [S2, ~, ~] = spectrogram(sig2, nWindow, noverlap, nfft, Fs, 'yaxis');

    %% 2D Cross Correlation
    xcorr2D = xcorr2(abs(S1), abs(S2));
    similarity = max(abs(xcorr2D(:)));

end

% Old version
% % % function varargout = signalSimilarity(sig1, sig2, Fs)
% % %     % This function measures similarity between audio signals containing 
% % %     % time-shifted events mixed with other sounds, using multiple metrics:
% % %     % 1. 2D Cross-Correlation of spectrograms
% % %     % 2. Mel-Frequency Cepstral Coefficients (MFCC) comparison
% % %     % 3. Spectral Flux correlation
% % %     % 4. RMS energy envelope correlation
% % %     % It uses Dynamic Time Warping to account for time offset in similar
% % %     % events.
% % %     %
% % %     % Inputs:
% % %     %   sig1, sig2 - Signals to measure. Assumed to be identical dimensions and
% % %     %                pre-processed (i.e. DC Filtered, Normalised, windowed start/end)
% % %     %   Fs         - Sampling frequency (Hz)
% % %     %
% % %     % Outputs:
% % %     %   similarity - Similarity score, 0 being maximally dissimilar, 1 being
% % %     %                maximally similar.
% % %     %   rawMetrics - (Optional) Raw similarity metrics before combination
% % %     %                [maxXCorr, mfccSim, fluxCorr, rmsCorr]
% % %     %
% % %     % Constants:
% % %     %   weights    - Weights for combining metrics [maxXCorr, mfccSim, fluxCorr, rmsCorr]
% % %     %   nWindow    - STFT window size (samples)
% % %     %   noverlap   - STFT window overlap (samples)
% % %     %   nfft       - Number of FFT points
% % %     %   numCoeffs  - Number of MFCC coefficients
% % %     %   hopLength  - Hop length for envelope calculation (samples)
% % %     %   alpha      - Non-linear transformation factor - increases
% % %     %   contribution of high raw simiilarity scores on the combine score.
% % %     %
% % %     % 2024 Ben Jancovich
% % %     % University of New South Wales
% % %     %% Define constants: 
% % %     nWindow = 250; 
% % %     noverlap = 200;
% % %     nfft = 512;
% % %     numCoeffs = 11;
% % %     hopLength = 25;
% % %     wts = [0.2, 0.4, 0.2, 0.2];
% % %     std_metrics = [0.020855617306641, 0.037285075349270, 0.016963336302996, 0.018046960260193];
% % % 
% % %     % Compute spectrograms once
% % %     [S1, ~, ~] = spectrogram(sig1, nWindow, noverlap, nfft, Fs, 'yaxis');
% % %     S2 = spectrogram(sig2, nWindow, noverlap, nfft, Fs, 'yaxis');
% % % 
% % %     % 2D Cross Correlation
% % %     xcorr2D = xcorr2(abs(S1), abs(S2));
% % %     maxXCorr = max(xcorr2D(:)) / (norm(S1(:)) * norm(S2(:)));
% % % 
% % %     % MFCC
% % %     numFilters = numCoeffs + 2;
% % %     melPoints = linspace(hz2mel(1), hz2mel((Fs/2)-1), numFilters);
% % %     customBandEdges = mel2hz(melPoints);
% % % 
% % %     mfcc1 = mfcc(sig1, Fs, 'NumCoeffs', numCoeffs, 'BandEdges', customBandEdges);
% % %     mfcc2 = mfcc(sig2, Fs, 'NumCoeffs', numCoeffs, 'BandEdges', customBandEdges);
% % % 
% % %     mfccDist = sum(min(pdist2(mfcc1', mfcc2', 'euclidean'), [], 2));
% % %     mfccSim = 1 / (1 + mfccDist / (size(mfcc1, 2) * numCoeffs));
% % % 
% % %     % Spectral Flux Correlation
% % %     flux1 = sum(abs(diff(abs(S1), 1, 2)), 1);
% % %     flux2 = sum(abs(diff(abs(S2), 1, 2)), 1);
% % %     fluxCorr = max(xcorr(flux1, flux2)) / (norm(flux1) * norm(flux2));
% % % 
% % %     % RMS Energy Envelope Correlation
% % %     rms1 = envelope(sig1, hopLength, 'rms');
% % %     rms2 = envelope(sig2, hopLength, 'rms');
% % %     rmsCorr = max(xcorr(rms1, rms2)) / (norm(rms1) * norm(rms2));
% % % 
% % %     % Final Similarity Score
% % %     rawMetrics = [maxXCorr, mfccSim, fluxCorr, rmsCorr];
% % %     normalizedMetrics = rawMetrics ./ std_metrics;
% % %     weightedSum = wts * normalizedMetrics';
% % % 
% % %     expectedMax = sum(wts .* (3 ./ std_metrics));
% % %     similarity = max(0, min(1, weightedSum / expectedMax));
% % % 
% % %     if nargout == 1
% % %         varargout = {similarity};
% % %     elseif nargout == 2
% % %         varargout = {similarity, rawMetrics};
% % %     end
% % % end
% % % 
% % % % function varargout = signalSimilarity(sig1, sig2, Fs)
% % % %     % This function measures similarity between audio signals containing 
% % % %     % time-shifted events mixed with other sounds, using multiple metrics:
% % % %     % 1. 2D Cross-Correlation of spectrograms
% % % %     % 2. Mel-Frequency Cepstral Coefficients (MFCC) comparison
% % % %     % 3. Spectral Flux correlation
% % % %     % 4. RMS energy envelope correlation
% % % %     % It uses Dynamic Time Warping to account for time offset in similar
% % % %     % events.
% % % %     %
% % % %     % Inputs:
% % % %     %   sig1, sig2 - Signals to measure. Assumed to be identical dimensions and
% % % %     %                pre-processed (i.e. DC Filtered, Normalised, windowed start/end)
% % % %     %   Fs         - Sampling frequency (Hz)
% % % %     %
% % % %     % Outputs:
% % % %     %   similarity - Similarity score, 0 being maximally dissimilar, 1 being
% % % %     %                maximally similar.
% % % %     %   rawMetrics - (Optional) Raw similarity metrics before combination
% % % %     %                [maxXCorr, mfccSim, fluxCorr, rmsCorr]
% % % %     %
% % % %     % Constants:
% % % %     %   weights    - Weights for combining metrics [maxXCorr, mfccSim, fluxCorr, rmsCorr]
% % % %     %   nWindow    - STFT window size (samples)
% % % %     %   noverlap   - STFT window overlap (samples)
% % % %     %   nfft       - Number of FFT points
% % % %     %   numCoeffs  - Number of MFCC coefficients
% % % %     %   hopLength  - Hop length for envelope calculation (samples)
% % % %     %   alpha      - Non-linear transformation factor - increases
% % % %     %   contribution of high raw simiilarity scores on the combine score.
% % % %     %
% % % %     % 2024 Ben Jancovich
% % % %     % University of New South Wales
% % % %     % Set Weightings to combine maxXCorr, mfccSim, fluxCorr, rmsCorr
% % % %     %
% % % %     %% Define constants: 
% % % %     % STFT
% % % %     nWindow = 250; 
% % % %     noverlap = 200;
% % % %     nfft = 512;
% % % % 
% % % %     % MFCC
% % % %     numCoeffs = 11;
% % % % 
% % % %     % DTW
% % % %     hopLength = 25;
% % % % 
% % % %     % Score Combiner
% % % %     % Weightings of each metric - must sum to equal 
% % % %     % [maxXCorr, mfccSim, fluxCorr, rmsCorr]
% % % %     wts = [0.2, 0.4, 0.2, 0.2];
% % % %     % Standard deviations of raw metrics from a pilot run of 1000 signals
% % % %     % [maxXCorr, mfccSim, fluxCorr, rmsCorr]
% % % %     std_metrics = [0.020855617306641, 0.037285075349270,...
% % % %         0.016963336302996, 0.018046960260193];
% % % % 
% % % %     %% 2D Cross Correlation:
% % % % 
% % % %     % Compute spectrograms
% % % %     S1 = spectrogram(sig1, nWindow, noverlap, nfft, Fs, 'yaxis');
% % % %     S2 = spectrogram(sig2, nWindow, noverlap, nfft, Fs, 'yaxis');
% % % % 
% % % %     % Compute 2D cross-correlation of spectrograms, ignoring phase.
% % % %     xcorr2D = xcorr2(abs(S1), abs(S2));
% % % % 
% % % %     % Convert to a single value by normalizing the CC peak by the 
% % % %     % the energy of the input signals:
% % % %     maxXCorr = max(xcorr2D(:)) / (norm(S1(:)) * norm(S2(:)));
% % % % 
% % % %     %% Mel-Frequency Cepstral Coefficients 
% % % % 
% % % %     % Define the number of mel filters (numCoeffs + 2 for edges)
% % % %     numFilters = numCoeffs + 2;
% % % % 
% % % %     % Calculate the lower and upper frequencies
% % % %     lowerFreq = 1;
% % % %     upperFreq = (Fs / 2) -1;
% % % % 
% % % %     % Convert the frequency range to the Mel scale
% % % %     lowerMel = hz2mel(lowerFreq);
% % % %     upperMel = hz2mel(upperFreq);
% % % % 
% % % %     % Generate evenly spaced points in the Mel scale
% % % %     melPoints = linspace(lowerMel, upperMel, numFilters);
% % % % 
% % % %     % Convert the Mel points back to the frequency scale
% % % %     customBandEdges = mel2hz(melPoints);
% % % % 
% % % %     % Calculate MFCCs with specified number of coefficients and custom band edges
% % % %     mfcc1 = mfcc(sig1, Fs, 'NumCoeffs', numCoeffs, 'BandEdges', customBandEdges);
% % % %     mfcc2 = mfcc(sig2, Fs, 'NumCoeffs', numCoeffs, 'BandEdges', customBandEdges);
% % % % 
% % % %     % Compute the DTW-like distance between MFCCs
% % % %     mfccDist = sum(min(pdist2(mfcc1', mfcc2', 'euclidean'), [], 2));
% % % %     mfccSim = 1 / (1 + mfccDist / (size(mfcc1, 2) * numCoeffs));
% % % % 
% % % %     %% Spectral Flux Correlation
% % % % 
% % % %     flux1 = sum(abs(diff(abs(S1), 1, 2)), 1);
% % % %     flux2 = sum(abs(diff(abs(S2), 1, 2)), 1);
% % % %     fluxCorr = max(xcorr(flux1, flux2)) / (norm(flux1) * norm(flux2));
% % % % 
% % % %     %% RMS Energy Envelope Correlation
% % % % 
% % % %     % Compute RMS energy similarity with time shift
% % % %     rms1 = envelope(sig1, hopLength, 'rms');
% % % %     rms2 = envelope(sig2, hopLength, 'rms');
% % % %     rmsCorr = max(xcorr(rms1, rms2)) / (norm(rms1) * norm(rms2));
% % % % 
% % % %     %% Final Similarity Score
% % % % 
% % % %     % Combine metrics
% % % %     rawMetrics = [maxXCorr, mfccSim, fluxCorr, rmsCorr];
% % % % 
% % % %     % Normalize the raw metrics by their standard deviations
% % % %     normalizedMetrics = rawMetrics ./ std_metrics;
% % % % 
% % % %     % Compute weighted sum of scaled metrics
% % % %     weightedSum  = wts * normalizedMetrics';
% % % % 
% % % %     expectedMin = 0;  % Theoretical minimum (perfect dissimilarity)
% % % %     expectedMax = sum(wts .* (3 ./ std_metrics));  % Assume max is 3 standard deviations above mean
% % % %     similarity = (weightedSum - expectedMin) / (expectedMax - expectedMin);
% % % % 
% % % %     % Ensure the similarity score is within [0, 1]
% % % %     similarity = max(0, min(1, similarity));
% % % % 
% % % %     % Optionally, return the raw metrics as well as the combined score.
% % % %     if nargout == 1
% % % %         varargout = {similarity};
% % % %     elseif nargout == 2
% % % %         varargout = {similarity, rawMetrics};
% % % %     end
% % % % end
% % % % 
