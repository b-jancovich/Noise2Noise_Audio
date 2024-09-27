function similarity = signalSimilarity(sig1, sig2, Fs, varargin)
    % This function measures similarity between audio signals containing 
    % time- and frequency-shifted events mixed with other sounds, using 2D 
    % Cross-Correlation of spectrograms.
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
