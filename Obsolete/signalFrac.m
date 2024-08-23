function frac = signalFrac(sig1, sig2, Fs)
    % Calculate Frequency response assurance criterion between two audio
    % signals.

    % sig1, sig2 = input audio signals
    % fs = sampling frequency
    
    % Ensure signals are the same length
    minLength = min(length(signal1), length(signal2));
    signal1 = signal1(1:minLength);
    signal2 = signal2(1:minLength);
    
    % Calculate FFT
    fft1 = fft(signal1);
    fft2 = fft(signal2);
    
    % Use only the first half of the FFT (up to Nyquist frequency)
    n = length(fft1);
    fft1 = fft1(1:floor(n/2)+1);
    fft2 = fft2(1:floor(n/2)+1);
    
    % Calculate magnitude spectra
    mag1 = abs(fft1);
    mag2 = abs(fft2);
    
    % Calculate FRAC
    numerator = abs(sum(mag1 .* conj(mag2)))^2;
    denominator = sum(abs(mag1).^2) * sum(abs(mag2).^2);
    frac = numerator / denominator;
    
    % FRAC value ranges from 0 to 1, where 1 indicates perfect similarity
end