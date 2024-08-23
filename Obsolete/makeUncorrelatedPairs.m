function pairs = makeUncorrelatedPairs(detectionAudio, nPairs, soiHiFreq, Fs, fadeLen)
% detectionAudio = [N x M] matrix containing 'M' audio signals, each of length 'N' samples




%% Compute correlations

% % Combinations of signals as indices
% idx = linspace(1, nSignals, nSignals);
% idxPairs = nchoosek(idx, 2);
% 
% for i = 1:length(idxPairs)
%     xcc = xcorr(detectionAudio(:, idxPairs(i, 1)), ...
%         detectionAudio(:, idxPairs(i, 2)));
%     % mi = mutualInfo(detectionAudio(:, idxPairs(i, 1)), ...
%     %     detectionAudio(:, idxPairs(i, 2)), bins);
% end
% 
% % Sort the signals according to xcc
% [~, xcc_sort_idx] = sort(xcc);
% 
% [~, mi_sort_idx] = sort(mi);
% 
% % Return Pairs
% for i = 1:nPairs
%     pairs{i, 1} = detectionAudio(:, idxPairs(xcc_sort_idx, 1));
%     pairs{i, 2} = detectionAudio(:, idxPairs(xcc_sort_idx, 2));
% end
% 
% 


end
