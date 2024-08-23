function mi = mutualInfo(x, y, bins)
    % Calculates the mutual information in two signals
    
    % Inputs:
    %   x: First 1D vector
    %   y: Second 1D vector
    %   bins: Number of bins for discretizing continuous variables (default: 20)
    %
    % Output:
    %   mi: Mutual information between x and y

    % Set default number of bins if not provided
    if nargin < 3
        bins = 20;
    end

    % Ensure inputs are column vectors
    x = x(:);
    y = y(:);

    if length(x) ~= length(y)
        error('Input vectors must have the same length');
    end

    % Calculate joint histogram
    [joint_hist, ~] = hist3([x y], [bins bins]);

    % Calculate marginal histograms
    x_hist = sum(joint_hist, 2);
    y_hist = sum(joint_hist, 1);

    % Calculate entropies
    % x_entropy = entropy(x_hist);
    % y_entropy = entropy(y_hist);
    % joint_entropy = entropy(joint_hist(:));
    x_entropy = approximateEntropy(x_hist);
    y_entropy = approximateEntropy(y_hist);
    joint_entropy = approximateEntropy(joint_hist(:));

    % Calculate mutual information
    mi = x_entropy + y_entropy - joint_entropy;
end

% function e = entropy(p)
%     % Calculate the entropy of a probability distribution
%     p = p(p > 0) / sum(p);  % Normalize and remove zeros
%     e = -sum(p .* log2(p));
% end
