% Set constants
sourceDir = 'D:\Isolated_Detection_Wavs';
destDir = 'D:\LSH_TEST';
nFiles = 1000;

% Get list of WAV files in the source directory
fileList = dir(fullfile(sourceDir, '*.wav'));

% Check if there are enough files
if length(fileList) < nFiles
    error('Not enough WAV files in the source directory. Found %d, but %d were requested.', length(fileList), nFiles);
end

% Randomly select files
randomIndices = randperm(length(fileList), nFiles);

% Copy files
fprintf('Copying files...\n');
for i = 1:nFiles
    sourceFile = fullfile(sourceDir, fileList(randomIndices(i)).name);
    destFile = fullfile(destDir, fileList(randomIndices(i)).name);
    [status, msg] = copyfile(sourceFile, destFile);
    
    if status
        fprintf('Copied file %d of %d: %s\n', i, nFiles, fileList(randomIndices(i)).name);
    else
        fprintf('Error copying file %s: %s\n', fileList(randomIndices(i)).name, msg);
    end
    
    % Display progress every 100 files
    if mod(i, 100) == 0
        fprintf('Progress: %d%%\n', round(i/nFiles*100));
    end
end

fprintf('Copy operation complete. Copied %d files.\n', nFiles);