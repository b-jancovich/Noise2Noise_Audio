function delete_silent_wav_files(folder_path)
    % Set silence threshold
    silence_threshold = 1e-7;

    % Get all .wav files in the specified folder
    file_list = dir(fullfile(folder_path, '*.wav'));
    
    % Initialize progress bar
    waitbar_h = waitbar(0, 'Processing files...');
    
    % Initialize counters
    total_files = length(file_list);
    processed_count = 0;
    deleted_count = 0;
    error_count = 0;
    
    % Process each file
    for i = 1:total_files
        file_path = fullfile(folder_path, file_list(i).name);
        
        try
            % Read WAV file
            [y, fs] = audioread(file_path);
            
            % DC center the signal
            y_centered = y - mean(y);
            
            % Calculate RMS of the centered signal
            rms_level = sqrt(mean(y_centered.^2));
            
            % Check if the file is silent based on RMS threshold
            if rms_level < silence_threshold
                % Delete the file
                delete(file_path);
                fprintf('Deleted silent file: %s (RMS: %.9f)\n', file_list(i).name, rms_level);
                deleted_count = deleted_count + 1;
            else
                fprintf('Kept non-silent file: %s (RMS: %.9f)\n', file_list(i).name, rms_level);
            end
            processed_count = processed_count + 1;
        catch ME
            warning('Error processing file %s: %s', file_list(i).name, ME.message);
            error_count = error_count + 1;
        end
        
        % Update progress bar
        waitbar(i / total_files, waitbar_h);
    end
    
    % Close progress bar
    close(waitbar_h);
    
    % Print summary
    fprintf('\nProcessing complete.\n');
    fprintf('Total files: %d\n', total_files);
    fprintf('Processed: %d\n', processed_count);
    fprintf('Deleted (silent): %d\n', deleted_count);
    fprintf('Kept (non-silent): %d\n', processed_count - deleted_count);
    fprintf('Errors: %d\n', error_count);
end