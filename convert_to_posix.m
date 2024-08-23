function target_posix = convert_to_posix(target_date_time)
% Converts a date and time string to a posix time without using datetime(),
% which is slow. Written to be as fast as possible.
% Millisecond precision. 
% 
% NOTE: ONLY WORKS FOR YEARS BETWEEN 2000 and 2999.
%
% Inputs:
% target_date_time = the input date time string in format 'yyMMdd-HHmmss'
%
% Outputs: 
% target_posix = target_date_time converted to posix time.

    % Extract date and time components - Do not use str2double - too slow
    year = sscanf(sprintf(' %s', target_date_time(1:2)),'%f',[1,Inf]) + 2000;
    month = sscanf(sprintf(' %s', target_date_time(3:4)),'%f',[1,Inf]);
    day = sscanf(sprintf(' %s', target_date_time(5:6)),'%f',[1,Inf]);
    hour = sscanf(sprintf(' %s', target_date_time(8:9)),'%f',[1,Inf]);
    minute = sscanf(sprintf(' %s', target_date_time(10:11)),'%f',[1,Inf]);
    second = sscanf(sprintf(' %s', target_date_time(12:13)),'%f',[1,Inf]);

    % Calculate days since March 1, 1900 (to simplify leap year calculations)
    if month <= 2
        month = month + 12;
        year = year - 1;
    end
    days = floor(365.25 * (year + 4716)) + floor(30.6001 * (month + 1)) + day - 2440588;

    % Calculate seconds since epoch
    target_posix = days * 86400 + hour * 3600 + minute * 60 + second;

    % Adjust for leap seconds
    leap_seconds = get_leap_seconds(year, month, day);
    target_posix = target_posix - leap_seconds;
end

function leap_seconds = get_leap_seconds(year, month, day)
    % Define leap second table (up to 2024, last leap second was in 2016)
    leap_second_table = [
        1972, 7, 1;
        1973, 1, 1;
        1974, 1, 1;
        1975, 1, 1;
        1976, 1, 1;
        1977, 1, 1;
        1978, 1, 1;
        1979, 1, 1;
        1980, 1, 1;
        1981, 7, 1;
        1982, 7, 1;
        1983, 7, 1;
        1985, 7, 1;
        1988, 1, 1;
        1990, 1, 1;
        1991, 1, 1;
        1992, 7, 1;
        1993, 7, 1;
        1994, 7, 1;
        1996, 1, 1;
        1997, 7, 1;
        1999, 1, 1;
        2006, 1, 1;
        2009, 1, 1;
        2012, 7, 1;
        2015, 7, 1;
        2017, 1, 1
    ];
    
    % Count leap seconds up to the given date
    if year < 2024
        leap_seconds = sum(leap_second_table(:,1) < year | ...
                          (leap_second_table(:,1) == year & ...
                          (leap_second_table(:,2) < month | ...
                          (leap_second_table(:,2) == month & ...
                           leap_second_table(:,3) <= day))));
    else
        % For dates in 2024 and beyond, use the last known leap second count
        % and add a warning
        leap_seconds = size(leap_second_table, 1);
        warning('Leap seconds beyond 2024 are not known. Using last known count: %d', leap_seconds);
        sprintf('Requested year is: %d', year)
    end
end