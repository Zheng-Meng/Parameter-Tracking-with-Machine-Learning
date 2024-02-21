function [data_output] = func_step_average(data, step)

% data should be one dimensional, and the step shuold be integer.

data_output_length = round(length(data) / step);
data_output = zeros(1, data_output_length);

for idx = 1:data_output_length-1
    data_output(idx) = mean(data(1 + step * (idx-1) : step * idx ));
end

data_output(end) = mean(data(1 + step * idx : end ));

end