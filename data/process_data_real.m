file = 'Ce_20_5.db';

is_start = 1;
is_end = 260;
D = dlmread(file);

save('real_data_discrete.mat', 'D', 'is_start', 'is_end');