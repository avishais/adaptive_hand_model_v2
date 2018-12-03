disp('Loading data...');

IsDiscrete = 0;
Save2Ros = 0;
n = 1000000;

if IsDiscrete
    file = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_discrete.db';
    is_start = 1;
    is_end = 260;%1900; %260;
    D = dlmread(file);
    
    D = dilute(D, n, is_start, is_end);
    
    save('sim_data_discrete.mat', 'D', 'is_start', 'is_end');
    if Save2Ros
        save('/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/sim_data_discrete.mat', 'D', 'is_start', 'is_end');
    end
else
    file = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_cont.db';
    is_start = 1;%250;
    is_end = 200;%750;
    D = dlmread(file);
    
    D = clean(D);
    
%     D = dilute(D, n, is_start, is_end);
    save('sim_data_cont.mat', 'D', 'is_start', 'is_end');
    if Save2Ros
        save('/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/sim_data_cont.mat', 'D', 'is_start', 'is_end');
    end
end

disp('Data saved.');

%% Functions

function Dnew = clean(D)
% Cleans the jumps at/before dropping
disp('Cleaning...');

i = 1; j = 1;
inx = zeros(size(D,1),1);
while i<= size(D,1)
    disp(i);
    if norm(D(i,1:2)-D(i,7:8)) < 2
        inx(j) = i;
        j = j + 1;
    end
    i = i + 1;
end

inx = inx(1:j-1);
Dnew = D(inx,:);

end

function D = dilute(D, n, is_start, is_end)

Dtest = D(is_start:is_end,:);
D = D(is_end+1:end,:);
inx = randperm(size(D,1));
D = D(inx(1:n),:);
D = [Dtest; D];

end