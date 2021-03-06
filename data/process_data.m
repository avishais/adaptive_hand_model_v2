disp('Loading data...');

IsDiscrete = 1;
Save2Ros = 0;
n = 1000000;

if IsDiscrete
    file = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/transition_data_discrete_v5.db';
    is_start = 1;
    is_end = 277;%340;%1900; %260;
    D = dlmread(file);
    
    D = D(:,[1:4 7:8 9:12]); % Position and load
%     D = D(:,[1:2 5:6 7:8 9:10 13:14]); % Position and velocity
    
    windowSize = 20; 
    b = (1/windowSize)*ones(1,windowSize);
%     D(:,3:4) = filter(b, 1, D(:,3:4));
%     D(:,9:10) = filter(b, 1, D(:,9:10));
%     D(:,5:6) = filter(b, 1, D(:,5:6));
%     D(:,13:14) = filter(b, 1, D(:,13:14));
    
    D = clean(D);
    
%     D = dilute(D, n, is_start, is_end);
    
%     save('sim_data_discrete_v5_vel.mat', 'D', 'is_start', 'is_end');
    if Save2Ros
        save('/home/pracsys/catkin_ws/src/beliefspaceplanning/gpup_gp_node/data/sim_data_discrete_v5_vel.mat', 'D', 'is_start', 'is_end');
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

plot(D(:,1),D(:,2),'.');

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
