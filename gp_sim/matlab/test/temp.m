load('sim_data_discrete_v6_d6_m1.mat')
plot(D(:,1),D(:,2),'.b')
hold on
load('sim_data_discrete_v8_d4_m1.mat')
plot(D(:,1),D(:,2),'.r')

load('test_v6_d6_m1.mat')
plot(S(:,1),S(:,2),'-k')

load('test_v8_d4_m1.mat')
plot(S(:,1),S(:,2),'-y')


hold off