clear all
warning('off','all')

px2mm = 1;

GP = gp_class(1, 1, 0);

A = GP.A;

s = GP.Xtest(1,1:4);
I = GP.I;

Q = load('../../data/real_data_discrete.mat');
X = Q.D;
n = size(X,1);

%%
for k = 15
    %%
    N = 1000;
    M = 600;
    
    s = [33.4020000000000,-325.930000000000,52,-198];%X(randi(n), 1:4);
    
    Y = cell(M,1);
    
    Y{1} = repmat(s, N, 1);
    m(1,:) = mean(Y{1});
    
    for i = 2 : M
        disp([k i]);
        Y{i} = zeros(N, 4);
        
        if i <= 150
            a = [-0.2, 0.2];
        end
        if i > 150 && i <= 300
            a = [-0.2, -0.2];
        end
        if i > 300 && i <= 450
            a = [0.2, -0.2];
        end
        if i > 450
            a = [0.2, 0.2];
        end
        
        
%         a = [0.200000000000000,0.200000000000000];%A(randi(8),:);
        for j = 1:N
            Y{i}(j,:) = GP.propagate(Y{i-1}(j,:), a);
        end
        m(i,:) = mean(Y{i});
        
    end
    
    save(['./propagation_test_results/propagation_data_' num2str(k) '.mat'],'Y', 'm', 'N', 'M');
end
%%
load(['./propagation_test_results/propagation_data_' num2str(k) '.mat']);

for i = 1:M
    figure(1)
    clf
    hold on
    plot(X(:,1),-X(:,2),'.c');
    plot(Y{i}(:,1),-Y{i}(:,2),'.r');
    plot(m(1:i,1),-m(1:i,2),'-k');
    plot(m(i,1),-m(i,2),'ok','markerfacecolor','b');
    hold off
    axis([-171.0700  293.8600 176.4200 430.5800 ]);
    drawnow;
end
drawnow;
hold off