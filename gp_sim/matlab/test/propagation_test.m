clear all
warning('off','all')

px2mm = 1;

GP = gp_class(1, 1, 0);

A = [-1,-1;-1,0;-1,1;0,-1;0,1;1,-1;1,0;1,1];

s = GP.Xtest(1,1:4);
I = GP.I;

%%
N = 100;
M = 100;

Y = cell(M,1);

Y{1} = repmat(s, N, 1);
m(1,:) = mean(Y{1});

figure(1)
clf
hold on
plot(Y{1}(:,1),Y{1}(:,2),'.r');
drawnow;

for i = 2:M
    Y{i} = zeros(N,4);
    
    a = A(randi(8),:);
    for j = 1:N
        Y{i}(j,:) = GP.propagate(Y{i-1}(j,:), a); 
    end
    m(i,:) = mean(Y{i});
    
    plot(Y{i}(:,1),Y{i}(:,2),'.r');
    plot(m(i-1:i,1),m(i-1:i,2),'-k');
    drawnow;
    
end

hold off