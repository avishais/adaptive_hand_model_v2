N = 500;
noise = 0.05;
t = 3*pi/2 * (1 + 2*rand(N,1));
h = 11 * rand(N,1);
X = [t.*cos(t), h, t.*sin(t)] + noise*randn(N,3);

plot3(X(:,1),X(:,2),X(:,3),'.');
axis equal
grid

save('SR.mat','X');