clear all

S = [2.2;1.5;1.6];
D = [1, 0, 0];
s = 2;

x = linspace(0,10,100);
for i = 1:length(x)
    k(i) = kde(x(i), S);
    k_fail(i) = kde(x(i), S(logical(D)), size(S,1));
end

figure(1)
clf
hold on
plot(x, k, '-b');
plot(s, kde(s, S),'ob');
plot(x, k_fail, '-r');
plot(s, kde(s, S(logical(D)), size(S,1)),'or');
plot(s*[1 1], ylim,':k')
plot(S, 0,'*k');
plot(S(logical(D)), 0,'ok','markerfacecolor','y');
hold off

function k = kde(sq, S, N)

if nargin==2
    N = size(S,1);
end

k = 0;
for i = 1:size(S,1)
    k = k + gaussian(norm(S(i,:)-sq));
end

k = k / N;


end




function g = gaussian(x, b)
if nargin==1
    b = 1;
end

g = exp(-x^2/(2*b^2))/(b*sqrt(2*pi));
end