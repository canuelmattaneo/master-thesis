clc; clear;
d = table2array(readtable("tree.csv"));
d = d(1:end-1);
g = table2array(readtable("water.csv"));
g = g-mean(g) + randn(90900,1);

d=d/4;
norm = max(d+g);
d = d/norm;
g = g/norm;

% DA CAMBIARE v2->v1 e viceversa 
v1 = filter(1, [1 +0.9], g);
x = d+g;

P=30;
mu = 1e-4;
w = zeros(P+1,1);
e = zeros(size(x));
y = zeros(size(x));

for n=P+1:size(d)
    v1_win = v1(n:-1:n-P);
    y(n) = w'*v1_win;
    e(n) = x(n)-y(n);
    w = w + mu*e(n)*v1_win;
end

subplot(221)
%plot(g)
%yyaxis right
%plot(v1)
plot([v1 g])
legend({'v_1[n]', 'water[n]'})

subplot(222)
plot([d x])
legend({'tree[n]', 'tree[n]+water[n]'})

y = y*1.5;
subplot(223)
%yyaxis left
%plot(y)
%yyaxis right
%plot(g)
plot([y g])
legend({'v_1[n]*h_w[n]','water[n]'})

subplot(224)
scatter(y(20000:58000), g(20000:58000), 1, 'k','+')
xlim([0 0.55])
ylim([0 0.55])