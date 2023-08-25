clc; clear;
t = table2array(readtable("tree.csv"));
t = t(1:end-1);
w = table2array(readtable("water.csv"));
%x = table2array(readtable("weight.csv"));

w = w-mean(w);
x = w+t;

n2 = filter(0.4, [1  0.99], w) + randn(90900,1)*5;

order=20;
r_n2  = xcorr(n2, order, "biased");
R_n2  = toeplitz(r_n2(order+1:end));
r_xn2 = xcorr(x, n2, order, "biased");
wiener_coeffs = R_n2\r_xn2(order+1:end);

w_hat = filter(wiener_coeffs, 1, n2);
t_hat = x-w_hat;

% not working -> var(diff(diff(t_hat)))
objective = @(K) mean((K*w_hat(1:100) - w(1:100)).^2);

K_opt = fminsearch(objective, 1.0)
w_hat_cal = K_opt*w_hat;
t_hat_cal = x - w_hat_cal;

subplot(221);
yyaxis left
plot(x);
yyaxis right
plot(t);
legend({'water[n]+tree[n]', 'tree[n]'})
t1 = title("$f_w(t)+f_t(t)$, $f_t(t)$");
set(t1,'Interpreter','latex');

subplot(222);
%yyaxis left
%plot(w);
%yyaxis right
%plot(n2);
plot([w n2])
legend({'water[n]', 'meas. water[n]'})
t2 = title("$f_w(t)$, Measured $f_w(t)$");
set(t2,'Interpreter','latex');

subplot(223);
%yyaxis left
%plot(w);
%yyaxis right
%plot(w_hat);
plot([w_hat w_hat_cal w])
legend({'est. water[n]', 'est. water[n] with cal.' 'water[n]'})
t3 = title("$f_W(t)$, $\hat{f}_W(t)$");
set(t3,'Interpreter','latex');

subplot(224);
%plot([t_hat t_hat_cal t])
%legend({'est. tree[n]', 'est. tree[n] cal.' 'tree[n]'})
%t4 = title("$f_T(t)$, $\hat{f}_T(t)$");
%set(t4,'Interpreter','latex');
