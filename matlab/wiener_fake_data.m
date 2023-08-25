N=2e2;
n  = randn(N,1);
n1 = filter(1, [1 -0.8], n);
n2 = filter(1, [1  0.6], n);

y = sin((0:N-1)'*0.05*pi);
x = y + n1;

order=12;
r_n2  = xcorr(n2, order, "biased");
R_n2  = toeplitz(r_n2(order+1:end));
r_xn2 = xcorr(x, n2, order, "biased");
wiener_coeffs = R_n2\r_xn2(order+1:end);

n1_hat = filter(wiener_coeffs, 1, n2);
e=x-n1_hat;

subplot(221);
plot([x y]);
subplot(222);
plot([e y]);
subplot(223);
plot(n2);
subplot(224);
plot(n1);