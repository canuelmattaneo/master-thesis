% working one
N=1500;
d = sin((0:N-1)'*5e-2*pi);
g = (randn(N,1)/3+((0:N-1).*(0:N-1))'./10e4);

% working one
N=1500;
g = sin((0:N-1)'*5e-2*pi);
d = (randn(N,1)+((0:N-1).*(0:N-1))'./10e4);

d = d./max(d+g);
g = g./max(d+g);
x = (d+g);

v2 = filter(1, [1 -0.6], g);

P=50;
mu = 0.01;
w = zeros(P+1,1);
e = zeros(size(x));
y = zeros(size(x));

for n=P+1:N
    V2 = v2(n:-1:n-P);
    y(n) = w'*V2;
    
    e(n) = x(n)-y(n);
    w = w + mu*e(n)*V2;

end

subplot(211)
plot(v2)
subplot(212)
plot([y])