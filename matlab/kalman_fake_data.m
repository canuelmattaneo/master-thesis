n = (0:500);
d = cos(pi/50*n + 2*pi*30)';
noise = (n.*n./(358801/8))' + 10*std(d)*randn(size(d));
z = d+noise;

A = [2*cos(pi/50) -1; 1 0];
H = [1 -cos(pi/50)];

Xhat = [0;0];
P = eye(2,2)*1e3;
Qw = zeros(2,2);
Qv = 0.5*var(z);

dhat = zeros(size(d));

for k=1:length(z)
    Xhatminus=A*Xhat;
    Pminus=A*P*A'+ Qw;
    K=(Pminus*H')/(H*Pminus*H'+Qv);
    Xhat = Xhatminus+K*(z(k)-H*Xhatminus);
    P=((eye(2)-K*H)*Pminus);
    dhat(k)=H*Xhat;
end

subplot(211),plot(z);
subplot(212),plot([d dhat]);