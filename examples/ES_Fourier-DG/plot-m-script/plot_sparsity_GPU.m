Qrh = ones(15,15);
Qth = ones(4,4);
eyet = eye(15);
eyet(7:15,7:15) = zeros(9,9);
Qrh_skew = Qrh;
Qrh_skew(7:15,7:15) = 0.0;
for i = 1:6
    Qrh_skew(i,i) = 0;
end
Qrh_tri = zeros(60,60);
Qrh_tri(1:15,1:15) = Qrh_skew;
%Qrh_tri(16:30,16:30) = Qrh_skew;
eyet_fourier = zeros(15);
eyet_fourier(1,1) = 1.0;
Qth_fourier = kron(Qth,eyet_fourier);
figure(1)
spy(kron(eye(4),Qrh_skew),'k');
hold on
spy(Qrh_tri)
x=get(gca,'children');
set(x(1),'color',[0.28,0.46,1.0])
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'xlabel',[])
Qrh_tri_3 = zeros(60,60);
Qrh_tri_3(16:30,16:30) = Qrh_skew;
spy(Qrh_tri_3)
x=get(gca,'children');
set(x(1),'color',[0.28,0.46,1.0])
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'xlabel',[])
Qrh_tri_2 = zeros(60,60);
Qrh_tri_2(20,16:30) = Qrh_skew(5,:);
spy(Qrh_tri_2)
x=get(gca,'children');
set(x(1),'color',[1.0,0.5,0.0])
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'xlabel',[])


figure(2)
spy(kron(Qth,eyet),'k')
hold on
spy(Qth_fourier)
x=get(gca,'children');
set(x(1),'color',[0.28,0.46,1.0])
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'xlabel',[])
eyet_fourier_2 = zeros(15);
eyet_fourier_2(2,2) = 1.0;
Qth_fourier_2 = kron(Qth,eyet_fourier_2);
spy(Qth_fourier_2)
x=get(gca,'children');
set(x(1),'color',[0.28,0.46,1.0])
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'xlabel',[])
Qth_fourier_2 = zeros(60,60);
for i = 0:3
    Qth_fourier_2(17,2+i*15) = 1;
end
spy(Qth_fourier_2)
x=get(gca,'children');
set(x(1),'color',[1.0,0.5,0.0])
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'xlabel',[])
