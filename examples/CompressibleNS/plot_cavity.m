% K = 450;
% Nplot = 66;

K = 16*16*2;
Nplot = 66;

folder_name = 'N3-K16-T100-isot-nodissp';

fileID = fopen(sprintf('./%s/xp.txt',folder_name));
formatSpec = '%f';
xp = fscanf(fileID,formatSpec);
xp = reshape(xp,Nplot,K);

fileID = fopen(sprintf('./%s/yp.txt',folder_name));
formatSpec = '%f';
yp = fscanf(fileID,formatSpec);
yp = reshape(yp,Nplot,K);

fileID = fopen(sprintf('./%s/thist.txt',folder_name));
formatSpec = '%f';
thist = fscanf(fileID,formatSpec);

fileID = fopen(sprintf('./%s/visc.txt',folder_name));
formatSpec = '%f';
visc = fscanf(fileID,formatSpec);

fileID = fopen(sprintf('./%s/squaredv.txt',folder_name));
formatSpec = '%f';
squaredv = fscanf(fileID,formatSpec);
squaredv = reshape(squaredv,Nplot,K);

fileID = fopen(sprintf('./%s/rhstesthist.txt',folder_name));
formatSpec = '%f';
rhstesthist = fscanf(fileID,formatSpec);


figure(1)
scatter3(xp(:),yp(:),squaredv(:),18,squaredv(:),'filled','s');
view(0,90)
colormap default;
c = colorbar;
pbaspect([1 1 1])
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
set(gca,'FontSize',15);
L=cellfun(@(x)sprintf('%.1f',x),num2cell(get(c,'xtick')),'Un',0);
set(c,'XTickLabel',L);



visc = visc(2:end);
thist = thist(2:end);
thist(end) = 100.00;
rhstesthist = rhstesthist(2:end);

figure(2)
scatter(thist,visc,5,'filled')
set(gca,'FontSize',15);
xlabel('Time')
%ylim([-inf 0])

figure(3)
scatter(thist,rhstesthist,5,'filled')
set(gca,'FontSize',15);
xlabel('Time')
ylim([-inf 0])
