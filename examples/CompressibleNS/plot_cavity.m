K = 450;
Nplot = 66;

folder_name = 'N2-K15-T20-isot-dissp';

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
scatter3(xp(:),yp(:),squaredv(:),18,squaredv(:),'filled');
view(0,90)
colormap default;
colorbar;

figure(2)
scatter(thist,visc,7,'filled')


figure(3)
scatter(thist,rhstesthist,7,'filled')
