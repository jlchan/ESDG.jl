fileID = fopen('../conv/trans_vortex_y/array_conv.txt');
formatSpec = '%f';
conv_arr = fscanf(fileID,formatSpec);
conv_arr = reshape(conv_arr,4,1,5);
K1D_arr = [2;4;10;20];
N_P_arr = [1;2;3;4;5];

loglog(1./(2*K1D_arr),conv_arr(:,1,1))
hold on
loglog(1./(2*K1D_arr),conv_arr(:,1,2))
loglog(1./(2*K1D_arr),conv_arr(:,1,3))
loglog(1./(2*K1D_arr),conv_arr(:,1,4))
loglog(1./(2*K1D_arr),conv_arr(:,1,5))


% fileID = fopen('array_1.txt','r');
% formatSpec = '%f';
% Q_1 = fscanf(fileID,formatSpec);
% 
% fileID = fopen('array_2.txt','r');
% formatSpec = '%f';
% Q_2 = fscanf(fileID,formatSpec);
% 
% fileID = fopen('array_3.txt','r');
% formatSpec = '%f';
% Q_3 = fscanf(fileID,formatSpec);
% 
% fileID = fopen('xq.txt','r');
% formatSpec = '%f';
% xq = fscanf(fileID,formatSpec);
% 
% fileID = fopen('yq.txt','r');
% formatSpec = '%f';
% yq = fscanf(fileID,formatSpec);
% 
% Nq_P = 16;
% Nd = 5;
% K = 30*30*2;
% Np_F = 8;
% xq = reshape(xq,Nq_P,Np_F,K);
% yq = reshape(yq,Nq_P,Np_F,K);
% Q_1 = reshape(Q_1,Nq_P,Np_F,Nd,K);
% Q_2 = reshape(Q_2,Nq_P,Np_F,Nd,K);
% Q_3 = reshape(Q_3,Nq_P,Np_F,Nd,K);
% 
% rho_1_f1 = zeros(Nq_P,K);
% rho_2_f1 = zeros(Nq_P,K);
% rho_3_f1 = zeros(Nq_P,K);
% xq_f1 = zeros(Nq_P,K);
% yq_f1 = zeros(Nq_P,K);
% for k = 1:K
%     rho_1_f1(:,k) = Q_1(:,1,1,k);
%     rho_2_f1(:,k) = Q_2(:,1,1,k);
%     rho_3_f1(:,k) = Q_3(:,1,1,k);
%     xq_f1(:,k) = xq(:,1,k);
%     yq_f1(:,k) = yq(:,1,k);
% end
% 
% % rho_1_f1 = rho_1_f1(:);
% % rho_2_f1 = rho_2_f1(:);
% % rho_3_f1 = rho_3_f1(:);
% % xq_f1 = xq_f1(:);
% % yq_f1 = yq_f1(:);
% 
% figure(1)
% contourf(xq_f1,yq_f1,rho_1_f1,'LineColor','none')
% colormap(parula)
% 
% % 
% % figure(2)
% % contourf(xq_f1,yq_f1,rho_2_f1,'LineColor','none')
% % 
% % figure(3)
% % contourf(xq_f1,yq_f1,rho_3_f1,'LineColor','none')
% 
