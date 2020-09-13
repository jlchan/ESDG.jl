fileID = fopen('../conv_z_vortex_5/array_conv.txt');
formatSpec = '%f';
conv_arr = fscanf(fileID,formatSpec);

fileID = fopen('../conv_z_vortex_5/array_approx.txt');
formatSpec = '%f';
approx_arr = fscanf(fileID,formatSpec);

% z vortex
figure(1)
semilogy(2:2:18,conv_arr(1:9),'-s',...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.73,1.0,1.0],...
    'LineWidth',2,...
    'Color',[0.28,0.46,1.0]);
hold on
% semilogy(2:2:18,approx_arr(1:9),'--',...
%     'LineWidth',4,...
%     'Color',[0.8,0.0,0.0]);
% % xlabel('Number of Fourier modes','fontweight','bold')
% % ylabel('L^2 error','fontweight','bold')
% legend({'L^2 Error','Approximation Error'})
ax = gca;
ax.FontSize = 16;
ax.TickLength = [.015,.015];
ax.LineWidth = 1;