fileID = fopen('../conv_z_vortex_2/array_conv.txt');
formatSpec = '%f';
conv_arr = fscanf(fileID,formatSpec);
approx_err = 3.686560949249596e-8;

% z vortex
figure(1)
semilogy([2;4;6;8;10;12;14;16;20;24;28;32],conv_arr,'-s',...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.73,1.0,1.0],...
    'LineWidth',2,...
    'Color',[0.28,0.46,1.0]);
hold on
semilogy([2;4;6;8;10;12;14;16;20;24;28;32],approx_err*ones(12),'--',...
    'LineWidth',4,...
    'Color',[0.8,0.0,0.0]);
xlabel('Number of Fourier modes','fontweight','bold')
ylabel('L^2 error','fontweight','bold')
legend({'L^2 Error','Approximation Error'})
ax = gca;
ax.FontSize = 13;
ax.TickLength = [.015,.015];
ax.LineWidth = 1;