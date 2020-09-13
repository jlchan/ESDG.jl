fileID = fopen('../entropy_plot/array_entropy_dissipation.txt');
formatSpec = '%f';
v_dissp_arr = fscanf(fileID,formatSpec);

fileID = fopen('../entropy_plot/array_entropy_conservative.txt');
formatSpec = '%f';
v_cons_arr = fscanf(fileID,formatSpec);

scatter(1:1440,v_dissp_arr,3,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.73,1.0,1.0])
hold on 
scatter(1:1440,v_cons_arr)
set(gca, 'YScale', 'log')
