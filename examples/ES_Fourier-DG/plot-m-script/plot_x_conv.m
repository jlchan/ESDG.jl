fileID = fopen('../conv_x_vortex/array_conv.txt');
formatSpec = '%f';
conv_arr = fscanf(fileID,formatSpec);
conv_arr = reshape(conv_arr,5,4,4);
h_arr = 1./[2;4;6;8;10]/2;

% x vortex
figure(1)
for i = 1:4
    loglog(h_arr,conv_arr(:,2,i),'-s',...
    'MarkerSize',10,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[0.73,1.0,1.0],...
    'LineWidth',3,...
    'Color',[0.28,0.46,1.0]);
    xticks([10^-1.5, 10^-1, 10^-0.5])
    xticklabels({})
    hold on
end

slope_arr = zeros(4,4);
avg_slope = zeros(4);
for i = 1:4
    for j = 1:4
        slope_arr(j,i) = (log(conv_arr(j,2,i))-log(conv_arr(j+1,2,i)))/(log(h_arr(j))-log(h_arr(j+1)));
    end
end