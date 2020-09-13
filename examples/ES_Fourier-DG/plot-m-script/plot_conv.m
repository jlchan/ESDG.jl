fileID = fopen('../conv/trans_vortex_y/array_conv_LF.txt');
formatSpec = '%f';
conv_arr_LF = fscanf(fileID,formatSpec);
conv_arr_LF = reshape(conv_arr_LF,4,1,5);

fileID = fopen('../conv/trans_vortex_y/array_conv_conservative.txt');
formatSpec = '%f';
conv_arr = fscanf(fileID,formatSpec);
conv_arr = reshape(conv_arr,4,1,5);
K1D_arr = [2;4;10;20];
N_P_arr = [1;2;3;4;5];

figure(1)
p = loglog(1./(2*K1D_arr),conv_arr_LF(:,1,1));
p.Color = [0 0.4470 0.7410];
p.LineWidth = 2;
hold on
loglog(1./(2*K1D_arr),conv_arr_LF(:,1,2))
loglog(1./(2*K1D_arr),conv_arr_LF(:,1,3))
loglog(1./(2*K1D_arr),conv_arr_LF(:,1,4))
loglog(1./(2*K1D_arr),conv_arr_LF(:,1,5))

figure(2)
p = loglog(1./(2*K1D_arr),conv_arr(:,1,1));
p.Color = [0 0.4470 0.7410];
p.LineWidth = 2;
hold on
loglog(1./(2*K1D_arr),conv_arr(:,1,2))
loglog(1./(2*K1D_arr),conv_arr(:,1,3))
loglog(1./(2*K1D_arr),conv_arr(:,1,4))
loglog(1./(2*K1D_arr),conv_arr(:,1,5))
