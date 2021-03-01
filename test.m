fileID = fopen('./indicator_modal.txt');
formatSpec = '%f';
indicator_modal = fscanf(fileID,formatSpec);

figure()
bar(indicator_modal)
title('Modal - density')


fileID = fopen('./indicator_proj_V.txt');
formatSpec = '%f';
indicator_proj_V = fscanf(fileID,formatSpec);

figure()
bar(indicator_proj_V)
title('Norm difference projected V')



fileID = fopen('./indicator_proj_U.txt');
formatSpec = '%f';
indicator_proj_U = fscanf(fileID,formatSpec);

figure()
bar(indicator_proj_U)
title('Norm difference projected U')


fileID = fopen('./indicator_L2err_V.txt');
formatSpec = '%f';
indicator_L2err_V = fscanf(fileID,formatSpec);

figure()
bar(indicator_L2err_V)
title('L2 error projected V')



fileID = fopen('./indicator_L2err_U.txt');
formatSpec = '%f';
indicator_L2err_U = fscanf(fileID,formatSpec);

figure()
bar(indicator_L2err_U)
title('L2 error projected U')


fileID = fopen('./indicator_L1err_V.txt');
formatSpec = '%f';
indicator_L1err_V = fscanf(fileID,formatSpec);

figure()
bar(indicator_L1err_V)
title('L1 error projected V')



fileID = fopen('./indicator_L1err_U.txt');
formatSpec = '%f';
indicator_L1err_U = fscanf(fileID,formatSpec);

figure()
bar(indicator_L1err_U)
title('L1 error projected U')


fileID = fopen('./indicator_L1err_normalized_V.txt');
formatSpec = '%f';
indicator_L1err_normalized_V = fscanf(fileID,formatSpec);

figure()
bar(indicator_L1err_normalized_V)
title('L1 error normalized projected V')



fileID = fopen('./indicator_L1err_normalized_U.txt');
formatSpec = '%f';
indicator_L1err_normalized_U = fscanf(fileID,formatSpec);

figure()
bar(indicator_L1err_normalized_U)
title('L1 error normalized projected U')



fileID = fopen('./indicator_modal_V1.txt');
formatSpec = '%f';
indicator_modal_V1 = fscanf(fileID,formatSpec);

figure()
bar(indicator_modal_V1)
title('Modal - v1')



fileID = fopen('./indicator_modal_V2.txt');
formatSpec = '%f';
indicator_modal_V2 = fscanf(fileID,formatSpec);

figure()
bar(indicator_modal_V2)
title('Modal - v2')


fileID = fopen('./indicator_modal_V3.txt');
formatSpec = '%f';
indicator_modal_V3 = fscanf(fileID,formatSpec);

figure()
bar(indicator_modal_V3)
title('Modal - v3')
