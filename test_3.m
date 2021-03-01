v = 0.0:0.1:2.0;  % plotting range from -5 to 5
[x, y, z] = meshgrid(v); 
cond1 = x >= z;  % check conditions for these values
cond2 = y >= z;
cond3 = x+y+2*z<=2;
cond4 = z >= 0;
cond5 = x >= 3*z;

cond1 = double(cond1);  % convert to double for plotting
cond2 = double(cond2);
cond3 = double(cond3);
cond4 = double(cond4);
cond5 = double(cond5);

cond_w = cond1 & cond2 & cond3 & cond4 & cond5;
cond_wo = cond1 & cond2 & cond3 & cond4;

figure()
scatter3(x(cond_w),y(cond_w),z(cond_w),'filled','k')
figure()
scatter3(x(cond_wo),y(cond_wo),z(cond_wo),'filled')

% scatter(x,y,z,cond)
% view(0,90)    % change to top view