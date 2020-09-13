% [x,y,z,v] = flow;
% p = patch(isosurface(x,y,z,v,-3));
% isonormals(x,y,z,v,p)
% p.FaceColor = 'red';
% p.EdgeColor = 'none';
% daspect([1 1 1])
% view(3); 
% axis tight
% camlight 
% lighting gouraud

tol = 0.01;
isoval = 0.05;
% q = importdata("q_criterion.txt");
% x = importdata("q_criterion_x.txt");
% y = importdata("q_criterion_y.txt");
% z = importdata("q_criterion_z.txt");
arr = double(abs(q-isoval)<tol);
arr(arr==0) = nan;
scatter3(x,y,z,arr);
