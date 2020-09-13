scaling = 1.3
f = @(x,y) x^2 + 2*(3/5*(x)^(2/3)-y)^2-1;
f_plot = @(x,y) f(scaling*x,y)
g = @(x,y) f(-scaling*x,y);
fimplicit(f_plot,[-2 2 -2 2])
hold on 
fimplicit(g,[-2 2 -2 2])