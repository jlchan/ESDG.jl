using Plots

# n=100; xr= 10*rand(n); yr= 10*rand(n); zr= Float64[ sin(xr[i])+cos(yr[i]) for i=1:n];
# surface(xr, yr, zr, size=[800,480])

# x = repeat(0:2/20:2,21)
# y = reshape(repeat(0:2/20:2,21),21,21)'[:]
# f(x,y) = @. sin(x)+cos(y)
# surface(x,y,f(x,y),colorbar=false, fc = :haline, legend=false)

x= [1.0:0.1:5;]; y= [4.0:0.1:9.0;]; f(x,y)= sin(x)*cos(y)+4;
p1= surface(x,y,f,camera=(30,30),legend=false,size=[800,500],xlims=[-1,6],zlims=[2,5],fc =:haline,grid=true,axis=false,grid_linewidth=3);	# 30 azimuth, 30 elevation is the default
plot(p1)

savefig("3d-surface.png")
