using Plots
gr(xlims=[-pi,3*pi],ylims=[-0.4,1.2],border=:none)

Np_F = 4
h = 2*pi/Np_F

Sinc(x) = @. sin(pi*x/h)/(2*pi/h*tan(x/2))
r_plot = -pi:pi/200:3*pi
plot!(r_plot,Sinc(r_plot.-h),lw=3,legend=nothing)
plot!(r_plot,Sinc(r_plot.-3*h),lw=3,legend=nothing)

y = ones(size(r_plot))
l = 0.3
u = 0.3
plot!([-pi,0],[2,2],fill=(0,:grey),fillalpha=0.3)
plot!([-pi,0],-[2,2],fill=(0,:grey),fillalpha=0.3)
plot!([2*pi,3*pi],[2,2],fill=(0,:grey),fillalpha=0.3)
plot!([2*pi,3*pi],-[2,2],fill=(0,:grey),fillalpha=0.3)
scatter!([0;pi/2;pi;3*pi/2;2*pi],[0;0;0;0;0],markerstrokewidth=0,markersize=5,markercolor=:black)

savefig("sinc_2.png")
