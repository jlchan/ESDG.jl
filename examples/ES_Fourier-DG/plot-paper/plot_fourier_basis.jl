using Plots
gr(xlims=[0,2*pi],ylims=[-1.2,1.2],border=:none)

Np_F = 4
h = 2*pi/Np_F

fourier(x) = @. cos(pi*x)
r_plot = 0:4/200:2
plot(r_plot*pi,fourier(r_plot),lw=3,legend=nothing)
plot!(r_plot*pi,fourier(r_plot,2),lw=3,legend=nothing)
plot!(r_plot*pi,fourier(r_plot,4),lw=3,legend=nothing)
# plot!(r_plot*pi,fourier(r_plot,8),lw=3,legend=nothing)

# y = ones(size(r_plot))
# l = 0.3
# u = 0.3
# plot!([-pi,0],[2,2],fill=(0,:grey),fillalpha=0.3)
# plot!([-pi,0],-[2,2],fill=(0,:grey),fillalpha=0.3)
# plot!([2*pi,3*pi],[2,2],fill=(0,:grey),fillalpha=0.3)
# plot!([2*pi,3*pi],-[2,2],fill=(0,:grey),fillalpha=0.3)

savefig("fourier_basis.png")
