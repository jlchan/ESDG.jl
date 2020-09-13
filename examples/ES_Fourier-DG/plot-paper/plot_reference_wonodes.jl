using Plots
push!(LOAD_PATH, "./src")
using Basis2DTri
using Basis1D
theme(:wong)
gr(aspect_ratio=1,border=:none,legend=:bottomright,axis=nothing,ticks=nothing,legendfontsize=14,xlims=[-1,4],
   markerstrokewidth=0,markersize=1)
r1D, w1D = gauss_quad(0,0,2)
Nfp = length(r1D) # number of points per face
e = ones(Nfp) # vector of all ones
z = zeros(Nfp) # vector of all zeros
rf = [r1D; -r1D; -e];
sf = [-e; r1D; -r1D];

r_tri,s_tri,_ = Basis2DTri.quad_nodes_tri(4)
plot([-1;-1],[-1;1],seriescolor=:darkorange1,lw=5,label = nothing)
plot!([-1;1],[1;-1],seriescolor=:darkorange1,lw=5,label = nothing)
plot!([-1;1],[-1;-1],seriescolor=:darkorange1,lw=5,label = nothing)
plot!([-1;-1].+3,[-1;1].+2,seriescolor=:darkorange1,lw=5,label = nothing)
plot!([-1;1].+3,[1;-1].+2,seriescolor=:darkorange1,lw=5,label = nothing)
plot!([-1;1].+3,[-1;-1].+2,seriescolor=:darkorange1,lw=5,label = nothing)
plot!([1;4],[-1,1],seriescolor=:darkorange1,lw=5,label = nothing)
plot!([-1;2],[-1,1],seriescolor=:darkorange1,lw=5,ls=:dash,label = nothing)
plot!([-1;2],[1,3],seriescolor=:darkorange1,lw=5,label = nothing)
savefig("wedge.png")
