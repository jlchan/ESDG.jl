using Plots
push!(LOAD_PATH, "./src")
using Basis2DTri
using Basis1D
theme(:wong)
gr(aspect_ratio=1,legend=:bottomright,axis=nothing,border=:none,ticks=nothing,legendfontsize=14,xlims=[1.5,4],ylims=[0.8,3],
   markerstrokewidth=0,markersize=1)
r1D, w1D = gauss_quad(0,0,2)
Nfp = length(r1D) # number of points per face
e = ones(Nfp) # vector of all ones
z = zeros(Nfp) # vector of all zeros
rf = [r1D; -r1D; -e];
sf = [-e; r1D; -r1D];

r_tri,s_tri,_ = Basis2DTri.quad_nodes_tri(4)
plot([-1;-1].+3,[-1;1].+2,seriescolor=:black,lw=3,label = nothing)
plot!([-1;1].+3,[1;-1].+2,seriescolor=:black,lw=3,label = nothing)
plot!([-1;1].+3,[-1;-1].+2,seriescolor=:black,lw=3,label = nothing)
#scatter!(r_tri.+3,s_tri.+2,markersize=8,markercolor=:deepskyblue,markerstrokecolor=:black,markerstrokewidth=4,label = nothing)
scatter!(rf.+3,sf.+2,markersize=8,markershape=:rect,markercolor=:darkorange1,markerstrokecolor=:black,markerstrokewidth=4,label = nothing)

savefig("triangle_facequad_nodes.png")
