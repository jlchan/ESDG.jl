using Plots
push!(LOAD_PATH, "./src")
using Basis2DTri
using Basis1D
theme(:wong)
gr(aspect_ratio=1,legend=:bottomright,axis=nothing,border=nothing,ticks=nothing,legendfontsize=14,xlims=[-1,7.3],
   markerstrokewidth=0,markersize=1)
r1D, w1D = gauss_quad(0,0,2)
Nfp = length(r1D) # number of points per face
e = ones(Nfp) # vector of all ones
z = zeros(Nfp) # vector of all zeros
rf = [r1D; -r1D; -e];
sf = [-e; r1D; -r1D];

r_tri,s_tri,_ = Basis2DTri.quad_nodes_tri(4)
plot([-1;-1],[-1;1],seriescolor=:black,lw=3,label = nothing)
plot!([-1;1],[1;-1],seriescolor=:black,lw=3,label = nothing)
plot!([-1;1],[-1;-1],seriescolor=:black,lw=3,label = nothing)
plot!([-1;-1].+3/4,[-1;1].+2/4,seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;1].+3/4,[1;-1].+2/4,seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;1].+3/4,[-1;-1].+2/4,seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;-1].+3,[-1;1].+2,seriescolor=:black,lw=3,label = nothing)
plot!([-1;1].+3,[1;-1].+2,seriescolor=:black,lw=3,label = nothing)
plot!([-1;1].+3,[-1;-1].+2,seriescolor=:black,lw=3,label = nothing)
plot!([1;4],[-1,1],seriescolor=:black,lw=3,label = nothing)
plot!([-1;2],[-1,1],seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;2],[1,3],seriescolor=:black,lw=3,label = nothing)
scatter!(r_tri.+3/4,s_tri.+2/4,markersize=5,markercolor=:deepskyblue,markerstrokecolor=:black,markerstrokewidth=2,label = nothing)
scatter!(rf.+3/4,sf.+2/4,markersize=5,markershape=:rect,markercolor=:darkorange1,markerstrokecolor=:black,markerstrokewidth=2,label = nothing)
scatter!(r_tri.+3,s_tri.+2,markersize=5,markercolor=:deepskyblue,markerstrokecolor=:black,markerstrokewidth=2,label = "Volume quadrature points")
scatter!(rf.+3,sf.+2,markersize=5,markershape=:rect,markercolor=:darkorange1,markerstrokecolor=:black,markerstrokewidth=2,label = "Face quadrature points")
scatter!([1.5;2;2.5],[0;0.5*2/3;1*2/3],markercolor=:black,markersize=4,label = nothing)
savefig("reference_elem.png")
