using Plots
push!(LOAD_PATH, "./src")
using Basis2DTri
using Basis1D
theme(:wong)
gr(aspect_ratio=1,legend=nothing,axis=nothing,border=nothing,ticks=nothing,
   markerstrokewidth=0,markersize=1)
r1D, w1D = gauss_quad(0,0,2)
Nfp = length(r1D) # number of points per face
e = ones(Nfp) # vector of all ones
z = zeros(Nfp) # vector of all zeros
rf = [r1D; -r1D; -e];
sf = [-e; r1D; -r1D];

r_tri,s_tri,_ = Basis2DTri.quad_nodes_tri(4)
# Front triangle
plot([-1;-1],[-1;1],seriescolor=:black,lw=3,label = nothing)
plot!([-1;1],[1;-1],seriescolor=:black,lw=3,label = nothing)
plot!([-1;1],[-1;-1],seriescolor=:black,lw=3,label = nothing)
plot!([1;4],[-1,1],seriescolor=:black,lw=3,label = nothing)
plot!([-1;2],[-1,1],seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;2],[1,3],seriescolor=:black,lw=3,label = nothing)

plot!([-1;-1].+3/4,[-1;1].+2/4,seriescolor=:royalblue1,lw=3,ls=:dash,label = nothing)
plot!([-1;1].+3/4,[1;-1].+2/4,seriescolor=:royalblue1,lw=3,ls=:dash,label = nothing)
plot!([-1;1].+3/4,[-1;-1].+2/4,seriescolor=:royalblue1,lw=3,ls=:dash,label = nothing)
scatter!(r_tri.+3/4,s_tri.+2/4,markersize=5,markercolor=:royalblue1,markerstrokecolor=:black,markerstrokewidth=2)
scatter!(rf.+3/4,sf.+2/4,markersize=3,markershape=:rect,markercolor=:royalblue1,markerstrokecolor=:black,markerstrokewidth=2)

plot!([-1;-1].+3/4*2,[-1;1].+2/4*2,seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;1].+3/4*2,[1;-1].+2/4*2,seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;1].+3/4*2,[-1;-1].+2/4*2,seriescolor=:black,lw=3,ls=:dash,label = nothing)
scatter!(r_tri.+3/4*2,s_tri.+2/4*2,markersize=5,markercolor=:white,markerstrokecolor=:black,markerstrokewidth=2)
scatter!(rf.+3/4*2,sf.+2/4*2,markersize=3,markershape=:rect,markercolor=:white,markerstrokecolor=:black,markerstrokewidth=2)

plot!([-1;-1].+3/4*3,[-1;1].+2/4*3,seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;1].+3/4*3,[1;-1].+2/4*3,seriescolor=:black,lw=3,ls=:dash,label = nothing)
plot!([-1;1].+3/4*3,[-1;-1].+2/4*3,seriescolor=:black,lw=3,ls=:dash,label = nothing)
scatter!(r_tri.+3/4*3,s_tri.+2/4*3,markersize=5,markercolor=:white,markerstrokecolor=:black,markerstrokewidth=2)
scatter!(rf.+3/4*3,sf.+2/4*3,markersize=3,markershape=:rect,markercolor=:white,markerstrokecolor=:black,markerstrokewidth=2)

plot!([-1;-1].+3,[-1;1].+2,seriescolor=:black,lw=3,label = nothing)
plot!([-1;1].+3,[1;-1].+2,seriescolor=:black,lw=3,label = nothing)
plot!([-1;1].+3,[-1;-1].+2,seriescolor=:black,lw=3,label = nothing)

scatter!(r_tri.+3,s_tri.+2,markersize=5,markercolor=:white,markerstrokecolor=:black,markerstrokewidth=2)
scatter!(rf.+3,sf.+2,markersize=3,markershape=:rect,markercolor=:white,markerstrokecolor=:black,markerstrokewidth=2)
# savefig("reference_tri.png")
