using Plots
using LinearAlgebra
using UnPack


push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

using SetupDG


theme(:wong)
gr(aspect_ratio=1,legend=:bottomright,axis=nothing,border=:none,ticks=nothing,legendfontsize=14,xlims=[-1,20],ylims=[-6,10],
   markerstrokewidth=0,markersize=1)

"Approximation parameters"
N = 2 # The order of approximation
K1D = 3
CFL = 2 # CFL goes up to 2.5ish
T = 1.0 # endtime

"Mesh related variables"
Kx = convert(Int,4/3*K1D)
Ky = K1D
VX, VY, EToV = uniform_tri_mesh(Kx, Ky)
@. VX = 15*(1+VX)/2
@. VY = 5*VY

# initialize ref element and mesh
# specify lobatto nodes to automatically get DG-SEM mass lumping
rd = init_reference_tri(N)
md = init_mesh((VX,VY),EToV,rd)

@unpack x,y,EToV = md

plot([1],[1],seriescolor=:white,lw=3,label = nothing)
for v in 1:2*Kx*Ky
   e1 = EToV[v,1:2]
   e2 = EToV[v,2:3]
   e3 = EToV[v,[3;1]]
   plot!(VX[e1],VY[e1],seriescolor=:black,lw=3,label = nothing)
   plot!(VX[e2],VY[e2],seriescolor=:black,lw=3,label = nothing)
   plot!(VX[e3],VY[e3],seriescolor=:black,lw=3,label = nothing)
   gui()
end
current()

plot!([0;0+3],[5;5+4],seriescolor=:black,lw=3,label = nothing)
plot!([15;15+3],[5;5+4],seriescolor=:black,lw=3,label = nothing)
plot!([15;15+3],[-5;-5+4],seriescolor=:black,lw=3,label = nothing)
plot!([3;18],[9;9],seriescolor=:black,lw=3,label = nothing)
plot!([18;18],[-1;9],seriescolor=:black,lw=3,label = nothing)

plot!([15/4;15/4+3],[5;5+4],seriescolor=:black,lw=3,label = nothing)
plot!([2*15/4;2*15/4+3],[5;5+4],seriescolor=:black,lw=3,label = nothing)
plot!([3*15/4;3*15/4+3],[5;5+4],seriescolor=:black,lw=3,label = nothing)

plot!([15;15+3],[-5+10/3;-5+4+10/3],seriescolor=:darkorange1,lw=5,label = nothing)
plot!([15;15+3],[-5+20/3;-5+4+20/3],seriescolor=:darkorange1,lw=5,label = nothing)

plot!([15;15],[-5+10/3;-5+20/3],seriescolor=:darkorange1,lw=5,label = nothing)
plot!([18;18],[-5+4+10/3;-5+4+20/3],seriescolor=:darkorange1,lw=5,label = nothing)

plot!([15-15/4;15],[-5+10/3;-5+10/3],seriescolor=:darkorange1,lw=5,label = nothing)
plot!([15-15/4;15],[-5+10/3;-5+20/3],seriescolor=:darkorange1,lw=5,label = nothing)

savefig("physical_domain.png")
