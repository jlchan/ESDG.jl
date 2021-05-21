using Plots
using GLMakie
using CairoMakie
using Triangulate
using ESDG
# CairoMakie.activate!()
GLMakie.activate!()

N = 3
Nplot = 2*N+1
elementType = Tri()
rd = RefElemData(elementType,N)
md = MeshData(uniform_mesh(elementType,4)...,rd)

# submesh visualization 
rp,sp = equi_nodes(elementType,Nplot)
Vp = NodesAndModes.vandermonde(elementType,N,rp,sp) / NodesAndModes.vandermonde(elementType,N,rd.r,rd.s)
triin = Triangulate.TriangulateIO()
triin.pointlist = permutedims([rp sp]) # hcat(unique([ Cdouble[rand(1:raster)/raster, rand(1:raster)/raster] for i in 1:n])...)
triout,_ = triangulate("Q", triin)
t = permutedims(triout.trianglelist) # mesh list
# t = triout.trianglelist # mesh list

@unpack x,y = md
u = @. 2+sin(pi*x)*exp(x+sin(pi*y))
zz = Vp*u
z_extrema = extrema(zz)

# makie version
scene = Scene()
for e = 1:md.K
    xyze = view.(md.xyz,:,e)
    zze = view(zz,:,e)
    coordinates = [(x->Vp*x).(xyze)... zze]
    mesh!(scene,coordinates,color=zze,colorrange=z_extrema, t)
end
display(scene)
