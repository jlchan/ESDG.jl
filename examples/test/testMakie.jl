using ESDG, GLMakie, Plots
using Triangulate

# dependencies
using GeometryBasics
using LinearAlgebra

N = 3
elementType = Quad()
rd = RefElemData(elementType,N,Nplot=N)
md = MeshData(uniform_mesh(elementType,100)...,rd)

@unpack x,y = md
x = @. x + .125*cos(pi/2 * (x-.5))*cos(3*pi/2*(y-.7))
y = @. y + .125*sin(4*pi/2 * (x-.75))*cos(pi/2*(y-.8))

xq,yq = rd.Vq*x,rd.Vq*y
u = @. sin(pi*x)*sin(pi*y)*exp(-5*(x^2+y^2))
u[:,1:md.K÷2] .+= .25
# u = rd.Pq*(@. xq + yq > 0)

Nfplot = Nplot+1
rfp,sfp = StartUpDG.map_face_nodes(elementType, LinRange(-1,1,Nfplot))
Vfp = vandermonde(elementType,N,rfp,sfp)/rd.VDM
xf,yf = (x->Vfp*x).((x,y)) 

# break line segments up with NaNs
xfp,yfp,ufp = (xf->vec(vcat(reshape(xf,Nfplot,rd.Nfaces*md.K),fill(NaN,1,rd.Nfaces*md.K)))).((xf,yf,Vfp*u))

Makie.mesh(rd.Vp*u,(x->rd.Vp*x).(rd.rst),(x->rd.Vp*x).((x,y)),color=vec(rd.Vp*u))
# draw mesh lines above/below
Makie.translate!(Makie.lines!(xfp,yfp,ufp,color=:white,linewidth=.5),0,0,1e-3) 
Makie.translate!(Makie.lines!(xfp,yfp,ufp,color=:white,linewidth=.5),0,0,-1e-3)
Makie.current_figure()

# Plots.plot(DGTriPseudocolor(rd.Vp*u,(x->rd.Vp*x).(rd.rst),(x->rd.Vp*x).((x,y))))
# Plots.plot!(xfp,yfp,color=:white,linewidth=1,leg=false)