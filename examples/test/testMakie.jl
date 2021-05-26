using GLMakie
# using CairoMakie
using Triangulate
using ESDG
# CairoMakie.activate!()
GLMakie.activate!()

# dependencies
using GeometryBasics, Makie 
using LinearAlgebra

N = 3
Nplot = 25
elementType = Tri()
rd = RefElemData(elementType,N)
md = MeshData(uniform_mesh(elementType,64)...,rd)

# submesh visualization 
rp,sp = equi_nodes(elementType,Nplot)
Vp = NodesAndModes.vandermonde(elementType,N,rp,sp) / NodesAndModes.vandermonde(elementType,N,rd.r,rd.s)

@unpack x,y = md
x = @. x + .125*cos(pi/2 * (x-.5))*cos(3*pi/2*(y-.7))
y = @. y + .125*sin(4*pi/2 * (x-.75))*cos(pi/2*(y-.8))

u = @. sin(pi*x)*sin(pi*y)*exp(-5*(x^2+y^2))
u[:,1:md.K÷2] .+= .25
zz = Vp*u
# md_curved = @set md.xyz = (x,y);

function MakieMeshPlot(u,rd::RefElemData{2},md::MeshData{2})
    return MakieMeshPlot(rd.Vp*u,rd.Vp,rd.rst,md.xyz)
end

function MakieMeshPlot(uplot,Vp,rst::NTuple{2},xyz::NTuple{2})

    # build reference triangulation
    rp,sp = (x->Vp*x).(rst)
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims([rp sp]) 
    triout,_ = triangulate("Q", triin)
    t = permutedims(triout.trianglelist) 
    makie_triangles = Makie.to_triangles(t)

    K = size(first(xyz),2)
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef,K)
    coordinates = zeros(size(Vp,1),3)
    for e = 1:md.K
        xyze = view.(xyz,:,e)
        zze = view(uplot,:,e)
        for d = 1:length(xyze)
            mul!(view(coordinates,:,d),Vp,xyze[d])
        end
        coordinates[:,3] .= zze        
        trimesh[e] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),makie_triangles) # speed this up?
    end
    return merge([trimesh...])
end

vis_mesh = MakieMeshPlot(zz,Vp,rd.rst,(x,y))
mesh(vis_mesh,color=vec(zz))

