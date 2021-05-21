using Plots
using Triangulate
using ESDG
using TriplotRecipes

N = 3
Nplot = N+1
elementType = Quad()
rd = RefElemData(elementType,N)
# triout = scramjet()
# VX,VY,EToV = triangulateIO_to_VXYEToV(triout)
VX,VY,EToV = uniform_mesh(elementType,16)
md = MeshData(VX,VY,EToV,rd)

# submesh visualization 
rp,sp = equi_nodes(elementType,Nplot)
Vp = NodesAndModes.vandermonde(elementType,N,rp,sp) / NodesAndModes.vandermonde(elementType,N,rd.r,rd.s)
triin = Triangulate.TriangulateIO()
triin.pointlist = permutedims([rp sp]) # hcat(unique([ Cdouble[rand(1:raster)/raster, rand(1:raster)/raster] for i in 1:n])...)
triout_plot,_ = triangulate("Q", triin)
# t = permutedims(triout.trianglelist) # mesh list
t = triout_plot.trianglelist # mesh list

@unpack x,y = md
u = @. 2 + sin(2.5*pi*x)*exp(x+sin(pi*y))
zz = Vp*u
z_extrema = extrema(zz)

num_ref_elements = size(t,2)
num_elements_total = num_ref_elements * md.K
tp = zeros(Int,3,num_elements_total)
zp = similar(tp,eltype(zz))
for e = 1:md.K
    for i = 1:size(t,2)
        tp[:,i + (e-1)*num_ref_elements] .= @views t[:,i] .+ (e-1)*size(Vp,1)
        zp[:,i + (e-1)*num_ref_elements] .= @views zz[t[:,i],e]
    end
end

dgtripcolor(vec(Vp*md.x),vec(Vp*md.y),zp,tp,color=:blues)

# trimesh!(VX,VY,EToV,fillalpha=0.0,linecolor=:white,linewidth=.1,aspect_ratio=:equal)
# # Plots.jl version - try NaN separators when Will implements them
# Plots.plot(aspect_ratio=:equal,size=(800,720)) 
# dgtripcolor!(xp,yp,zz,tp,color=:magma)
# for e = 1:md.K
#     xye = view.(md.xyz,:,e)
#     zze = view(zz,:,e)
#     # coordinates = [(x->Vp*x).(xye)... zze]
#     tripcolor!((x->Vp*x).(xye)...,zze,t,color=:magma)
#     # dgtripcolor!(x,y,zze,t,color=:magma)
#     # trimesh!(x,y,t,fillalpha=0.0,linecolor=:white)
# end
# display(Plots.plot!())




# using Plots,DelimitedFiles,TriplotRecipes

# f(x,y) = exp(0.1*sin(5.1*x + -6.2*y) + 0.3*cos(4.3*x + 3.4*y))

# x,y = eachcol(readdlm(joinpath(@__DIR__,"dolphin.xy")))
# t = readdlm(joinpath(@__DIR__,"dolphin.t"),Int)' .+ 1
# # z = f.(x,y)
# # z_dg = similar(t, eltype(z))

# for it=1:size(t,2)
#     z_dg[:,it] .= z[t[:,it]] .+ 0.05*(rand() - 0.5)
# end

# Plots.plot(aspect_ratio=:equal,size=(800,720))
# tripcolor!(x,y,z,t,color=:magma)
# # dgtripcolor!(x,y,z_dg,t,color=:magma)
# trimesh!(x,y,t,fillalpha=0.0,linecolor=:white)
# savefig("dolphin_dg.png")