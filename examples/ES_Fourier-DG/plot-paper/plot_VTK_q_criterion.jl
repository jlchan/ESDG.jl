push!(LOAD_PATH, "./src")
using DelimitedFiles
using Plots
using SetupDG
using UnPack
using UniformTriMesh
using PlotThemes
using ToeplitzMatrices
using Basis2DTri
using Basis1D
using LinearAlgebra

using WriteVTK
using Test
const FloatType = Float64
const vtk_filename_noext = "unstructured"

S_N(x) = @. sin(pi*x/h)/(2*pi/h)/tan(x/2)
function vandermonde_Sinc(h,r)
    N = convert(Int, 2*pi/h)
    V = zeros(length(r),N)
    for n = 1:N
        V[:,n] = S_N(r.-n*h)
    end
    V[1,1] = 1
    V[end,end] = 1
    return V
end

T = 1.0
N = 4
Nq_P = countlines("src/QuadratureData/quad_nodes_tri_N$(2*N).txt")
Nd = 5
K1D = 128
K = K1D*K1D*2
Np_F = 8
Nplot = 2
Nplot_f = 250
Mach = ".3"
delta = ".05"
extra_attribute = "varwidthsquared"

folder_name = "K$(K1D)NPF$(Np_F)N$(N)delta$(delta)Mach$(Mach)$(extra_attribute)"
file_name = "array_2"
Q = readdlm("./examples/ES_Fourier-DG/$folder_name/$file_name.txt",'\t',Float64,'\n')
xq = readdlm("./examples/ES_Fourier-DG/$folder_name/xq.txt",'\t',Float64,'\n')
yq = readdlm("./examples/ES_Fourier-DG/$folder_name/yq.txt",'\t',Float64,'\n')

xq = reshape(xq,Nq_P,Np_F,K)
yq = reshape(yq,Nq_P,Np_F,K)

h       = 2*pi/Np_F
column  = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)]
Dt      = Array{Float64,2}(Toeplitz(column,column[[1;Np_F:-1:2]]))

rd = init_reference_tri(N);
@unpack fv,Nfaces,r,s,VDM,V1,Vp,Dr,Ds,rf,sf,wf,nrJ,nsJ,rq,sq,wq,Vq,M,Pq,Vf,LIFT = rd
rp, sp = Basis2DTri.equi_nodes_2D(Nplot)
Vp = Basis2DTri.vandermonde_2D(N,rp,sp)/VDM
Np = length(rp)

"Mesh related variables"
# Initialize 2D triangular mesh
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
@. VX = VX
@. VY = VY
md   = init_mesh((VX,VY),EToV,rd)
# Intialize triangular prism
VX   = repeat(VX,2)
VY   = repeat(VY,2)
VZ   = [2/Np_F*ones((K1D+1)*(K1D+1),1); ones((K1D+1)*(K1D+1),1)]
@unpack rxJ,sxJ,ryJ,syJ = md

JP = 1/K1D^2
JF = 1/pi
J = JF*JP
Dr = JF*Dr
Ds = JF*Ds
Dt = JP*Dt
VpPq = Vp*Pq

Q = reshape(Q,Nq_P,Np_F,Nd,K)
u = reshape(Q[:,:,2,:],Nq_P,Np_F*K)
v = reshape(Q[:,:,3,:],Nq_P,Np_F*K)
w = reshape(Q[:,:,4,:],Nq_P,Np_F*K)
norm_v = @. u^2+v^2+w^2

dudr = Vq*Dr*Pq*u
dvdr = Vq*Dr*Pq*v
dwdr = Vq*Dr*Pq*w
duds = Vq*Ds*Pq*u
dvds = Vq*Ds*Pq*v
dwds = Vq*Ds*Pq*w

dudt = zeros(size(dudr))
dvdt = zeros(size(dvdr))
dwdt = zeros(size(dwdr))


for k = 1:K
   f_idx = 1+(k-1)*Np_F:k*Np_F
   dudt[:,f_idx] = u[:,f_idx]*Dt'
   dvdt[:,f_idx] = v[:,f_idx]*Dt'
   dwdt[:,f_idx] = w[:,f_idx]*Dt'
end

q_criterion = zeros(Nq_P,Np_F*K)
tz = 1/JF
for k = 1:K
   rx,ry,sx,sy = rxJ[1,k],ryJ[1,k],sxJ[1,k],syJ[1,k]
   for f = 1:Np_F
      for q = 1:Nq_P
         idx = f+(k-1)*Np_F
         v_tensor = [dudr[q,idx]*rx+duds[q,idx]*sx dudr[q,idx]*ry+duds[q,idx]*sy dudt[q,idx]*tz;
                     dvdr[q,idx]*rx+dvds[q,idx]*sx dvdr[q,idx]*ry+dvds[q,idx]*sy dvdt[q,idx]*tz;
                     dwdr[q,idx]*rx+dwds[q,idx]*sx dwdr[q,idx]*ry+dwds[q,idx]*sy dwdt[q,idx]*tz;]
         s_tensor = 1/2*(v_tensor+transpose(v_tensor))
         Ω_tensor = 1/2*(v_tensor-transpose(v_tensor))
         q_criterion[q,idx] = 1/2*(norm(Ω_tensor,2)^2-norm(s_tensor,2)^2)
      end
   end
end

q_criterion = q_criterion/maximum(abs.(q_criterion))
@show maximum(abs.(q_criterion))
q_criterion = reshape(q_criterion,Nq_P,Np_F,K)
zq = reshape(repeat(collect(-1+2/Np_F:(2/Np_F):1),inner=(1,Nq_P),outer=(K,1))',Nq_P,Np_F,K)
zp = LinRange(-1+2/Np_F,1,Nplot_f)
tp = LinRange(h,2*pi,Nplot_f)
VDM_F = vandermonde_Sinc(h,tp)
V2 = vandermonde_1D(1,tp)/vandermonde_1D(1,LinRange(h,2*pi,Np_F))
p'p
x_plot = zeros(size(Vp,1),Nplot_f,K)
y_plot = zeros(size(Vp,1),Nplot_f,K)
z_plot = zeros(size(Vp,1),Nplot_f,K)
q_criterion_plot = zeros(size(Vp,1),Nplot_f,K)
norm_v = reshape(norm_v,Nq_P,Np_F,K)
norm_v_plot = zeros(size(Vp,1),Nplot_f,K)
for k = 1:K
   x_plot[:,:,k] = Vp*Pq*xq[:,:,k]*V2'
   y_plot[:,:,k] = Vp*Pq*yq[:,:,k]*V2'
   z_plot[:,:,k] = Vp*Pq*zq[:,:,k]*V2'
   q_criterion_plot[:,:,k] = Vp*Pq*q_criterion[:,:,k]*VDM_F'
   norm_v_plot[:,:,k] = Vp*Pq*norm_v[:,:,k]*VDM_F'
end

@show maximum(abs.(q_criterion_plot))


# plotting for 2 linear-quadratic wedge elements
Np = 12
Nplot_f = Nplot_f - 1
pts = Array{FloatType}(undef,3,Np*Nplot_f*K)
order_v = [1;6;3]
order_m = [4;5;2]
h_f = 1.75/Nplot_f

qcriterion_data = Array{FloatType}(undef,Np*Nplot_f*K)
norm_v_data = Array{FloatType}(undef,Np*Nplot_f*K)
for k = 1:K
    for f = 1:Nplot_f
        trivx = x_plot[order_v,f,k]
        trivy = y_plot[order_v,f,k]
        trimx = x_plot[order_m,f,k]
        trimy = y_plot[order_m,f,k]
        triz = z_plot[order_v,f,k]
        qv = q_criterion_plot[order_v,f,k]
        qm = q_criterion_plot[order_m,f,k]
        qv_next = q_criterion_plot[order_v,f+1,k]
        qm_next = q_criterion_plot[order_m,f+1,k]

        normv = norm_v_plot[order_v,f,k]
        normm = norm_v_plot[order_m,f,k]
        normv_next = norm_v_plot[order_v,f+1,k]
        normm_next = norm_v_plot[order_m,f+1,k]
        pts[1,(k-1)*(Np*Nplot_f)+(f-1)*Np+1:(k-1)*(Np*Nplot_f)+f*Np] = [trivx trivx trimx trimx]
        pts[2,(k-1)*(Np*Nplot_f)+(f-1)*Np+1:(k-1)*(Np*Nplot_f)+f*Np] = [trivy trivy trimy trimy]
        pts[3,(k-1)*(Np*Nplot_f)+(f-1)*Np+1:(k-1)*(Np*Nplot_f)+f*Np] = [triz triz.+h_f triz triz.+h_f]
        qcriterion_data[(k-1)*(Np*Nplot_f)+(f-1)*Np+1:(k-1)*(Np*Nplot_f)+f*Np] = [qv; qv_next; qm; qm_next]
        norm_v_data[(k-1)*(Np*Nplot_f)+(f-1)*Np+1:(k-1)*(Np*Nplot_f)+f*Np] = [normv; normv_next; normm; normm_next]
    end
end

# Create cells
celltype = VTKCellTypes.VTK_QUADRATIC_LINEAR_WEDGE
cells = MeshCell[]
@test VTKCellType(celltype.vtk_id) === celltype
@test_throws BoundsError   VTKCellType(-3)   # ids start at 0
@test_throws ArgumentError VTKCellType(200)  # 200 is an unknown cell type
inds = 1:Np
for i = 1:Nplot_f*K
    c = MeshCell(celltype,inds.+(i-1)*Np)
    push!(cells, c)
end

# Initialise new vtu file (unstructured grid).
fname = "$(vtk_filename_noext)_3D_$(Mach)_T=$(T)"
outfile = vtk_grid(fname, pts, cells, compress=3) do vtk
    vtk["q-criterion"] = qcriterion_data
    vtk["norm-velocity"] = norm_v_data
end
