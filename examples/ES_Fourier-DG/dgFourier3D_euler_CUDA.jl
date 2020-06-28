using Revise # reduce recompilation time
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using ToeplitzMatrices
using Test

using KernelAbstractions
using CUDA
using CUDAapi
CUDA.allowscalar(false)

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

using SetupDG

S_N(x) = @. sin(pi*x/h)/(2*pi/h)/tan(x/2)
"""
Vandermonde matrix of sinc basis functions determined by h,
evaluated at r
"""
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

vector_norm(U) = CuArrays.sum((x->x.^2).(U))

"Constants"
const sp_tol      = 1e-12
const shmem_limit = 1024 # Max number of elements in shared mem
const enable_test = false
const gamma       = 1.4f0
const NUM_TYPE    = Float32
"Program parameters"
const compute_L2_err          = false
const add_LF_dissipation      = true

"Approximation Parameters"
const N_P   = 2;    # The order of approximation in polynomial dimension
const Np_P  = Int((N_P+1)*(N_P+2)/2)
const Np_F  = 10;    # The order of approximation in Fourier dimension
const K1D   = 30;   # Number of elements in polynomial (x,y) dimension
const Nd    = 5;
const CFL   = 1.0;
const T     = 2.0;  # End time

"Time integration Parameters"
rk4a,rk4b,rk4c = rk45_coeffs()
CN             = (N_P+1)*(N_P+2)*3/2  # estimated trace constant for CFL
dt             = CFL * 2 / CN / K1D
Nsteps         = convert(Int,ceil(T/dt))
dt             = T/Nsteps

"Initialize Reference Element in Fourier dimension"
h       = 2*pi/Np_F
column  = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)]
column2 = [-pi^2/3/h^2-1/6; -((-1).^(1:Np_F-1)./(2*(sin.((1:Np_F-1)*h/2)).^2))]
Dt      = Array{Float64,2}(Toeplitz(column,column[[1;Np_F:-1:2]]))
D2t     = Array{Float64,2}(Toeplitz(column2,column2[[1;Np_F:-1:2]]))
t       = LinRange(h,2*pi,Np_F)

"Initialize Reference Element in polynomial dimension"
rd = init_reference_tri(N_P);
@unpack fv,Nfaces,r,s,VDM,V1,Dr,Ds,rf,sf,wf,nrJ,nsJ,rq,sq,wq,Vq,M,Pq,Vf,LIFT = rd
Lq    = LIFT

"Problem parameters"
Nq_P   = length(rq)
Nfp_P  = length(rf)
Nh_P   = Nq_P+Nfp_P # Number of hybridized points
Nq     = Nq_P*Np_F
Nh     = Nh_P*Np_F
Nfp    = Nfp_P*Np_F

# Adpatively determine number of threads
thread_count = 2^8
for n = 8:-1:5
    global thread_count
    thread_count = 2^n
    Nk_tri_max = div(thread_count-2,Nh_P)+2 # Max amount of triangles in a block
    Nk_fourier_max = div(thread_count-2,Np_F)+2 # Max amount of fourier slides in a block
    if max(Nd*Nh_P*Nk_tri_max,Nd*Np_F*Nk_fourier_max) < shmem_limit
        break
    end
end
const num_threads = thread_count
Nk_max = num_threads == 1 ? 1 : div(num_threads-2,Nh)+2 # Max amount of element in a block
Nk_tri_max = num_threads == 1 ? 1 : div(num_threads-2,Nh_P)+2 # Max amount of triangles in a block
Nk_fourier_max = num_threads == 1 ? 1 : div(num_threads-2,Np_F)+2 # Max amount of fourier slides in a block

"Mesh related variables"
# Initialize 2D triangular mesh
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
@. VX = 1+VX
@. VY = 1+VY
md   = init_mesh((VX,VY),EToV,rd)
# Intialize triangular prism
VX   = repeat(VX,2)
VY   = repeat(VY,2)
VZ   = [2/Np_F*ones((K1D+1)*(K1D+1),1); 2*ones((K1D+1)*(K1D+1),1)]
EToV = [EToV EToV.+(K1D+1)*(K1D+1)]

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB

# Initialize 3D mesh
@unpack x,y,xf,yf,xq,yq,rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapM,mapP,mapB = md
x,y,xf,yf,xq,yq,rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ = (x->reshape(repeat(x,inner=(1,Np_F)),size(x,1),Np_F,K)).((x,y,xf,yf,xq,yq,rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ))
z,zq,zf = (x->reshape(repeat(collect(2/Np_F:(2/Np_F):2),inner=(1,x),outer=(K,1))',x,Np_F,K)).((Np_P,Nq_P,Nfp_P))
mapM = reshape(1:Nfp_P*Np_P*K,Nfp_P,Np_P,K)
mapP_2D = (x->mod1(x,Nfp_P)+div(x-1,Nfp_P)*Nfp_P*Np_F).(mapP)
mapP = reshape(repeat(mapP_2D,inner=(1,Np_F)),Nfp_P,Np_F,K)
for j = 1:Np_F
    mapP[:,j,:] = mapP[:,j,:].+(j-1)*Nfp_P
end
mapP_d = zeros(Int32,Nfp_P,Np_F,Nd,K)
for k = 1:K
    for j = 1:Np_F
        for i = 1:Nfp_P
            val = mapP[i,j,k]
            elem = div(val-1,Nfp)
            n = mod1(val,Nfp)
            mapP_d[i,j,1,k] = elem*Nd*Nfp+n
            mapP_d[i,j,2,k] = elem*Nd*Nfp+n+Nfp
            mapP_d[i,j,3,k] = elem*Nd*Nfp+n+2*Nfp
            mapP_d[i,j,4,k] = elem*Nd*Nfp+n+3*Nfp
            mapP_d[i,j,5,k] = elem*Nd*Nfp+n+4*Nfp
        end
    end
end
mapP = mapP[:]
mapP_d = mapP_d[:] 


#############################################
######  Construct hybridized operators ######
#############################################

# scale by Fourier dimension
M = h*M
wq = h*wq
wf = h*wf

# Hybridized operators
Vh = [Vq;Vf]
rxJ,sxJ,ryJ,syJ = (x->mapslices((y->Vh*y),x,dims=(1,2))).((rxJ,sxJ,ryJ,syJ))
Ef = Vf*Pq
Br,Bs = (x->diagm(wf.*x)).((nrJ,nsJ))
Qr,Qs = (x->Pq'*M*x*Pq).((Dr,Ds))
Qrh,Qsh = (x->1/2*[x[1]-x[1]' Ef'*x[2];
                   -x[2]*Ef   x[2]]).(((Qr,Br),(Qs,Bs)))
Qrh_skew,Qsh_skew = (x->1/2*(x-x')).((Qrh,Qsh))
Qt = Dt
Qth = Qt # Not the SBP operator, weighted when flux differencing
Ph = [Vq;Vf]*Pq

# TODO: assume mesh uniform affine, so Jacobian are constants
# TODO: fix other Jacobian parts
JP = 1/K1D^2
JF = 1/pi
J = JF*JP
wq = J*wq
wf = JF*wf
Lq = 1/JP*Lq
Qrh = JF*Qrh
Qsh = JF*Qsh
Qth = JP*Qth
Qrh_skew = 1/2*(Qrh-Qrh')
Qsh_skew = 1/2*(Qsh-Qsh')
LIFTq = Vq*Lq
VPh = Vq*[Pq Lq]*diagm(1 ./ [wq;wf])
Winv = diagm(1 ./ [wq;wf])
Wq = diagm(wq)

# convert precision
Vq,wq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh,Wq,rxJ,sxJ,ryJ,syJ,nxJ,nyJ,sJ,J,h,rk4a,rk4b = (A->convert.(NUM_TYPE,A)).((Vq,wq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh,Wq,rxJ,sxJ,ryJ,syJ,nxJ,nyJ,sJ,J,h,rk4a,rk4b))
 
# TODO: refactor
Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh,Wq,Winv,rxJ,sxJ,ryJ,syJ = (x->CuArray(x)).((Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh,Wq,Winv,rxJ,sxJ,ryJ,syJ))
nxJ = CuArray(nxJ[:])
nyJ = CuArray(nyJ[:])
sJ  = CuArray(sJ[:])

ops = (Vq,wq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh,Wq)
mesh = (rxJ,sxJ,ryJ,syJ,nxJ,nyJ,sJ,J,h,mapP,mapP_d)
param = (K,Np_P,Nq_P,Nfp_P,Nh_P,Np_F,Nq,Nfp,Nh,Nk_max,Nk_tri_max,Nk_fourier_max)

# ================================= #
# ============ Routines =========== #
# ================================= #
function u_to_v!(VU,Q,Nd,Nq)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    @inbounds begin
    for i = index:stride:ceil(Int,length(VU)/Nd)
        k = div(i-1,Nq) # Current element
        n = mod1(i,Nq) # Current quad node

        idx = k*Nd*Nq+n
        rho = Q[idx]
        rhou = Q[idx+Nq]
        rhov = Q[idx+2*Nq]
        rhow = Q[idx+3*Nq]
        E = Q[idx+4*Nq]
        rhoUnorm = rhou^2+rhov^2+rhow^2
        rhoe = E-.5*rhoUnorm/rho
        sU = CUDA.log((gamma-1.0f0)*rhoe/CUDA.exp(gamma*CUDA.log(rho)))

        VU[idx] = (-E+rhoe*(gamma+1.0f0-sU))/rhoe
        VU[idx+Nq] = rhou/rhoe
        VU[idx+2*Nq] = rhov/rhoe
        VU[idx+3*Nq] = rhow/rhoe
        VU[idx+4*Nq] = -rho/rhoe
    end
    end
end

function v_to_u!(Qh,Nd,Nh)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    @inbounds begin
    for i = index:stride:ceil(Int,length(Qh)/Nd)
        k = div(i-1,Nh) # Current element
        n = mod1(i,Nh) # Current hybridized node

        # TODO: wrong variable names
        idx = k*Nd*Nh+n
        rho = Qh[idx]
        rhou = Qh[idx+Nh]
        rhov = Qh[idx+2*Nh]
        rhow = Qh[idx+3*Nh]
        E = Qh[idx+4*Nh]
        vUnorm = rhou^2+rhov^2+rhow^2
        rhoeV =
CUDA.exp(1.0f0/(gamma-1.0f0)*CUDA.log((gamma-1.0f0)/CUDA.exp(gamma*CUDA.log(-E))))*CUDA.exp(-(gamma-rho+vUnorm/(2.0f0*E))/(gamma-1.0f0))

        Qh[idx] = -rhoeV*E
        Qh[idx+Nh] = rhoeV*rhou
        Qh[idx+2*Nh] = rhoeV*rhov
        Qh[idx+3*Nh] = rhoeV*rhow
        Qh[idx+4*Nh] = rhoeV*(1.0f0-vUnorm/(2.0f0*E))
    end
    end
end

#TODO: combine with v_to_u
function u_to_primitive!(Qh,Nd,Nh)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    @inbounds begin
    for i = index:stride:ceil(Int,length(Qh)/Nd)
        k = div(i-1,Nh) # Current element
        n = mod1(i,Nh) # Current hybridized node
        # TODO: wrong varaible names
        idx = k*Nd*Nh+n

        rho = Qh[idx]
        rhou = Qh[idx+Nh]
        rhov = Qh[idx+2*Nh]
        rhow = Qh[idx+3*Nh]
        E = Qh[idx+4*Nh]

        beta = rho/(2.0f0*(gamma-1.0f0)*(E-.5*(rhou^2+rhov^2+rhow^2)/rho))

        Qh[idx+Nh] = rhou/rho
        Qh[idx+2*Nh] = rhov/rho
        Qh[idx+3*Nh] = rhow/rho
        Qh[idx+4*Nh] = beta
    end
    end
end

function extract_face_val_conservative!(Uf,Qh,Nfp_P,Nq_P,Nh_P)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    @inbounds begin
    for i = index:stride:length(Uf)
        k = div(i-1,Nfp_P)
        n = mod1(i,Nfp_P)
        Uf[i] = Qh[k*Nh_P+Nq_P+n]
    end
    end
end

function extract_face_val!(QM,Qh,Nfp_P,Nq_P,Nh_P)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    @inbounds begin
    for i = index:stride:length(QM)
        k = div(i-1,Nfp_P)
        n = mod1(i,Nfp_P)
        QM[i] = Qh[k*Nh_P+Nq_P+n]
    end
    end
end

function CU_logmean(uL,uR,logL,logR)
    da = uR-uL
    aavg = .5f0*(uL+uR)
    f = da/aavg
    if CUDA.abs(f)<1e-4
        v = f^2
        return aavg*(1.0f0 + v*(-.2f0-v*(.0512f0 - v*0.026038857142857f0)))
    else
        return -da/(logL-logR)
    end
end

function CU_euler_flux(rhoM,uM,vM,wM,betaM,rhoP,uP,vP,wP,betaP,rhologM,betalogM,rhologP,betalogP)
    rholog = CU_logmean(rhoM,rhoP,rhologM,rhologP)
    betalog = CU_logmean(betaM,betaP,betalogM,betalogP)

    # TODO: write in functions
    rhoavg = .5f0*(rhoM+rhoP)
    uavg = .5f0*(uM+uP)
    vavg = .5f0*(vM+vP)
    wavg = .5f0*(wM+wP)

    unorm = uM*uP+vM*vP+wM*wP
    pa = rhoavg/(betaM+betaP)
    E_plus_p = rholog/(2.0f0*(gamma-1.0f0)*betalog) + pa + .5f0*rholog*unorm

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. FxS1*vavg) # rho * u * v
    FxS4 = (@. FxS1*wavg) # rho * u * w
    FxS5 = (@. E_plus_p*uavg)
    FyS1 = (@. rholog*vavg)
    FyS2 = (@. FxS3) # rho * u * v
    FyS3 = (@. FyS1*vavg + pa)
    FyS4 = (@. FyS1*wavg) # rho * v * w
    FyS5 = (@. E_plus_p*vavg)
    FzS1 = (@. rholog*wavg)
    FzS2 = (@. FxS4) # rho * u * w
    FzS3 = (@. FyS4) # rho * v * w
    FzS4 = (@. FzS1*wavg + pa) # rho * w^2 + p
    FzS5 = (@. E_plus_p*wavg)

    return FxS1,FxS2,FxS3,FxS4,FxS5,FyS1,FyS2,FyS3,FyS4,FyS5,FzS1,FzS2,FzS3,FzS4,FzS5
end

function construct_lam!(lam,Uf,nxJ,nyJ,sJ,Nfp,K,Nd)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x

    @inbounds begin
    for i = index:stride:Nfp*K
        k = div(i-1,Nfp)
        n = mod1(i,Nfp)

        rhoM  = Uf[k*Nd*Nfp+n      ]
        rhouM = Uf[k*Nd*Nfp+n+  Nfp]
        rhovM = Uf[k*Nd*Nfp+n+2*Nfp]
        rhowM = Uf[k*Nd*Nfp+n+3*Nfp]
        EM    = Uf[k*Nd*Nfp+n+4*Nfp]
        nxJ_val = nxJ[k*Nfp+n]
        nyJ_val = nyJ[k*Nfp+n]
        sJ_val  = sJ[k*Nfp+n]
        rhouM_n = (rhouM*nxJ_val+rhovM*nyJ_val)/sJ_val
        lam[k*Nfp+n] = CUDA.sqrt(CUDA.abs(rhouM_n/rhoM))+CUDA.sqrt(gamma*(gamma-1.0f0)*(EM-0.5f0*rhouM_n^2/rhoM)/rhoM)
    end
    end
end

function surface_kernel!(flux,QM,QP,Uf,UfP,LFc,nxJ,nyJ,Nfp,K,Nd)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
    @inbounds begin
    for i = index:stride:Nfp*K
        k = div(i-1,Nfp)
        n = mod1(i,Nfp)

        rhoM  = QM[k*Nd*Nfp+n      ]
        uM    = QM[k*Nd*Nfp+n+  Nfp]
        vM    = QM[k*Nd*Nfp+n+2*Nfp]
        wM    = QM[k*Nd*Nfp+n+3*Nfp]
        betaM = QM[k*Nd*Nfp+n+4*Nfp]
        rhoP  = QP[k*Nd*Nfp+n      ]
        uP    = QP[k*Nd*Nfp+n+  Nfp]
        vP    = QP[k*Nd*Nfp+n+2*Nfp]
        wP    = QP[k*Nd*Nfp+n+3*Nfp]
        betaP = QP[k*Nd*Nfp+n+4*Nfp]
        nxJ_val = nxJ[k*Nfp+n]
        nyJ_val = nyJ[k*Nfp+n]

        rhouM = Uf[k*Nd*Nfp+n+  Nfp]
        rhovM = Uf[k*Nd*Nfp+n+2*Nfp]
        rhowM = Uf[k*Nd*Nfp+n+3*Nfp]
        EM    = Uf[k*Nd*Nfp+n+4*Nfp]
        rhouP = UfP[k*Nd*Nfp+n+  Nfp]
        rhovP = UfP[k*Nd*Nfp+n+2*Nfp]
        rhowP = UfP[k*Nd*Nfp+n+3*Nfp]
        EP    = UfP[k*Nd*Nfp+n+4*Nfp] 
        LFc_val = LFc[k*Nfp+n]

        rhologM = CUDA.log(rhoM)
        rhologP = CUDA.log(rhoP)
        betalogM = CUDA.log(betaM)
        betalogP = CUDA.log(betaP)

        FxS1,FxS2,FxS3,FxS4,FxS5,FyS1,FyS2,FyS3,FyS4,FyS5,_,_,_,_,_ = CU_euler_flux(rhoM,uM,vM,wM,betaM,rhoP,uP,vP,wP,betaP,rhologM,betalogM,rhologP,betalogP)

        flux[k*Nd*Nfp+n      ] = nxJ_val*FxS1+nyJ_val*FyS1-LFc_val*(rhoP-rhoM)
        flux[k*Nd*Nfp+n+  Nfp] = nxJ_val*FxS2+nyJ_val*FyS2-LFc_val*(rhouP-rhouM)
        flux[k*Nd*Nfp+n+2*Nfp] = nxJ_val*FxS3+nyJ_val*FyS3-LFc_val*(rhovP-rhovM)
        flux[k*Nd*Nfp+n+3*Nfp] = nxJ_val*FxS4+nyJ_val*FyS4-LFc_val*(rhowP-rhowM)
        flux[k*Nd*Nfp+n+4*Nfp] = nxJ_val*FxS5+nyJ_val*FyS5-LFc_val*(EP-EM)
    end
    end
end

function flux_differencing_xy_kernel!(gradfh,Qh,rxJ,sxJ,ryJ,syJ,Qrh_skew,Qsh_skew,Nh,Nh_P,Np_F,K,Nd,Nq_P,num_threads,Nk_max,Nk_tri_max)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
    #TODO: using stride to avoid possible insufficient blocks?
    tid_start = (blockIdx().x-1)*blockDim().x + 1 # starting threadid of current block
    tid_end = (blockIdx().x-1)*blockDim().x + num_threads # ending threadid of current block
    k_start = div(tid_start-1,Nh) # starting elementid of current block
    k_end = tid_end < Nh*K ? div(tid_end-1,Nh) : K-1 # ending elementid of current block
    Nk = k_end-k_start+1 # Number of elements in current block
    k_tri_start = div(tid_start-1,Nh_P) # starting triangle id of current block
    k_tri_end = tid_end < Nh*K ? div(tid_end-1,Nh_P) : Np_F*K-1 # ending triangle id of current block
    Nk_tri = k_tri_end-k_tri_start+1 # Number of triangles in current block

    # Parallel read
    Qh_shared = @cuDynamicSharedMem(NUM_TYPE,Nh_P*Nd*Nk_tri_max)
    load_size = div(Nh_P*Nd*Nk_tri-1,num_threads)+1 # size of read of each thread
    @inbounds begin
    for i in (threadIdx().x-1)*load_size+1:min(threadIdx().x*load_size,Nh_P*Nd*Nk_tri)
        # TODO: cleanup
        offset_tri = div(i-1,Nh_P*Nd) # distance with k_tri_start
        n_local = mod1(i,Nh_P*Nd)
        n_d = div(n_local-1,Nh_P) # Current component
        m_xy = mod1(n_local,Nh_P) # Current hybridized node index at xy slice
        i_tri = k_tri_start+offset_tri # Current triangle
        k = div(i_tri,Np_F) # Current element
        i_xy = mod(i_tri,Np_F) # Current local xy slice id on the elment
        Qh_shared[i] = Qh[k*Nd*Nh+i_xy*Nh_P+m_xy+n_d*Nh]
    end
    end

    sync_threads()

    @inbounds begin
    if index <= Nh*K
        i = index
        k = div(i-1,Nh)   # Current element
        m = mod1(i,Nh)    # Current node on element
        i_xy = div(m-1,Nh_P) # Current local xy slice id on the element
        m_xy = mod1(i,Nh_P)  # Current hybridized node index at x-y slice

        k_tri_offset = div(i-1,Nh_P)-k_tri_start
        rhoL  = Qh_shared[k_tri_offset*Nd*Nh_P+m_xy       ] 
        uL    = Qh_shared[k_tri_offset*Nd*Nh_P+m_xy+  Nh_P] 
        vL    = Qh_shared[k_tri_offset*Nd*Nh_P+m_xy+2*Nh_P] 
        wL    = Qh_shared[k_tri_offset*Nd*Nh_P+m_xy+3*Nh_P] 
        betaL = Qh_shared[k_tri_offset*Nd*Nh_P+m_xy+4*Nh_P] 
        rhologL  = CUDA.log(rhoL)
        betalogL = CUDA.log(betaL)

        rho_sum = 0.0f0
        u_sum = 0.0f0
        v_sum = 0.0f0
        w_sum = 0.0f0
        beta_sum = 0.0f0

        # Assume Affine meshes

        rxJ_val = rxJ[1,1,k+1] 
        sxJ_val = sxJ[1,1,k+1]
        ryJ_val = ryJ[1,1,k+1]
        syJ_val = syJ[1,1,k+1]

        # TODO: better way to indexing
        for n_xy = 1:Nh_P
            rhoR  = Qh_shared[k_tri_offset*Nd*Nh_P+n_xy       ] 
            uR    = Qh_shared[k_tri_offset*Nd*Nh_P+n_xy+  Nh_P] 
            vR    = Qh_shared[k_tri_offset*Nd*Nh_P+n_xy+2*Nh_P] 
            wR    = Qh_shared[k_tri_offset*Nd*Nh_P+n_xy+3*Nh_P] 
            betaR = Qh_shared[k_tri_offset*Nd*Nh_P+n_xy+4*Nh_P] 
            rhologR = CUDA.log(rhoR)
            betalogR = CUDA.log(betaR)

            FxS1,FxS2,FxS3,FxS4,FxS5,FyS1,FyS2,FyS3,FyS4,FyS5,FzS1,FzS2,FzS3,FzS4,FzS5 = CU_euler_flux(rhoL,uL,vL,wL,betaL,rhoR,uR,vR,wR,betaR,rhologL,betalogL,rhologR,betalogR)

            # col_idx = mod1(n,Nh_P)
            Qx_val = 2.0f0*(rxJ_val*Qrh_skew[m_xy,n_xy]+sxJ_val*Qsh_skew[m_xy,n_xy])
            Qy_val = 2.0f0*(ryJ_val*Qrh_skew[m_xy,n_xy]+syJ_val*Qsh_skew[m_xy,n_xy])
            rho_sum  += Qx_val*FxS1+Qy_val*FyS1
            u_sum    += Qx_val*FxS2+Qy_val*FyS2
            v_sum    += Qx_val*FxS3+Qy_val*FyS3
            w_sum    += Qx_val*FxS4+Qy_val*FyS4
            beta_sum += Qx_val*FxS5+Qy_val*FyS5
        end

        gradfh[k*Nd*Nh+m     ] = rho_sum
        gradfh[k*Nd*Nh+m+  Nh] = u_sum
        gradfh[k*Nd*Nh+m+2*Nh] = v_sum
        gradfh[k*Nd*Nh+m+3*Nh] = w_sum
        gradfh[k*Nd*Nh+m+4*Nh] = beta_sum
    end
    end
    return
end

function flux_differencing_z_kernel!(gradfh,Qh,wq,h,J,Qth,Nh,Nq,Nh_P,Np_F,K,Nd,Nq_P,num_threads,Nk_max,Nk_fourier_max)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
    #TODO: using stride to avoid possible insufficient blocks?
    tid_start = (blockIdx().x-1)*blockDim().x + 1 # starting threadid of current block
    tid_end = (blockIdx().x-1)*blockDim().x + num_threads # ending threadid of current block
    k_start = div(tid_start-1,Nq) # starting elementid of current block
    k_end = tid_end < Nq*K ? div(tid_end-1,Nq) : K-1 # ending elementid of current block
    Nk = k_end-k_start+1 # Number of elements in current block
    k_fourier_start = div(tid_start-1,Np_F)
    k_fourier_end = tid_end < Nq*K ? div(tid_end-1,Np_F) : Nq_P*K-1
    Nk_fourier = k_fourier_end-k_fourier_start+1
    
    # TODO: Parallel read
    Qh_shared = @cuDynamicSharedMem(NUM_TYPE,Np_F*Nd*Nk_fourier_max)
    load_size = div(Np_F*Nd*Nk_fourier-1,num_threads)+1 # size of read of each thread
    @inbounds begin
    for i in (threadIdx().x-1)*load_size+1:min(threadIdx().x*load_size,Np_F*Nd*Nk_fourier)
        # TODO: cleanup
        offset_fourier = div(i-1,Np_F*Nd) # distance with k_fourier_start
        n_local = mod1(i,Np_F*Nd)
        n_d = div(n_local-1,Np_F) # Current component
        m_z = mod1(n_local,Np_F) # Current local node index at fourier slice
        i_fourier = k_fourier_start+offset_fourier
        k = div(i_fourier,Nq_P) # Current element
        i_z = mod(i_fourier,Nq_P) # Current local z slice id on the element
        Qh_shared[i] = Qh[k*Nd*Nh+(m_z-1)*Nh_P+i_z+1+n_d*Nh]
    end
    end
    sync_threads()
    
    @inbounds begin
    if index <= Nq*K
        i = index
        k = div(i-1,Nq) # Current element
        m = mod1(i,Nq) # local node number in current element
        i_z = div(m-1,Np_F) # Fourier element id on current element
        m_z = mod1(m,Np_F) # local node number on Fourier element

        k_fourier_offset = div(i-1,Np_F)-k_fourier_start
        rhoL  = Qh_shared[k_fourier_offset*Nd*Np_F+m_z       ]
        uL    = Qh_shared[k_fourier_offset*Nd*Np_F+m_z+  Np_F]
        vL    = Qh_shared[k_fourier_offset*Nd*Np_F+m_z+2*Np_F]
        wL    = Qh_shared[k_fourier_offset*Nd*Np_F+m_z+3*Np_F]
        betaL = Qh_shared[k_fourier_offset*Nd*Np_F+m_z+4*Np_F]
        rhologL = CUDA.log(rhoL)
        betalogL = CUDA.log(betaL)

        rho_sum = 0.0f0
        u_sum = 0.0f0
        v_sum = 0.0f0
        w_sum = 0.0f0
        beta_sum = 0.0f0
        
        # TODO: better way to indexing
        for n_z = 1:Np_F
            rhoR  = Qh_shared[k_fourier_offset*Nd*Np_F+n_z       ]
            uR    = Qh_shared[k_fourier_offset*Nd*Np_F+n_z+  Np_F]
            vR    = Qh_shared[k_fourier_offset*Nd*Np_F+n_z+2*Np_F]
            wR    = Qh_shared[k_fourier_offset*Nd*Np_F+n_z+3*Np_F]
            betaR = Qh_shared[k_fourier_offset*Nd*Np_F+n_z+4*Np_F]

            rhologR = CUDA.log(rhoR)
            betalogR = CUDA.log(betaR)

            _,_,_,_,_,_,_,_,_,_,FzS1,FzS2,FzS3,FzS4,FzS5 = CU_euler_flux(rhoL,uL,vL,wL,betaL,rhoR,uR,vR,wR,betaR,rhologL,betalogL,rhologR,betalogR)

            Qz_val = 2.0f0/J*wq[i_z+1]*Qth[m_z,n_z]
            rho_sum += Qz_val*FzS1
            u_sum += Qz_val*FzS2
            v_sum += Qz_val*FzS3
            w_sum += Qz_val*FzS4
            beta_sum += Qz_val*FzS5
        end
        gradfh[k*Nd*Nh+(m_z-1)*Nh_P+i_z+1     ] += rho_sum
        gradfh[k*Nd*Nh+(m_z-1)*Nh_P+i_z+1+  Nh] += u_sum
        gradfh[k*Nd*Nh+(m_z-1)*Nh_P+i_z+1+2*Nh] += v_sum
        gradfh[k*Nd*Nh+(m_z-1)*Nh_P+i_z+1+3*Nh] += w_sum
        gradfh[k*Nd*Nh+(m_z-1)*Nh_P+i_z+1+4*Nh] += beta_sum
    end
    end
    return
end

function volume_kernel!(gradfh,Qh,rxJ,sxJ,ryJ,syJ,wq,h,J,Qrh_skew,Qsh_skew,Qth,Nh,Nh_P,Np_F,K,Nd,Nq_P,num_threads,Nk_max)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
    #TODO: using stride to avoid possible insufficient blocks?
    tid_start = (blockIdx().x-1)*blockDim().x + 1 # starting threadid of current block
    tid_end = (blockIdx().x-1)*blockDim().x + num_threads # ending threadid of current block
    k_start = div(tid_start-1,Nh) # starting elementid of current block
    k_end = tid_end < Nh*K ? div(tid_end-1,Nh) : K-1 # ending elementid of current block
    Nk = k_end-k_start+1 # Number of elements in current block

    # Parallel read
    Qh_shared = @cuDynamicSharedMem(NUM_TYPE,Nh*Nd*Nk_max)
    load_size = div(Nh*Nd*Nk-1,num_threads)+1 # size of read of each thread
    
    @inbounds begin
    for r_id in (threadIdx().x-1)*load_size+1:min(threadIdx().x*load_size,Nh*Nd*Nk)
        Qh_shared[r_id] = Qh[k_start*Nd*Nh+r_id]
    end
    end
    sync_threads()

    @inbounds begin
    if index <= Nh*K
        i = index
        k = div(i-1,Nh)
        m = mod1(i,Nh)

        k_offset = k-k_start

        rhoL  = Qh_shared[k_offset*Nd*Nh+m     ]
        uL    = Qh_shared[k_offset*Nd*Nh+m+  Nh]
        vL    = Qh_shared[k_offset*Nd*Nh+m+2*Nh]
        wL    = Qh_shared[k_offset*Nd*Nh+m+3*Nh]
        betaL = Qh_shared[k_offset*Nd*Nh+m+4*Nh]
        rhologL = CUDA.log(rhoL)
        betalogL = CUDA.log(betaL)

        t = div(m-1,Nh_P) # Current x-y slice
        s = mod1(m,Nh_P) # Current hybridized node index at x-y slice
        xy_idx = t*Nh_P+1:(t+1)*Nh_P # Nonzero index for Qrh, Qsh
        z_idx = s:Nh_P:s+(Np_F-1)*Nh_P

        rho_sum = 0.0f0
        u_sum = 0.0f0
        v_sum = 0.0f0
        w_sum = 0.0f0
        beta_sum = 0.0f0

        # Assume Affine meshes

        rxJ_val = rxJ[1,1,k+1] 
        sxJ_val = sxJ[1,1,k+1]
        ryJ_val = ryJ[1,1,k+1]
        syJ_val = syJ[1,1,k+1]

        # TODO: better way to indexing
        for n = 1:Nh
            if n in xy_idx || n in z_idx

                rhoR  = Qh_shared[k_offset*Nd*Nh+n     ]
                uR    = Qh_shared[k_offset*Nd*Nh+n+  Nh]
                vR    = Qh_shared[k_offset*Nd*Nh+n+2*Nh]
                wR    = Qh_shared[k_offset*Nd*Nh+n+3*Nh]
                betaR = Qh_shared[k_offset*Nd*Nh+n+4*Nh]
                rhologR = CUDA.log(rhoR)
                betalogR = CUDA.log(betaR)

                FxS1,FxS2,FxS3,FxS4,FxS5,FyS1,FyS2,FyS3,FyS4,FyS5,FzS1,FzS2,FzS3,FzS4,FzS5 = CU_euler_flux(rhoL,uL,vL,wL,betaL,rhoR,uR,vR,wR,betaR,rhologL,betalogL,rhologR,betalogR)

                if n in xy_idx
                    col_idx = mod1(n,Nh_P)
                    Qx_val = 2.0f0*(rxJ_val*Qrh_skew[s,col_idx]+sxJ_val*Qsh_skew[s,col_idx])
                    Qy_val = 2.0f0*(ryJ_val*Qrh_skew[s,col_idx]+syJ_val*Qsh_skew[s,col_idx])
                    rho_sum += Qx_val*FxS1+Qy_val*FyS1
                    u_sum += Qx_val*FxS2+Qy_val*FyS2
                    v_sum += Qx_val*FxS3+Qy_val*FyS3
                    w_sum += Qx_val*FxS4+Qy_val*FyS4
                    beta_sum += Qx_val*FxS5+Qy_val*FyS5
                end
                #=
                if n in z_idx && s <= Nq_P
                    col_idx = div(n-1,Nh_P)+1
                    wqn = 2.0f0/J*wq[s]
                    Qz_val = wqn*Qth[t+1,col_idx]
                    rho_sum += Qz_val*FzS1
                    u_sum += Qz_val*FzS2
                    v_sum += Qz_val*FzS3
                    w_sum += Qz_val*FzS4
                    beta_sum += Qz_val*FzS5
                end
                =#
            end
        end

        gradfh[k*Nd*Nh+m     ] = rho_sum
        gradfh[k*Nd*Nh+m+  Nh] = u_sum
        gradfh[k*Nd*Nh+m+2*Nh] = v_sum
        gradfh[k*Nd*Nh+m+3*Nh] = w_sum
        gradfh[k*Nd*Nh+m+4*Nh] = beta_sum
    end
    end
    return
end

 
# ================================= #
# ============ Routines =========== #
# ================================= #

function rhs(Q,ops,mesh,param,num_threads,compute_rhstest,enable_test)
    Vq,wq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh,Wq = ops
    rxJ,sxJ,ryJ,syJ,nxJ,nyJ,sJ,J,h,mapP,mapP_d = mesh
    K,Np_P,Nq_P,Nfp_P,Nh_P,Np_F,Nq,Nfp,Nh,Nk_max,Nk_tri_max,Nk_fourier_max = param
    
    VU = CUDA.fill(CUDA.zero(NUM_TYPE),Nq*Nd*K)
    Uf = CUDA.fill(CUDA.zero(NUM_TYPE),Nfp*Nd*K)
    QM = CUDA.fill(CUDA.zero(NUM_TYPE),Nfp*Nd*K)
    flux = CUDA.fill(CUDA.zero(NUM_TYPE),Nfp*K*Nd)
    gradfh = CUDA.fill(CUDA.zero(NUM_TYPE),Nh*Nd*K)
    lam = CUDA.fill(CUDA.zero(NUM_TYPE),Nfp*K)
    LFc = CUDA.fill(CUDA.zero(NUM_TYPE),Nfp*K)
    
    # Entropy Projection
    @cuda threads=num_threads blocks = ceil(Int,Nq*K/num_threads) u_to_v!(VU,Q,Nd,Nq)
    synchronize()
    Qh = reshape(Ph*reshape(VU,Nq_P,Np_F*Nd*K),Nh*Nd*K)
    @cuda threads=num_threads blocks = ceil(Int,Nh*K/num_threads) v_to_u!(Qh,Nd,Nh)
    synchronize()
    @cuda threads=num_threads blocks = ceil(Int,Nfp*Nd*K/num_threads) extract_face_val_conservative!(Uf,Qh,Nfp_P,Nq_P,Nh_P)
    # TODO: redundant. should remove 
    UfP = Uf[mapP_d] 
    synchronize()
    @cuda threads=num_threads blocks = ceil(Int,Nh*K/num_threads) u_to_primitive!(Qh,Nd,Nh)
    synchronize()

    # Compute Surface values
    @cuda threads=num_threads blocks = ceil(Int,Nfp*Nd*K/num_threads) extract_face_val!(QM,Qh,Nfp_P,Nq_P,Nh_P)
    synchronize()
    QP = QM[mapP_d]

    # LF dissipation
    @cuda threads=num_threads blocks = ceil(Int,Nfp*K/num_threads) construct_lam!(lam,Uf,nxJ,nyJ,sJ,Nfp,K,Nd)
    synchronize()
    LFc .= .5f0*CUDA.max.(lam,lam[mapP]).*sJ

    # Surface kernel
    @cuda threads=num_threads blocks = ceil(Int,Nfp*K/num_threads) surface_kernel!(flux,QM,QP,Uf,UfP,LFc,nxJ,nyJ,Nfp,K,Nd)
    synchronize()
    flux = reshape(LIFTq*reshape(flux,Nfp_P,Np_F*Nd*K),Nq*Nd*K)

    # Volume kernel
    @cuda threads=num_threads blocks = ceil(Int,Nh*K/num_threads) shmem = sizeof(NUM_TYPE)*Nh_P*Nd*Nk_tri_max flux_differencing_xy_kernel!(gradfh,Qh,rxJ,sxJ,ryJ,syJ,Qrh_skew,Qsh_skew,Nh,Nh_P,Np_F,K,Nd,Nq_P,num_threads,Nk_max,Nk_tri_max)
    @cuda threads=num_threads blocks = ceil(Int,Nq*K/num_threads) shmem = sizeof(NUM_TYPE)*Np_F*Nd*Nk_fourier_max flux_differencing_z_kernel!(gradfh,Qh,wq,h,J,Qth,Nh,Nq,Nh_P,Np_F,K,Nd,Nq_P,num_threads,Nk_max,Nk_fourier_max)
    #@cuda threads=num_threads blocks = ceil(Int,Nh*K/num_threads) shmem = sizeof(NUM_TYPE)*Nh*Nd*Nk_max volume_kernel!(gradfh,Qh,rxJ,sxJ,ryJ,syJ,wq,h,J,Qrh_skew,Qsh_skew,Qth,Nh,Nh_P,Np_F,K,Nd,Nq_P,num_threads,Nk_max)
    synchronize()
    gradf = reshape(VPh*reshape(gradfh,Nh_P,Np_F*Nd*K),Nq*Nd*K)#reshape(Vq*[Pq Lq]*Winv*reshape(gradfh,Nh_P,Np_F*Nd*K),Nq*Nd*K)

    # Combine
    rhsQ = -(gradf+flux)

    # Compute rhstest
    rhstest = 0
    if compute_rhstest
        rhstest = CUDA.sum(Wq*reshape(VU.*rhsQ,Nq_P,Np_F*Nd*K))
    end

    if enable_test
        @show CUDA.maximum(VU)
        @show CUDA.minimum(VU)
        @show CUDA.sum(VU)
        @show CUDA.maximum(Qh)
        @show CUDA.minimum(Qh)
        @show CUDA.sum(Qh)
        @show CUDA.maximum(QM)
        @show CUDA.minimum(QM)
        @show CUDA.sum(QM)
        @show CUDA.maximum(QP)
        @show CUDA.minimum(QP)
        @show CUDA.sum(QP)
        @show CUDA.maximum(lam)
        @show CUDA.minimum(lam)
        @show CUDA.sum(lam)
        @show CUDA.maximum(LFc)
        @show CUDA.minimum(LFc)
        @show CUDA.sum(LFc)
        @show CUDA.maximum(flux)
        @show CUDA.minimum(flux)
        @show CUDA.sum(flux)
        @show CUDA.maximum(gradf)
        @show CUDA.minimum(gradf)
        @show CUDA.sum(gradf)
        @show CUDA.maximum(rhsQ)
        @show CUDA.minimum(rhsQ)
        @show CUDA.sum(rhsQ)
        @show rhstest
        @show typeof(Q)
        @show typeof(VU)
        @show typeof(Qh)
        @show typeof(QM)
        @show typeof(QP)
        @show typeof(flux)
        @show typeof(gradf)
        @show typeof(rhsQ)
    end
    return rhsQ,rhstest
end

xq,yq,zq = (x->reshape(x,Nq_P,Np_F,K)).((xq,yq,zq))
ρ_exact(x,y,z,t) = @. 1.0f0+0.2f0*sin(pi*(x+y+z-3/2*t))
ρ = @. 1.0f0+0.2f0*sin(pi*(xq+yq+zq))
u = ones(size(xq))
v = -0.5f0*ones(size(xq))
w = ones(size(xq))
p = ones(size(xq))
Q_exact(x,y,z,t) = (ρ_exact(x,y,z,t),ones(size(x)),-0.5f0*ones(size(x)),ones(size(x)),ones(size(x)))

Q = primitive_to_conservative(ρ,u,v,w,p)
Q = collect(Q)
Q_vec = zeros(Nq_P,Np_F,Nd,K)
Q_ex_vec = zeros(Nq_P,Np_F,Nd,K)
# TODO: clean up
rq2,sq2,wq2 = quad_nodes_2D(N_P+2)
Vq2 = vandermonde_2D(N_P,rq2,sq2)/VDM
xq2,yq2,zq2 = (x->Vq2*reshape(x,Np_P,Np_F*K)).((x,y,z))
Q_ex = Q_exact(xq2,yq2,zq2,T)

for k = 1:K
    for d = 1:5
        @. Q_vec[:,:,d,k] = Q[d][:,:,k]
    end
end
Q = Q_vec[:]
Q = convert.(NUM_TYPE,Q)
Q = CuArray(Q)
resQ = CUDA.fill(0.0f0,Nq*Nd*K)

################################
######   Time stepping   #######
################################

@time begin

@inbounds begin
for i = 1:Nsteps
    rhstest = 0

    for INTRK = 1:5
        if enable_test
            @show "==============="
            @show INTRK
            @show "==============="
        end
        compute_rhstest = INTRK==5
        rhsQ,rhstest = rhs(Q,ops,mesh,param,num_threads,compute_rhstest,enable_test)
        resQ .= rk4a[INTRK]*resQ+dt*rhsQ
        Q .= Q + rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest") 
    end
end
end # end @inbounds

end # end @time


Q = Array(Q)
Q = reshape(Q,Nq_P,Np_F,Nd,K)
Q_mat = [zeros(Nq_P,Np_F,K),zeros(Nq_P,Np_F,K),zeros(Nq_P,Np_F,K),zeros(Nq_P,Np_F,K),zeros(Nq_P,Np_F,K)]
for k = 1:K
    for d = 1:5
        Q_mat[d][:,:,k] = Q[:,:,d,k]
    end
end
Pq = Array(Pq)
Q = (x->Vq2*Pq*reshape(x,Nq_P,Np_F*K)).(Q_mat)
p = pfun(Q[1],(Q[2],Q[3],Q[4]),Q[5])
Q = (Q[1],Q[2]./Q[1],Q[3]./Q[1],Q[4]./Q[1],p)
Q_ex = (x->reshape(x,length(rq2),Np_F*K)).(Q_ex)
L2_err = 0.0
for fld in 1:Nd
    global L2_err
    L2_err += sum(h*J*wq2.*(Q[fld]-Q_ex[fld]).^2)
end
println("L2err at final time T = $T is $L2_err\n")

#=
@show maximum(Q[1])
@show maximum(Q[2])
@show maximum(Q[3])
@show maximum(Q[4])
@show maximum(Q[5])
@show minimum(Q[1])
@show minimum(Q[2])
@show minimum(Q[3])
@show minimum(Q[4])
@show minimum(Q[5])
@show sum(Q[1])
@show sum(Q[2])
@show sum(Q[3])
@show sum(Q[4])
@show sum(Q[5])
@show maximum(Q[5])
=#
@show num_threads
@show Nd*Nh_P*Nk_tri_max
@show Nd*Np_F*Nk_fourier_max
