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
const sp_tol = 1e-12
# const has_gpu = CUDAapi.has_cuda_gpu()
const enable_test = false 
"Program parameters"
compute_L2_err = false

"Approximation Parameters"
N_P   = 2;    # The order of approximation in polynomial dimension
Np_P  = Int((N_P+1)*(N_P+2)/2)
Np_F  = 10;    # The order of approximation in Fourier dimension
K1D   = 10;   # Number of elements in polynomial (x,y) dimension
Nd = 5;
CFL   = 1.0;
T     = 0.5;  # End time

"Time integration Parameters"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N_P+1)*(N_P+2)*3/2  # estimated trace constant for CFL
dt = CFL * 2 / CN / K1D
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Initialize Reference Element in Fourier dimension"
h = 2*pi/Np_F
column = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)]
column2 = [-pi^2/3/h^2-1/6; -((-1).^(1:Np_F-1)./(2*(sin.((1:Np_F-1)*h/2)).^2))]
Dt = Array{Float64,2}(Toeplitz(column,column[[1;Np_F:-1:2]]))
D2t = Array{Float64,2}(Toeplitz(column2,column2[[1;Np_F:-1:2]]))
t = LinRange(h,2*pi,Np_F)


"Initialize Reference Element in polynomial dimension"
rd = init_reference_tri(N_P);
@unpack fv,Nfaces,r,s,VDM,V1,Dr,Ds,rf,sf,wf,nrJ,nsJ,rq,sq,wq,Vq,M,Pq,Vf,LIFT = rd
Nq_P = length(rq)
Nfp_P = length(rf)
Nh_P = Nq_P+Nfp_P # Number of hybridized points
Lq = LIFT

"Mesh related variables"
# First initialize 2D triangular mesh
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
@. VX = 1+VX
@. VY = 1+VY
md = init_mesh((VX,VY),EToV,rd)
VX = repeat(VX,2)
VY = repeat(VY,2)
VZ = [2/Np_F*ones((K1D+1)*(K1D+1),1); 2*ones((K1D+1)*(K1D+1),1)]
EToV = [EToV EToV.+(K1D+1)*(K1D+1)]

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

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
Ph = [Vq;Vf]*Pq # TODO: refactor

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

# TODO: refactor
Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh = (x->CuArray(x)).((Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh))
Nq = Nq_P*Np_F
Nh = Nh_P*Np_F
Nfp = Nfp_P*Np_F
mapP_vec = zeros(Int64,Nfp_P,Np_F,Nd,K)

for k = 1:K
    for j = 1:Np_F
        for i = 1:Nfp_P
            val = mapP[i,j,k]
            elem = div(val-1,Nfp)
            n = mod1(val,Nfp)
            mapP_vec[i,j,1,k] = elem*Nd*Nfp+n
            mapP_vec[i,j,2,k] = elem*Nd*Nfp+n+Nfp
            mapP_vec[i,j,3,k] = elem*Nd*Nfp+n+2*Nfp
            mapP_vec[i,j,4,k] = elem*Nd*Nfp+n+3*Nfp
            mapP_vec[i,j,5,k] = elem*Nd*Nfp+n+4*Nfp
        end
    end
end
mapP_vec = mapP_vec[:]
nxJ = CuArray(nxJ[:])
nyJ = CuArray(nyJ[:])

rxJ = CuArray(rxJ)
sxJ = CuArray(sxJ)
ryJ = CuArray(ryJ)
syJ = CuArray(syJ)

Winv = CuArray(diagm(1 ./ [Array(wq);Array(wf)])) # TODO: ugly
Wq = CuArray(diagm(Array(wq)))
 
# ================================= #
# ============ Routines =========== #
# ================================= #
function u_to_v!(VU,Q,Nd,Nq)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
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
        sU = CUDA.log(0.4*rhoe/CUDA.exp(1.4*CUDA.log(rho)))

        VU[idx] = (-E+rhoe*(2.4-sU))/rhoe
        VU[idx+Nq] = rhou/rhoe
        VU[idx+2*Nq] = rhov/rhoe
        VU[idx+3*Nq] = rhow/rhoe
        VU[idx+4*Nq] = -rho/rhoe
    end
end

function v_to_u!(Qh,Nd,Nh)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
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
        rhoeV = CUDA.exp(2.5*CUDA.log(0.4/CUDA.exp(1.4*CUDA.log(-E))))*CUDA.exp(-(1.4-rho+vUnorm/(2*E))/0.4)

        Qh[idx] = -rhoeV*E
        Qh[idx+Nh] = rhoeV*rhou
        Qh[idx+2*Nh] = rhoeV*rhov
        Qh[idx+3*Nh] = rhoeV*rhow
        Qh[idx+4*Nh] = rhoeV*(1-vUnorm/(2*E))
    end
end

#TODO: combine with v_to_u
function u_to_primitive!(Qh,Nd,Nh)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
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

        beta = rho/(0.8*(E-.5*(rhou^2+rhov^2+rhow^2)/rho))

        Qh[idx+Nh] = rhou/rho
        Qh[idx+2*Nh] = rhov/rho
        Qh[idx+3*Nh] = rhow/rho
        Qh[idx+4*Nh] = beta
    end
end

function extract_face_val!(QM,Qh,Nfp_P,Nq_P,Nh_P)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
    for i = index:stride:length(QM)
        k = div(i-1,Nfp_P)
        n = mod1(index,Nfp_P)
        QM[i] = Qh[k*Nh_P+Nq_P+n]
    end
end

function CU_logmean(uL,uR,logL,logR)
    da = uR-uL
    aavg = .5*(uL+uR)
    f = da/aavg
    if CUDA.abs(f)<1e-4
        v = f^2
        return aavg*(1 + v*(-.2-v*(.0512 - v*0.026038857142857)))
    else
        return -da/(logL-logR)
    end
end

function CU_euler_flux(rhoM,uM,vM,wM,betaM,rhoP,uP,vP,wP,betaP,rhologM,betalogM,rhologP,betalogP)
    rholog = CU_logmean(rhoM,rhoP,rhologM,rhologP)
    betalog = CU_logmean(betaM,betaP,betalogM,betalogP)

    # TODO: write in functions
    rhoavg = .5*(rhoM+rhoP)
    uavg = .5*(uM+uP)
    vavg = .5*(vM+vP)
    wavg = .5*(wM+wP)

    unorm = uM*uP+vM*vP+wM*wP
    pa = rhoavg/(betaM+betaP)
    E_plus_p = rholog/(0.8*betalog) + pa + .5*rholog*unorm

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

function surface_kernel!(flux,QM,QP,nxJ,nyJ,Nfp,K,Nd)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
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

        rhologM = CUDA.log(rhoM)
        rhologP = CUDA.log(rhoP)
        betalogM = CUDA.log(betaM)
        betalogP = CUDA.log(betaP)

        FxS1,FxS2,FxS3,FxS4,FxS5,FyS1,FyS2,FyS3,FyS4,FyS5,_,_,_,_,_ = CU_euler_flux(rhoM,uM,vM,wM,betaM,rhoP,uP,vP,wP,betaP,rhologM,betalogM,rhologP,betalogP)
        flux[k*Nd*Nfp+n      ] = nxJ_val*FxS1+nyJ_val*FyS1
        flux[k*Nd*Nfp+n+  Nfp] = nxJ_val*FxS2+nyJ_val*FyS2
        flux[k*Nd*Nfp+n+2*Nfp] = nxJ_val*FxS3+nyJ_val*FyS3
        flux[k*Nd*Nfp+n+3*Nfp] = nxJ_val*FxS4+nyJ_val*FyS4
        flux[k*Nd*Nfp+n+4*Nfp] = nxJ_val*FxS5+nyJ_val*FyS5
    end
end

function volume_kernel!(gradfh,Qh,rxJ,sxJ,ryJ,syJ,wq,h,J,Qrh_skew,Qsh_skew,Qth,Nh,Nh_P,Np_F,K,Nd,Nq_P)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
    for i = index:stride:Nh*K
        # TODO: load shared memory
        k = div(i-1,Nh)
        m = mod1(i,Nh)

        rhoL  = Qh[k*Nd*Nh+m     ]
        uL    = Qh[k*Nd*Nh+m+  Nh]
        vL    = Qh[k*Nd*Nh+m+2*Nh]
        wL    = Qh[k*Nd*Nh+m+3*Nh]
        betaL = Qh[k*Nd*Nh+m+4*Nh]
        rhologL = CUDA.log(rhoL)
        betalogL = CUDA.log(betaL)

        t = div(m-1,Nh_P) # Current x-y slice
        s = mod1(m,Nh_P) # Current hybridized node index at x-y slice
        xy_idx = t*Nh_P+1:(t+1)*Nh_P # Nonzero index for Qrh, Qsh
        z_idx = s:Nh_P:s+(Np_F-1)*Nh_P

        rho_sum = 0.0
        u_sum = 0.0
        v_sum = 0.0
        w_sum = 0.0
        beta_sum = 0.0

        # Assume Affine meshes
        rxJ_val = rxJ[1,1,k+1] 
        sxJ_val = sxJ[1,1,k+1]
        ryJ_val = ryJ[1,1,k+1]
        syJ_val = syJ[1,1,k+1]

        # TODO: better way to indexing
        for n = 1:Nh
            if n in xy_idx || n in z_idx
                rhoR  = Qh[k*Nd*Nh+n     ]
                uR    = Qh[k*Nd*Nh+n+  Nh]
                vR    = Qh[k*Nd*Nh+n+2*Nh]
                wR    = Qh[k*Nd*Nh+n+3*Nh]
                betaR = Qh[k*Nd*Nh+n+4*Nh]
                rhologR = CUDA.log(rhoR)
                betalogR = CUDA.log(betaR)

                FxS1,FxS2,FxS3,FxS4,FxS5,FyS1,FyS2,FyS3,FyS4,FyS5,FzS1,FzS2,FzS3,FzS4,FzS5 = CU_euler_flux(rhoL,uL,vL,wL,betaL,rhoR,uR,vR,wR,betaR,rhologL,betalogL,rhologR,betalogR)

                if n in xy_idx

                    col_idx = mod1(n,Nh_P)
                    Qx_val = 2*(rxJ_val*Qrh_skew[s,col_idx]+sxJ_val*Qsh_skew[s,col_idx])
                    Qy_val = 2*(ryJ_val*Qrh_skew[s,col_idx]+syJ_val*Qsh_skew[s,col_idx])
                    rho_sum += Qx_val*FxS1+Qy_val*FyS1
                    u_sum += Qx_val*FxS2+Qy_val*FyS2
                    v_sum += Qx_val*FxS3+Qy_val*FyS3
                    w_sum += Qx_val*FxS4+Qy_val*FyS4
                    beta_sum += Qx_val*FxS5+Qy_val*FyS5
                end
                
                # TODO: fix  
                if n in z_idx && s <= Nq_P
                    col_idx = div(n-1,Nh_P)+1
                    wqn = 2/J*wq[s]
                    Qz_val = wqn*Qth[t+1,col_idx]
                    rho_sum += Qz_val*FzS1
                    u_sum += Qz_val*FzS2
                    v_sum += Qz_val*FzS3
                    w_sum += Qz_val*FzS4
                    beta_sum += Qz_val*FzS5
                end  
            end
        end
        gradfh[k*Nd*Nh+m     ] = rho_sum
        gradfh[k*Nd*Nh+m+  Nh] = u_sum
        gradfh[k*Nd*Nh+m+2*Nh] = v_sum
        gradfh[k*Nd*Nh+m+3*Nh] = w_sum
        gradfh[k*Nd*Nh+m+4*Nh] = beta_sum
        
    end
end

# ================================= #
# ============ Routines =========== #
# ================================= #



ops = (Vq,wq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh,Wq)
mesh = (rxJ,sxJ,ryJ,syJ,nxJ,nyJ,J,h,mapP_vec)
param = (K,Np_P,Nq_P,Nfp_P,Nh_P,Np_F,Nq,Nfp,Nh)
function rhs(Q,ops,mesh,param,compute_rhstest,enable_test)
    Vq,wq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh,Wq = ops
    rxJ,sxJ,ryJ,syJ,nxJ,nyJ,J,h,mapP_vec = mesh
    K,Np_P,Nq_P,Nfp_P,Nh_P,Np_F,Nq,Nfp,Nh = param
    
    # Entropy Projection
    VU = CUDA.fill(0.0,length(Q))
    num_blocks = ceil(Int,length(VU)/256/Nd)
    @cuda threads=256 blocks = num_blocks u_to_v!(VU,Q,Nd,Nq)
    synchronize()

    Qh = reshape(Ph*reshape(VU,Nq_P,Np_F*Nd*K),Nh*Nd*K)
    synchronize()

    num_blocks = ceil(Int,length(Qh)/256/Nd)
    @cuda threads=256 blocks = num_blocks v_to_u!(Qh,Nd,Nh)
    synchronize()

    @cuda threads=256 blocks = num_blocks u_to_primitive!(Qh,Nd,Nh)
    synchronize()

    # Compute Surface values
    QM = CUDA.fill(0.0,Nfp*Nd*K)
    num_blocks = ceil(Int,length(QM)/256)
    @cuda threads=256 blocks = num_blocks extract_face_val!(QM,Qh,Nfp_P,Nq_P,Nh_P)
    synchronize()
    QP = QM[mapP_vec]
    synchronize()

    # Surface kernel
    flux = CUDA.fill(0.0,Nfp*K*Nd)
    num_blocks = ceil(Int,length(flux)/256/Nd)
    @cuda threads=256 blocks = num_blocks surface_kernel!(flux,QM,QP,nxJ,nyJ,Nfp,K,Nd)
    synchronize()
    flux = reshape(LIFTq*reshape(flux,Nfp_P,Np_F*Nd*K),Nq*Nd*K)
    synchronize()

    # Volume kernel
    # TODO: implement share memory
    gradfh = CUDA.fill(0.0,Nh*Nd*K)
    num_blocks = ceil(Int,Nh*K/256)
    @cuda threads=256 blocks = num_blocks volume_kernel!(gradfh,Qh,rxJ,sxJ,ryJ,syJ,wq,h,J,Qrh_skew,Qsh_skew,Qth,Nh,Nh_P,Np_F,K,Nd,Nq_P)
    synchronize()
    gradf = reshape(VPh*reshape(gradfh,Nh_P,Np_F*Nd*K),Nq*Nd*K)#reshape(Vq*[Pq Lq]*Winv*reshape(gradfh,Nh_P,Np_F*Nd*K),Nq*Nd*K)
    synchronize()

    # Combine
    rhsQ = -(gradf+flux)
    synchronize()

    # Compute rhstest
    rhstest = 0
    compute_rhstest = true
    if compute_rhstest
        rhstest = CUDA.sum(Wq*reshape(VU.*rhsQ,Nq_P,Np_F*Nd*K))
        synchronize()
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
    end
    return rhsQ,rhstest
end

xq,yq,zq = (x->reshape(x,Nq_P,Np_F,K)).((xq,yq,zq))
ρ_exact(x,y,z,t) = @. 1+0.2*sin(pi*(x+y+z-3/2*t))
ρ = @. 1+0.2*sin(pi*(xq+yq+zq))
u = ones(size(xq))
v = -1/2*ones(size(xq))
w = ones(size(xq))
p = ones(size(xq))
Q_exact(x,y,z,t) = (ρ_exact(x,y,z,t),u,v,w,p)

Q = primitive_to_conservative(ρ,u,v,w,p)
Q = collect(Q)
Q_vec = zeros(Nq_P,Np_F,Nd,K)
Q_ex_vec = zeros(Nq_P,Np_F,Nd,K)
# TODO: clean up
rq2,sq2,wq2 = quad_nodes_2D(N_P+2)
Vq2 = vandermonde_2D(N_P,rq2,sq2)/VDM
xq2,yq2,zq2 = (x->Vq2*reshape(x,Nq_P,Np_F*K)).((x,y,z))
Q_ex = Q_exact(xq2,yq2,zq2,T)

for k = 1:K
    for d = 1:5
        @. Q_vec[:,:,d,k] = Q[d][:,:,k]
    end
end
Q = Q_vec[:]
Q = CuArray(Q)
resQ = CUDA.fill(0.0,Nq*Nd*K)


# rhs(Q,ops,mesh,param,true,true)
@time begin
for i = 1:Nsteps
    rhstest = 0
    for INTRK = 1:5
        if enable_test
            @show "==============="
            @show INTRK
            @show "==============="
        end
        compute_rhstest = INTRK==5
        rhsQ,rhstest = rhs(Q,ops,mesh,param,compute_rhstest,enable_test)
        resQ .= rk4a[INTRK]*resQ+dt*rhsQ
        Q .= Q + rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest") 
    end
end

end


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
Q_ex = (x->reshape(x,Nq_P,Np_F*K)).(Q_ex)

L2_err = 0.0
for fld in 1:Nd
    global L2_err
    L2_err += sum(h*J*wq2.*(Q[fld]-Q_ex[fld]).^2)
end
println("L2err at final time T = $T is $L2_err\n")
