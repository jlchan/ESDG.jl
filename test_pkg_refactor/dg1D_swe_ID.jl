using Plots
using LinearAlgebra
using ForwardDiff
using SparseArrays
using StaticArrays
using UnPack

using NodesAndModes

"Approximation parameters"
N   = 3 # The order of approximation
K1D = 8
CFL = 1/4
T   = 0.5 # endtime

"Mesh related variables"
VX = LinRange(-1,1,K1D+1)
EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]),2,K1D))

"Construct matrices on reference elements"
r,w = gauss_lobatto_quad(0,0,N)

avg(a,b) = .5*(a+b)
function fS1D_LF(UL,UR)
    hL,huL = UL
    hR,huR = UR
    uL = huL./hL
    uR = huR./hR
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL*uL,huR*uR) + .5*avg(hL*hL,hR*hR)
    return fxS1,fxS2
end

function fSWE(h,hu)
    u = hu./h
    return hu, hu*u + .5*g*h^2
end


function build_rhs_matrix(applyRHS,Np,K,vargs...)
    u = zeros(Np,K)
    A = spzeros(Np*K,Np*K)
    for i in eachindex(u)
        u[i] = one(eltype(u))
        r_i = applyRHS(u,vargs...)
        A[:,i] = droptol!(sparse(r_i[:]),1e-12)
        u[i] = zero(eltype(u))
    end
    return A
end

# Q,E,B,ψ = make_meshfree_ops(r, w)
Q = diagm(1=>ones(N),-1=>-ones(N))
Q[1,1] = -1
Q[N+1,N+1] = 1
B = [-1 0; 0 1]
E = zeros(2,N+1)
E[1,1] = 1
E[2,N+1] = 1

Vf = E

Q_skew = .5*(Q-transpose(Q))

"Nodes on faces, and face node coordinate"
wf = [1;1]

"Construct global coordinates"
V1 = Line.vandermonde(1,r)/Line.vandermonde(1,[-1;1])
x = V1*VX[transpose(EToV)]

"Connectivity maps"
xf = E*x
mapM = reshape(1:2*K1D,2,K1D)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

"Make periodic"
mapP[1] = mapM[end]
mapP[end] = mapM[1]

"Geometric factors and surface normals"
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K1D)
rxJ = 1

M_inv = diagm(1 ./(w.*J[1]))
M_inv = spdiagm(0 => 1 ./vec(diagm(w)*J))

"initial conditions"
# h = @. exp(-25*x^2)+2
# u = @. exp(-25*x^2)
# h = h*0 .+2
const g = 1
h = [0.43012     0.00283327   4.32944e-6  0.00587629   0.461943  0.989807  0.99953   0.999644;
 0.0737789   0.000115821  1.01984e-8  0.000110116  0.937357  0.99886   1.00033   1.0128;
 1.80755e-8  1.06873e-8   8.2332e-5   0.0724844    1.01279   1.00033   0.99886   0.937346;
 0.00550536  5.01346e-6   0.00173385  0.431363     0.999649  0.99953   0.989807  0.456408]
hu= [0.554957    0.00709513  -1.41038e-6   -0.00332459  -0.277515   -0.0101393     0.000147021   0.0195437;
  -0.00350728  9.98422e-5   0.0          -7.46314e-5  -0.0800892  -0.00114606   -0.000350967  -0.0246873;
   0.0         0.0         -4.36845e-5   -0.0474182    0.0246741   0.000350967   0.00114606    0.0801153;
   0.109488    2.80551e-6  -0.000938431  -0.279944    -0.0195245  -0.00014702    0.0101397     0.278559]

# h[:,1:convert(Int,K1D/2)] .= 1e-10
# h[:,convert(Int,K1D/2)+1:K1D] .= 1.0
# hu = h*0;

"Time integration"
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dT = T/Nsteps

"pack arguments into tuples - will "
ops = (Q_skew,E,B,M_inv, Vf)
vgeo = (rxJ,J)
fgeo = (nxJ)
nodemaps = (mapM, mapP)

function rhsx(u,ops,vgeo,fgeo,nodemaps)
    # unpack args
    Q,E,B,M_inv,Vf = ops
    rxJ,J = vgeo
    nxJ = fgeo
    (mapM,mapP) = nodemaps

    uM = Vf*u # can replace with nodal extraction
    dudx = rxJ.*(Q*u)
    rhsu = dudx + transpose(E)*(.5*nxJ.*uM[mapP])
    return rhsu
end

Qx = build_rhs_matrix(rhsx,size(h,1),size(h,2),ops,vgeo,fgeo,nodemaps)


function rhs(h,hu,m,Qx)
    rows = rowvals(Qx)
    vals = nonzeros(Qx)

    rhsh = zero.(h)
    rhshu = zero.(hu)
    for j = 1:size(Qx,2) # loop over columns
        hj = h[j]
        huj = hu[j]
        for index in nzrange(Qx,j) # loop over rows
            i = rows[index]
            hi = h[i]
            hui = hu[i]
            Qxij = vals[index]
            lambdaj = abs(huj./hj) + sqrt.(g.*hj)
            lambdai = abs(hui./hi) + sqrt.(g.*hi)
            dij = abs.(Qxij)*max(lambdaj, lambdai)
            #Fx1, Fx2 = fS1D_LF((hj, huj), (hj, huj))
            Fx1,Fx2 = avg.(fSWE(hi,hui),fSWE(hj,huj))
            rhsh[i]  += Qxij*Fx1  - dij * (hj - hi)
            rhshu[i] += Qxij*Fx2  - dij * (huj- hui)
        end
    end
    return -rhsh./m, -rhshu./m
end


# lambda = maximum(abs.(hu./h)+sqrt.(g.*h))
# dt1 = minimum(vec(diagm(w)*J))/(2*lambda)
# rhsh1, rhshu1  = rhs(h,hu,diagm(w)*J,Qx)
# htmp  = h  + dt1*rhsh1
# hutmp = hu + dt1*rhshu1
#
# @show minimum(htmp)

"plotting nodes"
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(x,h)

resu = zeros(size(x))
h = vec(h)
hu = vec(hu)
t = 0
@gif for i = 1:100000
    @show i, t
    global h, hu, t
    h = vec(h)
    hu = vec(hu)
    # for INTRK = 1:1
    #     # rhsu = rhs(u,ops,vgeo,fgeo,mapP)
    #     rhsu = rhs(u,M_inv,Qx)
    #     @. resu = rk4a[INTRK]*resu + dt*rhsu
    #     @. u   += rk4b[INTRK]*resu
    # end
    # Heun's method - this is an example of a 2nd order SSP RK method
    lambda = maximum(abs.(hu./h)+sqrt.(g.*h))
    dt1 = min(T-t, minimum(w*J[1])/(2*lambda), dT);
    rhsh1, rhshu1  = rhs(h,hu,M_inv,Qx)
    htmp  = h  + dt1*rhsh1
    hutmp = hu + dt1*rhshu1
    lambda = maximum(abs.(hutmp./htmp)+sqrt.(g.*htmp))
    dt2 = min(T-t, minimum(w*J[1])/(2*lambda), dT);
    while dt2<dt1
        dt1 = dt2
        htmp  = h  + dt1*rhsh1
        hutmp = hu + dt1*rhshu1
        lambda = maximum(abs.(hutmp./htmp)+sqrt.(g.*htmp))
        dt2 = min(T-t, minimum(w)/(2*lambda), dT);
    end
    dt = min(dt1, dt2)
    rhsh2 , rhshu2 = rhs(htmp, hutmp, M_inv,Qx)
    h  .+= .5*dt*(rhsh1 + rhsh2)
    hu .+= .5*dt*(rhshu1 + rhshu2)

    t +=dt
    if t>=T
            break
    end
    @show L1, L2
    i +=1
    if i%10==0
        h = reshape(h,size(x,1),size(x,2))
        println("Number of time steps $i out of $Nsteps")
        # display(plot(Vp*x,Vp*u,ylims=(-.1,1.1)))
        # push!(plt, Vp*x,Vp*u,ylims=(-.1,1.1))
        plot(Vp*x,Vp*h,ylims=(-.1,2),title="Timestep $i out of $Nsteps",lw=2)
        scatter!(x,h)
        # sleep(.0)
    end
end every 5

# plot(Vp*x,Vp*u,ylims=(-.1,1.1))
