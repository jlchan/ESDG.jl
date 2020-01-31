using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N    = 3 # The order of approximation
Kvec = [4 8 16 32 64 128 256] # mesh resolutions
err = zeros(length(Kvec))

for kk = 1:length(Kvec) # number of elements
    K = Kvec[kk]
    T   = 2 # endtime
    CFL = .5
    tau = 1 # upwind penalty parameter

    "Mesh related variables"
    VX = LinRange(-1,1,K+1)
    EToV = repeat([0 1],K,1) + repeat(1:K,1,2)

    "Construct matrices on reference elements"
    r,w = gauss_lobatto_quad(0,0,N)
    V = vandermonde_1D(N, r)
    Dr = grad_vandermonde_1D(N, r)/V
    M = inv(V*V')

    "Nodes on faces, and face node coordinate"
    Vf = vandermonde_1D(N,[-1;1])/V
    B = zeros(N+1,2)
    B[1,1] = 1
    B[N+1,2] = 1
    LIFT = M\B # lift matrix

    "Construct global coordinates"
    V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
    x = V1*VX[transpose(EToV)]

    "Connectivity maps"
    xf = Vf*x
    mapM = reshape(1:2*K,2,K)
    mapP = copy(mapM)
    mapP[1,2:end] .= mapM[2,1:end-1]
    mapP[2,1:end-1] .= mapM[1,2:end]

    "Make maps periodic"
    mapP[1] = mapM[end]
    mapP[end] = mapM[1]

    "Geometric factors and surface normals"
    J = repeat(transpose(diff(VX)/2),N+1,1)
    rxJ = 1
    nxJ = repeat([-1;1],1,K)

    "initial conditions"
    u0(x) = @. sin(pi*x)
    u = u0(x)

    "Time integration"
    rk4a,rk4b,rk4c = rk45_coeffs()
    dx = minimum(minimum(diff(x,dims=1),dims=1))
    dt = CFL * dx
    Nsteps = convert(Int,ceil(T/dt))
    dt = T/Nsteps

    "pack arguments into tuples"
    ops = (Dr,LIFT,Vf)
    vgeo = (rxJ,J)
    fgeo = (nxJ,)

    function rhs(u,ops,vgeo,fgeo,mapP)
        # unpack args
        Dr,LIFT,Vf = ops
        rxJ,J = vgeo
        nxJ, = fgeo

        uf = Vf*u # can replace with nodal extraction
        uavg = .5*(uf[mapP]+uf)
        uflux = uavg - uf
        du = uf[mapP] - uf

        ux = rxJ*(Dr*u)
        rhsu = ux + LIFT*(@. uflux*nxJ - .5*tau*abs(nxJ)*du)

        return -rhsu./J
    end

    resu = zeros(size(x))
    for i = 1:Nsteps
        for INTRK = 1:5
            rhsu = rhs(u,ops,vgeo,fgeo,mapP)
            @. resu = rk4a[INTRK]*resu + dt*rhsu
            @. u   += rk4b[INTRK]*resu
        end
    end

    # compute L2 errors
    rq,wq = gauss_quad(0,0,N+8)
    Vq = vandermonde_1D(N,rq)/V
    xq = Vq*x
    wJq = diagm(wq)*(Vq*J)
    err[kk] = sqrt(sum(wJq.*(Vq*u - @. u0(xq)).^2))
    # err[kk] = maximum(@. abs(u - u0(x))) # computes discrete max error

end

gr(size=(300,300),markerstrokewidth=1,markersize=2,
        xlabel="Mesh size h",ylabel="L2 error",legend=:bottomright)

h = @. 2/Kvec[:]
plot(h,err,
        xaxis=:log,yaxis=:log,
        linestyle=:dash,
        markershape = :hexagon,markersize = 6,label="Error")

plot!(h,h.^(N+1)*err[end]/h[end]^(N+1),
        xaxis=:log,yaxis=:log,
        linestyle=:dash,label="h^$(N+1)")
