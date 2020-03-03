using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

"Define approximation parameters"
N   = 2 # The order of approximation
Kvec = [4 8 16 32]
err = zeros(length(Kvec))
for kk = 1:length(Kvec)
    K1D = Kvec[kk] # number of elements along each edge of a rectangle
    CFL = .5 # relative size of a time-step
    T   = 2/3 # final time

    "Define mesh and compute connectivity
    - (VX,VY) are and EToV is a connectivity matrix
    - connect_mesh computes a vector FToF such that face i is connected to face j=FToF[i]"
    VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
    FToF = connect_mesh(EToV,tri_face_vertices())
    Nfaces,K = size(FToF)

    "Construct matrices on reference elements
    - r,s are vectors of interpolation nodes
    - V is the matrix arising in polynomial interpolation, e.g. solving V*u = [f(x_1),...,f(x_Np)]
    - inv(V) transforms from a nodal basis to an orthonormal basis.
    - If Vr evaluates derivatives of the orthonormal basis at nodal points
    - then, Vr*inv(V) = Vr/V transforms nodal values to orthonormal coefficients, then differentiates the orthonormal basis"
    r, s = nodes_2D(N)
    V = vandermonde_2D(N, r, s)
    Vr, Vs = grad_vandermonde_2D(N, r, s)
    invM = (V*V')
    Dr = Vr/V
    Ds = Vs/V

    "Nodes on faces, and face node coordinate
    - r1D,w1D are quadrature nodes and weights
    - rf,sf = mapping 1D quad nodes to the faces of a triangle"
    r1D, w1D = gauss_quad(0,0,N)
    Nfp = length(r1D) # number of points per face
    e = ones(Nfp,1) # vector of all ones
    z = zeros(Nfp,1) # vector of all zeros
    rf = [r1D; -r1D; -e];
    sf = [-e; r1D; -r1D];
    wf = vec(repeat(w1D,3,1));
    Vf = vandermonde_2D(N,rf,sf)/V # interpolates from nodes to face nodes
    LIFT = invM*(transpose(Vf)*diagm(wf)) # lift matrix used in rhs evaluation

    "Construct global coordinates"
    #- vx = VX[EToV'] = a 3xK matrix of vertex values for each element
    #- V1*vx uses the linear polynomial defined by 3 vertex values to interpolate nodal points on the reference element to physical elements"
    r1,s1 = nodes_2D(1)
    V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
    x = V1*VX[transpose(EToV)]
    y = V1*VY[transpose(EToV)]

    "Compute connectivity maps: uP = exterior value used in DG numerical fluxes"
    xf,yf = (x->Vf*x).((x,y))
    mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
    mapM = reshape(mapM,Nfp*Nfaces,K)
    mapP = reshape(mapP,Nfp*Nfaces,K)

    "Make boundary maps periodic"
    LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
    mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
    mapP[mapB] = mapPB

    "Compute geometric factors and surface normals"
    rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)

    "nhat = (nrJ,nsJ) are reference normals scaled by edge length
    - physical normals are computed via G*nhat, G = matrix of geometric terms
    - sJ is the normalization factor for (nx,ny) to be unit vectors"
    nrJ = [z; e; -e]
    nsJ = [-e; e; z]
    nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
    nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
    sJ = @. sqrt(nxJ^2 + nyJ^2)

    "=========== Done defining geometry and mesh ============="

    "Time integration coefficients"
    CN = (N+1)*(N+2)/2  # estimated trace constant
    dt = CFL * 2 / (CN*K1D)
    Nsteps = convert(Int,ceil(T/dt))
    dt = T/Nsteps

    "Define the initial conditions by interpolation"
    k = 1
    pex = (x,y,t)-> @. sin(pi*x)*sin(pi*y)*cos(sqrt(2)*pi*t)
    p = pex(x,y,dt)
    pprev = pex(x,y,0)

    "pack arguments into tuples"
    ops = (Dr,Ds,LIFT,Vf)
    vgeo = (rxJ,sxJ,ryJ,syJ,J)
    fgeo = (nxJ,nyJ,sJ)
    mapP = reshape(mapP,Nfp*Nfaces,K)
    nodemaps = (mapP,mapB)

    "Define function to evaluate the RHS: Q = (p,u,v)"
    function rhs_2ndorder(p,ops,vgeo,fgeo,nodemaps)
        # unpack arguments
        Dr,Ds,LIFT,Vf = ops
        rxJ,sxJ,ryJ,syJ,J = vgeo
        nxJ,nyJ,sJ = fgeo
        (mapP,mapB) = nodemaps

        # construct sigma
        pf = Vf*p
        dp = pf[mapP]-pf
        pr = Dr*p
        ps = Ds*p
        dpdx = @. rxJ*pr + sxJ*ps
        dpdy = @. ryJ*pr + syJ*ps
        σxflux = @. dp*nxJ
        σyflux = @. dp*nyJ
        σx = (dpdx + .5*LIFT*σxflux)./J
        σy = (dpdy + .5*LIFT*σyflux)./J

        # compute div(σ)
        σxf,σyf = (x->Vf*x).((σx,σy))
        σxP,σyP = (x->x[mapP]).((σxf,σyf))
        pflux = @. .5*((σxP-σxf)*nxJ + (σyP-σyf)*nyJ)
        σxr,σyr = (x->Dr*x).((σx,σy))
        σxs,σys = (x->Ds*x).((σx,σy))
        dσxdx = @. rxJ*σxr + sxJ*σxs
        dσydy = @. ryJ*σyr + syJ*σys

        tau = .5
        rhsp = dσxdx + dσydy + LIFT*(pflux + tau*dp)

        return rhsp./J
    end

    # Perform time-stepping
    for i = 2:Nsteps

        rhsQ = rhs_2ndorder(p,ops,vgeo,fgeo,nodemaps)
        pnew = 2*p - pprev + dt^2 * rhsQ
        @. pprev = p
        @. p = pnew

        if i%10==0 || i==Nsteps
            println("Number of time steps $i out of $Nsteps")
            # vv = Vp*p
            # scatter(Vp*x,Vp*y,vv,zcolor=vv)
        end
    end

    rq,sq,wq = quad_nodes_2D(2*N+2)
    Vq = vandermonde_2D(N,rq,sq)/V
    xq,yq = (x->Vq*x).((x,y))
    wJq = diagm(wq)*(Vq*J)

    perr = Vq*p - pex(xq,yq,T)
    err[kk] = sqrt(sum(wJq.*(perr.^2)))
end

h = @. 2/Kvec[:]

gr(size=(300,300),legend=false,
xscale=:log10,yscale=:log10,
markerstrokewidth=2,markersize=4)

plot!(h,err,markershape=:circle)
scale = err[1]/h[1]^(N+1)
display(plot!(h,h.^(N+1)*scale,linestyle=:dash))

C = [ones(length(h[end-1:end])) log.(h[end-1:end])] \ log.(err[end-1:end])
rate = C[2]
println("Computed rate of convergence = $rate")
