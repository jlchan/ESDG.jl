
push!(LOAD_PATH, "./src")
using DelimitedFiles
using Plots
using SetupDG
using UnPack
using UniformTriMesh
using PlotThemes
using ToeplitzMatrices
using Basis2DTri

macro Name(arg)
   string(arg)
end

gr(aspect_ratio=1,legend=false,colorbar=:right,framestyle=:none,xlims=(-1,1),ylims=(-1,1),
   markerstrokewidth=0,markersize=1,c=:haline)

N = 4
Nq_P = countlines("src/QuadratureData/quad_nodes_tri_N$(2*N).txt")
Nd = 5
K1D = 128
K = K1D*K1D*2
Np_F = 8
Nplot = 15
Mach = ".7"
delta = ".05"
extra_attribute = "varwidthsquared"

#T = 2.5
#T_arr = [0.5;1.0;1.5;2.0;2.5]
T_arr = [2.0]
for T in T_arr
    folder_name = "K$(K1D)NPF$(Np_F)N$(N)delta$(delta)Mach$(Mach)$(extra_attribute)"
    file_name = "array_end"
    if T == 0.5
        file_name = "array_1"
    elseif T == 1.0
        file_name = "array_2"
    elseif T == 1.5
        file_name = "array_3"
    elseif T == 2.0
        file_name = "array_4"
    elseif T == 2.5
        file_name = "array_end"
    end
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

    for f = [1]
        xq_f = zeros(Nq_P,K)
        yq_f = zeros(Nq_P,K)
        for k = 1:K
            @. xq_f[:,k] = xq[:,f,k]
            @. yq_f[:,k] = yq[:,f,k]
        end

        #rho_f = zeros(Nq_P,K)
        u_f = zeros(Nq_P,K)
        v_f = zeros(Nq_P,K)
        #w_f = zeros(Nq_P,K)
        vort_z_f = zeros(Nq_P,K)
        for k = 1:K
            # @. rho_f[:,k] = Q[:,f,1,k]
            @. u_f[:,k] = Q[:,f,2,k]./Q[:,f,1,k]
            @. v_f[:,k] = Q[:,f,3,k]./Q[:,f,1,k]
            #@. w_f[:,k] = Q[:,f,4,k]./Q[:,f,1,k]
        end


        v_f_modal = Pq*v_f
        u_f_modal = Pq*u_f
        #w_f_modal = Pq*w_f

        vort_z_f = Vq*(rxJ.*(Dr*v_f_modal)+sxJ.*(Ds*v_f_modal)) - Vq*(ryJ.*(Dr*u_f_modal)+syJ.*(Ds*u_f_modal))

        xq_f = VpPq*xq_f
        yq_f = VpPq*yq_f
        # rho_f = VpPq*rho_f
        # u_f = VpPq*u_f
        # v_f = VpPq*v_f
        # w_f = VpPq*w_f
        vort_z_f = VpPq*vort_z_f

        # rho_f = rho_f[:]
        # u_f = u_f[:]
        # v_f = v_f[:]
        # w_f = w_f[:]
        xq_f = xq_f[:]
        yq_f = yq_f[:]
        vort_z_f = vort_z_f[:]
        # absu_f = @. u_f^2+v_f^2+w_f^2

        # scatter(xq_f,yq_f,rho_f,zcolor=rho_f,camera=(0,90),axis=nothing)
        # png("./examples/ES_Fourier-DG/$folder_name/rho_f$(f)_T=$T.png")
        # scatter(xq_f,yq_f,u_f,zcolor=u_f,camera=(0,90),axis=nothing)
        # png("./examples/ES_Fourier-DG/$folder_name/u_f$(f)_T=$T.png")
        # scatter(xq_f,yq_f,v_f,zcolor=v_f,camera=(0,90),axis=nothing)
        # png("./examples/ES_Fourier-DG/$folder_name/v_f$(f)_T=$T.png")
        # scatter(xq_f,yq_f,w_f,zcolor=w_f,camera=(0,90),axis=nothing)
        # png("./examples/ES_Fourier-DG/$folder_name/w_f$(f)_T=$T.png")
        # scatter(xq_f,yq_f,absu_f,zcolor=absu_f,camera=(0,90),axis=nothing)
        # png("./examples/ES_Fourier-DG/$folder_name/speed_f$(f)_T=$T.png")
        scatter(xq_f,yq_f,vort_z_f,zcolor=vort_z_f,camera=(0,90),clims=(-0.001,0.001),axis=nothing)
        png("./examples/ES_Fourier-DG/$folder_name/vort_f$(f)_T=$T.png")
    end
end
