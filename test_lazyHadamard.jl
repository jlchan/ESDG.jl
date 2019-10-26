using Revise # reduce need for recompile
using LinearAlgebra
using BenchmarkTools

N = 25
K = 2048

A = randn(N,N)
u = rand(size(A,2),K)
U = (rand(size(A,2),K), rand(size(A,2),K))

function flux_fun(u,v)
    return u*v;
end

function dense_hadamard(ATr,u,flux_fun)
    (N,K) = size(u)
    AFe = zeros(N)
    AF = zeros(N,K)
    for e = 1:K
        for i = 1:N
            uei = u[i,e]
            AFi = 0.0
            for j = 1:N
                AFi = AFi + ATr[j,i]*flux_fun(uei,u[j,e])
            end
            AFe[i] = AFi
        end
        AF[:,e] = AFe
    end
    return AF
end


function flux_fun_vec(U,V)
    return (U[1]*V[2], U[2]*V[1]);
end

function dense_hadamard_vec(ATr,U,flux_fun_vec)
    Nfields = length(U)
    N = size(U[1],1)
    K = size(U[1],2)
    AF = [zeros(N,K) for fld in eachindex(U)]
    AFe = [zeros(N) for fld in eachindex(U)]
    Ui = zeros(length(U))
    AFi = zeros(length(U))
    Uj = zeros(length(U))

    for e = 1:K
        for i = 1:N
            for fld=1:Nfields
                Ui[fld] = U[fld][i]
            end
            for j = 1:N
                for fld=1:Nfields
                    Uj[fld] = U[fld][j]
                end
                Fij = flux_fun_vec(Ui,Uj)
                for fld=1:Nfields
                    AFi[fld] = AFi[fld] + ATr[j,i]*Fij[fld]
                end
            end
            for fld=1:Nfields
                AFe[fld][i] = AFi[fld]
            end
        end
        for fld = 1:Nfields
            AF[fld][:,e] = AFe[fld]
        end
    end
    return AF
end

@btime dense_hadamard(transpose(A),u,flux_fun)
@btime dense_hadamard_vec(transpose(A),U,flux_fun_vec)
