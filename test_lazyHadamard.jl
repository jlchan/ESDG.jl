using Revise # reduce need for recompile
using LinearAlgebra
using SparseArrays

N = 10
Id = diagm(ones(N))
A = kron(Id,ones(N,N))
A = droptol!(sparse(A),1e-12)

u = rand(N*N,1000)

function fun(uL,uR)
    return uL^2 + uL*uR + uR^2
end

function meshgrid(x,y)
    X = repeat(transpose(x[:]), size(y[:], 1), 1)
    Y = repeat(y, 1, size(x, 1))
    return X, Y
end

function full_hadamard_sum(A,u,fun)
    ux,uy = meshgrid(u,u)
    return sum(A.*fun.(ux,uy),dims=2)
end

function lazy_hadamard_sum(A,u,fun)
    NN = size(A,1)
    AF = zeros(NN)
    for i = 1:NN
        Ai = A[i,:]
        ui = u[i]
        AFi = 0.0
        for j = Ai.nzind
            uj = u[j]
            AFi += Ai[j]*fun(ui,uj)
        end
        AF[i] = AFi
    end
    return AF
end

timefull = 0.0
timelazy = 0.0
for k = 1:100
    global timefull,timelazy
    a = @timed full_hadamard_sum(A,u[:,k],fun)
    b = @timed lazy_hadamard_sum(A,u[:,k],fun)
    if norm(a[1]-b[1]) > 1e-10
        error("Results differ!")
    end
    timefull += a[2]
    timelazy += b[2]
end

print("timefull = ",timefull/10,", timelazy = ",timelazy/10,"\n")
