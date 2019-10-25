"""
function euler_fluxes(UL,UR)
    UL, UR = states = (rho,u,v,beta)
    2D version.  Figure out multiple dispatch version for 3D?

"""
function wavespeed(U,γ=1.4)
    (rho,rhou,rhov,E) = U
    unorm2 = @. (rhou^2 + rhov^2)/rho
    p = @. (γ-1.0)*(E - .5*rho*unorm2)
    cvel = @. sqrt(γ*p/rho)
    lam = @. sqrt(unorm2)+cvel
end

function euler_fluxes(UL,UR)
    (rhoL,uL,vL,betaL) = UL
    (rhoR,uR,vR,betaR) = UR
    logL = (log.(rhoL),log.(betaL))
    logR = (log.(rhoR),log.(betaR))
    return euler_fluxes(UL,UR,logL,logR)
end

"vectorized version"
function euler_fluxes(UL,UR,logL,logR)
    γ = 1.4
    (rhoL,uL,vL,betaL) = UL
    (rhoR,uR,vR,betaR) = UR
    (rhologL,betalogL) = logL
    (rhologR,betalogR) = logR

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = @. .5*(rhoL+rhoR)
    uavg   = @. .5*(uL+uR)
    vavg   = @. .5*(vL+vR)

    unorm = @. uL*uR + vL*vR
    pa    = @. rhoavg/(betaL+betaR)
    f4aux = @. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm

    FxS1 = @. rholog*uavg
    FxS2 = @. FxS1*uavg + pa
    FxS3 = @. FxS1*vavg
    FxS4 = @. f4aux*uavg

    FyS1 = @. rholog*vavg
    FyS2 = @. FxS3
    FyS3 = @. FyS1*vavg + pa
    FyS4 = @. f4aux*vavg
    return ((FxS1,FxS2,FxS3,FxS4),(FyS1,FyS2,FyS3,FyS4))
end