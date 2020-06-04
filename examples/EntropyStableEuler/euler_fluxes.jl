"""
function euler_fluxes(UL,UR)
    UL, UR = states = (rho,u,v,beta)
    2D version.  Figure out multiple dispatch version for 3D?

"""

# assumes field ordering: (rhoL,uL,...,betaL) = UL, (rhoR,uR,...,betaR) = UR
function euler_fluxes(UL,UR)
    rhoL = first(UL); betaL = last(UL)
    rhoR = first(UR); betaR = last(UR)
    logL = (log.(rhoL),log.(betaL))
    logR = (log.(rhoR),log.(betaR))
    return euler_fluxes(UL,UR,logL,logR)
end

# dispatches to 2D
function euler_fluxes(UL,UR,logL,logR)
    return euler_fluxes(UL...,UR...,logL...,logR...)
end

# 2d version
function euler_fluxes(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR,
                      rhologL,betalogL,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))
    vavg   = (@. .5*(vL+vR))

    unorm = (@. uL*uR + vL*vR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. FxS1*vavg)
    FxS4 = (@. f4aux*uavg)

    FyS1 = (@. rholog*vavg)
    FyS2 = (@. FxS3)
    FyS3 = (@. FyS1*vavg + pa)
    FyS4 = (@. f4aux*vavg)
    return (FxS1,FxS2,FxS3,FxS4),(FyS1,FyS2,FyS3,FyS4)
end

function euler_flux_x(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR,
                      rhologL,betalogL,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))
    vavg   = (@. .5*(vL+vR))

    unorm = (@. uL*uR + vL*vR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. FxS1*vavg)
    FxS4 = (@. f4aux*uavg)

    return (FxS1,FxS2,FxS3,FxS4)
end

function euler_flux_y(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR,
                      rhologL,betalogL,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))
    vavg   = (@. .5*(vL+vR))

    unorm = (@. uL*uR + vL*vR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FyS1 = (@. rholog*vavg)
    FyS2 = (@. FyS1*uavg)
    FyS3 = (@. FyS1*vavg + pa)
    FyS4 = (@. f4aux*vavg)
    return (FyS1,FyS2,FyS3,FyS4)
end


# # 3d version
# function euler_fluxes(rhoL,uL,vL,wL,betaL,rhoR,uR,vR,wR,betaR,
#                       rhologL,betalogL,rhologR,betalogR)
#
#     rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
#     betalog = logmean.(betaL,betaR,betalogL,betalogR)
#
#     # arithmetic avgs
#     rhoavg = (@. .5*(rhoL+rhoR))
#     uavg   = (@. .5*(uL+uR))
#     vavg   = (@. .5*(vL+vR))
#     wavg   = (@. .5*(wL+wR))
#
#     unorm = (@. uL*uR + vL*vR + wL*wR)
#     pa    = (@. rhoavg/(betaL+betaR))
#     E_plus_p  = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)
#
#     FxS1 = (@. rholog*uavg)
#     FxS2 = (@. FxS1*uavg + pa)
#     FxS3 = (@. FxS1*vavg) # rho * u * v
#     FxS4 = (@. FxS1*wavg) # rho * u * w
#     FxS5 = (@. E_plus_p*uavg)
#
#     FyS1 = (@. rholog*vavg)
#     FyS2 = (@. FxS3) # rho * u * v
#     FyS3 = (@. FyS1*vavg + pa)
#     FyS4 = (@. FyS1*wavg) # rho * v * w
#     FyS5 = (@. E_plus_p*vavg)
#
#     FzS1 = (@. rholog*wavg)
#     FzS2 = (@. FxS4) # rho * u * w
#     FzS3 = (@. FyS4) # rho * v * w
#     FzS4 = (@. FzS1*wavg + pa) # rho * w^2 + p
#     FzS5 = (@. E_plus_p*wavg)
#
#     Fx = (FxS1,FxS2,FxS3,FxS4,FxS5)
#     Fy = (FyS1,FyS2,FyS3,FyS4,FyS5)
#     Fz = (FzS1,FzS2,FzS3,FzS4,FzS5)
#     return Fx,Fy,Fz
# end
