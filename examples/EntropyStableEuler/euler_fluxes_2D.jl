#####
##### two-dimensional fluxes
#####
"
function euler_fluxes_2D(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR,
                         rhologL,betalogL,rhologR,betalogR)
assumes primitive variables ordering: UL = (rhoL,uL,...,betaL),
                                      UR = (rhoR,uR,...,betaR)
"
function euler_fluxes_2D(rhoL,uL,vL,betaL,rhologL,betalogL,
                         rhoR,uR,vR,betaR,rhologR,betalogR)

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

function euler_fluxes_2D_x(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_2D_x(rhoL,uL,vL,betaL,rhologL,betalogL,
                             rhoR,uR,vR,betaR,rhologR,betalogR)
end
function euler_fluxes_2D_y(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_2D_y(rhoL,uL,vL,betaL,rhologL,betalogL,
                             rhoR,uR,vR,betaR,rhologR,betalogR)
end

function euler_fluxes_2D_x(rhoL,uL,vL,betaL,rhologL,betalogL,
                           rhoR,uR,vR,betaR,rhologR,betalogR)

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

function euler_fluxes_2D_y(rhoL,uL,vL,betaL,rhologL,betalogL,
                           rhoR,uR,vR,betaR,rhologR,betalogR)

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


#####
##### shim functions
#####

# dispatch to n-dimensional constitutive routines, with optional entropy scaling
function primitive_to_conservative(rho,u,v,p)
   rho,rhoU,E = primitive_to_conservative_nd(rho,(u,v),p)
   return rho,rhoU...,E
end
function v_ufun(rho,rhou,rhov,E)
    v1,vU,vE = v_ufun_nd(rho,(rhou,rhov),E)
    return scale_entropy_output(v1,vU...,vE)
end
function u_vfun(v1,vU1,vU2,vE)
    v1,vU1,vU2,vE = scale_entropy_input(v1,vU1,vU2,vE)
    rho,rhoU,E = u_vfun_nd(v1,(vU1,vU2),vE)
    return rho,rhoU...,E
end
function conservative_to_primitive_beta(rho,rhou,rhov,E)
    rho,U,beta = conservative_to_primitive_beta_nd(rho,(rhou,rhov),E)
    return rho,U...,beta
end

Sfun(rho,rhou,rhov,E) = Sfun_nd(rho,(rhou,rhov),E)
pfun(rho,rhou,rhov,E) = pfun_nd(rho,(rhou,rhov),E)
betafun(rho,rhou,rhov,E) = betafun_nd(rho,(rhou,rhov),E)

"function euler_fluxes_2D(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR)
assumes primitive variables ordering: UL = (rhoL,uL,...,betaL),
                                      UR = (rhoR,uR,...,betaR)"
function euler_fluxes(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_2D(rhoL,uL,vL,betaL,rhologL,betalogL,
                           rhoR,uR,vR,betaR,rhologR,betalogR)
end

euler_fluxes(rhoL,uL,vL,betaL,rhologL,betalogL,
             rhoR,uR,vR,betaR,rhologR,betalogR) =
     euler_fluxes_2D(rhoL,uL,vL,betaL,rhologL,betalogL,
                     rhoR,uR,vR,betaR,rhologR,betalogR)
