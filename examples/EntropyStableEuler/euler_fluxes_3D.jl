#####
##### three-dimensional fluxes
#####

"function euler_fluxes_3D(rhoL,uL,vL,wL,betaL,rhologL,betalogL,
                         rhoR,uR,vR,wR,betaR,rhologR,betalogR)"
function euler_fluxes_3D(rhoL,uL,vL,wL,betaL,rhologL,betalogL,
                         rhoR,uR,vR,wR,betaR,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))
    vavg   = (@. .5*(vL+vR))
    wavg   = (@. .5*(wL+wR))

    unorm = (@. uL*uR + vL*vR + wL*wR)
    pa    = (@. rhoavg/(betaL+betaR))
    E_plus_p  = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

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

    Fx = (FxS1,FxS2,FxS3,FxS4,FxS5)
    Fy = (FyS1,FyS2,FyS3,FyS4,FyS5)
    Fz = (FzS1,FzS2,FzS3,FzS4,FzS5)
    return Fx,Fy,Fz
end


#####
##### shim functions
#####

# dispatch to n-dimensional constitutive routines, with optional entropy scaling
function primitive_to_conservative(rho,u,v,w,p)
   rho,rhoU,E = primitive_to_conservative_nd(rho,(u,v,w),p)
   return rho,rhoU...,E
end
function v_ufun(rho,rhou,rhov,rhow,E)
    v1,vU,vE = v_ufun_nd(rho,(rhou,rhov,rhow),E)
    return scale_entropy_output(v1,vU...,vE)
end
function u_vfun(v1,vU1,vU2,vU3,vE)
    v1,vU1,vU2,vU3,vE = scale_entropy_input(v1,vU1,vU2,vU3,vE)
    rho,rhoU,E = u_vfun_nd(v1,(vU1,vU2,vU3),vE)
    return rho,rhoU...,E
end
function conservative_to_primitive_beta(rho,rhou,rhov,rhow,E)
    rho,U,beta = conservative_to_primitive_beta_nd(rho,(rhou,rhov,rhow),E)
    return rho,U...,beta
end

Sfun(rho,rhou,rhov,rhow,E) = Sfun_nd(rho,(rhou,rhov,rhow),E)
pfun(rho,rhou,rhov,rhow,E) = pfun_nd(rho,(rhou,rhov,rhow),E)
betafun(rho,rhou,rhov,rhow,E) = betafun_nd(rho,(rhou,rhov,rhow),E)

"function euler_fluxes_3D(rhoL,uL,vL,wL,betaL,rhoR,uR,vR,wR,betaR)"
function euler_fluxes(rhoL,uL,vL,wL,betaL,rhoR,uR,vR,wR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_3D(rhoL,uL,vL,wL,betaL,rhologL,betalogL,
                           rhoR,uR,vR,wR,betaR,rhologR,betalogR)
end
euler_fluxes(rhoL,uL,vL,wL,betaL,rhologL,betalogL,
             rhoR,uR,vR,wR,betaR,rhologR,betalogR) =
     euler_fluxes_3D(rhoL,uL,vL,wL,betaL,rhologL,betalogL,
                     rhoR,uR,vR,wR,betaR,rhologR,betalogR)
