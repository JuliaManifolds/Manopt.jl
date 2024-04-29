mutable struct ConjugateResidualCost{TA,Tb}
    A::TA
    b::Tb  
end
function set_manopt_parameter!(crc::ConjugateResidualCost, ::Val{:A}, A)
    crc.A = A
    return crc
end                        
function set_manopt_parameter!(crc::ConjugateResidualCost, ::Val{:b}, b)
    crc.b = b
    return crc
end
function (Q::ConjugateResidualCost)(TpM::TangentSpace, X)
    M = base_manifold(TpM)
    p = TpM.point
    return 0.5 * inner(M, p, X, (Q.A)(M, p, X)) - inner(M, p, Q.b, X)
end

mutable struct ConjugateResidualGrad{TA, Tb}
    A::TA 
    b::Tb   
end
function set_manopt_parameter!(crg::ConjugateResidualGrad, ::Val{:A}, A)
    crg.A = A
    return crg
end
function set_manopt_parameter!(crg::ConjugateResidualGrad, ::Val{:b}, b)
    crg.b = b
    return crg
end
function (GQ::ConjugateResidualGrad)(TpM::TangentSpace, X)
    M = base_manifold(TpM)
    p = TpM.point
    return (GQ.A)(M, p, X) - GQ.b
end

mutable struct ConjugateResidualHess{TA}
    A::TA
end
function set_manopt_parameter!(crh::ConjugateResidualHess, ::Val{:A}, A)
    crh.A = A
    return crh
end
function (HQ::ConjugateResidualHess)(TpM::TangentSpace, X, Y)
    M = base_manifold(TpM)
    p = TpM.point
    return (HQ.A)(M, p, Y)
end



