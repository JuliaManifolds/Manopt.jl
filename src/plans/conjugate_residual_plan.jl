# Maybe rename?
# (RB:) We could maybe even “rework” this to jiust have an HessianObjective inside?
# This would require the ReducedLagrangianCost to exist, but make quite a few things, especially
# passing on parameters easier:
#
# Updating the CRCost's λ (without all the deep pass down explxlity mentioned)
# ( that is `set_manopt_parameter(crc::ConjugateResidulaCost, ::Val{:λ}, λ)` is what you define)
# would mean this struct passes the lamda to its innter cost, grad and Hessian.
# That way this can even be agnostic how the inner grad and Hess are organized.
# and it would just store am mho (manifold hessian objective)
#
# I think that would be a good way to encapsulate this
mutable struct ConjugateResidualCost{TA,Tb}
    A::TA
    b::Tb
end
function set_manopt_parameter!(crc::ConjugateResidualCost, ::Val{:A}, args...)
    set_manopt_parameter!(crc.A, args...)
    return crc
end
function set_manopt_parameter!(crc::ConjugateResidualCost, ::Val{:b}, args...)
    set_manopt_parameter!(crc.b, args...)
    return crc
end
function (Q::ConjugateResidualCost)(TpM::TangentSpace, X)
    M = base_manifold(TpM)
    p = TpM.point
    return 0.5 * inner(M, p, X, (Q.A)(M, p, X)) - inner(M, p, Q.b(M, p), X)
end

mutable struct ConjugateResidualGrad{TA,Tb}
    A::TA
    b::Tb
end
function set_manopt_parameter!(crg::ConjugateResidualGrad, ::Val{:A}, args...)
    set_manopt_parameter!(crg.A, args...)
    return crg
end
function set_manopt_parameter!(crg::ConjugateResidualGrad, ::Val{:b}, args...)
    set_manopt_parameter!(crg.b, args...)
    return crg
end
function (GQ::ConjugateResidualGrad)(TpM::TangentSpace, X)
    M = base_manifold(TpM)
    p = TpM.point
    return (GQ.A)(M, p, X) - GQ.b(M,p)
end

mutable struct ConjugateResidualHess{TA}
    A::TA
end
function set_manopt_parameter!(crh::ConjugateResidualHess, ::Val{:A}, args...)
    set_manopt_parameter!(crh.A, args...)
    return crh
end
function (HQ::ConjugateResidualHess)(TpM::TangentSpace, X, Y)
    M = base_manifold(TpM)
    p = TpM.point
    return (HQ.A)(M, p, Y)
end
