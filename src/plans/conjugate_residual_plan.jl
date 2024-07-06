struct SymmetricLinearSystemObjective{E<:AbstractEvaluationType,TA,Tb} <:
       AbstractManifoldObjective{E}
    A::TA
    b::Tb
end

function SymmetricLinearSystemObjective(A::TA, b::Tb) where {TA,Tb}
    return SymmetricLinearSystemObjective{AllocatingEvaluation,TA,Tb}(A, b)
end

function set_manopt_parameter!(slso::SymmetricLinearSystemObjective, symbol::Symbol, value)
    set_manopt_parameter!(slso.A, symbol, value)
    set_manopt_parameter!(slso.b, symbol, value)
    return slso
end

# TODO: Be more careful and precise with in-place and allocating evaluations here
#
# evaluate the quadratic cost: Q(X) = 1/2 ⟨X, A(X)⟩ - ⟨b, X⟩ associated to the system Ax = b
function get_cost(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    return 0.5 * inner(M, p, X, slso.A(M, p, X)) - inner(M, p, slso.b(M, p), X)
end

# evauate the gradient: ∇Q(X) = A(X) - b
function get_gradient(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    return slso.A(M, p, X) - slso.b(M, p)
end

# in-place gradient evaluation
function get_gradient!(TpM::TangentSpace, Y, slso::SymmetricLinearSystemObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    # This was inconcsistent: slso.A(M, Y, p, X) - slso.b(M, p)
    # Does A always work in-place?
    slso.A(M, Y, p, X)
    Y .-= slso.b(M, p)
    return Y
end

# evaluate Hessian: ∇²Q(X) = A(X)
function get_hessian(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, X, V)
    M = base_manifold(TpM)
    p = TpM.point
    # or both?
    return slso.A(M, p, V)
end

# in-place Hessian evaluation
function get_hessian!(TpM::TangentSpace, W, slso::SymmetricLinearSystemObjective, X, V)
    M = base_manifold(TpM)
    p = TpM.point
    return slso.A(M, W, p, V)
end
