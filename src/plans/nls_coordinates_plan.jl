@doc """
    LevenbergMarquardtLinearCoordinatesSurrogateObjective{E<:AbstractEvaluationType, VF<:AbstractManifoldFirstOrderObjective{E}, R} <: AbstractManifoldFirstOrderObjective{E, VF}


A subobjective similar to `LevenbergMarquardtLinearSurrogateObjective` but which uses
coordinate-based Jacobians in a single, selected basis instead of being centered around
linear operators.
## Fields

* `objective`:     the [`NonlinearLeastSquaresObjective`](@ref) to penalize
* `penalty::Real`: the damping term ``λ``
* `ε::Real`:       stabilization for ``α ≤ 1-ε`` in the rescaling of the Jacobian, that
* `mode::Symbol`:  which ode to use to stabilize α, see the internal helper [`get_LevenbergMarquardt_scaling`](@ref)
* `value_cache`:   a vector to store the residuals ``F(p)`` at the current point `p` internally to avoid recomputations
* `jacobian_cache`: a vector to store the coordinate-based Jacobian of the residuals at the
  current point `p` internally to avoid recomputations. If the Jacobian is used as a linear
  operator, this is just a vector of `nothing`s.

## Constructor

    LevenbergMarquardtLinearCoordinatesSurrogateObjective(objective; penalty::Real = 1e-6, ε::Real = 1e-4, mode::Symbol = :Default )

"""
mutable struct LevenbergMarquardtLinearCoordinatesSurrogateObjective{
        E <: AbstractEvaluationType, R <: Real, TO <: NonlinearLeastSquaresObjective{E}, TVC <: AbstractVector{R}, TJC <: AbstractVector, TB <: AbstractBasis,
    } <: AbstractLinearSurrogateObjective{E, NonlinearLeastSquaresObjective{E}}
    objective::TO
    penalty::R
    ε::R
    mode::Symbol
    value_cache::TVC
    jacobian_cache::TJC
    basis::TB
    function LevenbergMarquardtLinearCoordinatesSurrogateObjective(
            objective::NonlinearLeastSquaresObjective{E};
            penalty::R = 1.0e-6, ε::R = 1.0e-4, mode::Symbol = :Default,
            residuals::TVC = zeros(sum(length(o) for o in get_objective(objective).objective)),
            jacobian_cache::TJC = fill(nothing, length(get_objective(objective).objective)),
            basis::TB = DefaultOrthonormalBasis(),
        ) where {E, R <: Real, TVC <: AbstractVector, TJC <: AbstractVector, TB <: AbstractBasis}
        return new{E, R, typeof(objective), TVC, TJC, TB}(objective, penalty, ε, mode, residuals, jacobian_cache, basis)
    end
end

get_objective(lmsco::LevenbergMarquardtLinearCoordinatesSurrogateObjective) = lmsco.objective

function set_parameter!(lmlso::LevenbergMarquardtLinearCoordinatesSurrogateObjective, ::Val{:Penalty}, penalty::Real)
    lmlso.penalty = penalty
    return lmlso
end

function linear_normal_operator!(
        M::AbstractManifold, A::AbstractMatrix, lmsco::LevenbergMarquardtLinearCoordinatesSurrogateObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco.objective)
    # For every block
    fill!(A, 0)
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len_o = length(o)
        add_linear_normal_operator_coord!(
            M, A, o, r, p, B; value_cache = view(lmsco.value_cache, (start + 1):(start + len_o)), jacobian_cache = jc,
            ε = lmsco.ε, mode = lmsco.mode
        )
        start += len_o
    end
    # Finally add the damping term
    (penalty != 0) && (LinearAlgebra.diagview(A) .+= penalty)
    return A
end

"""
    add_linear_normal_operator_coord!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractVectorGradientFunction,
        r::AbstractRobustifierFunction, p, basis::AbstractBasis;
        value_cache, jacobian_cache, ε::Real, mode::Symbol
    )

Accumulate the contribution of a single block (vectorial function with its robustifier) to
the linear normal operator, i.e. compute ``A += J_F^*(p)[C^T C J_F(p)[X]]`` in-place of `A`
for the given block.
"""
function add_linear_normal_operator_coord!(
        M::AbstractManifold, A::AbstractMatrix, o::AbstractVectorGradientFunction,
        r::AbstractRobustifierFunction, p, basis::AbstractBasis;
        value_cache, jacobian_cache, ε::Real, mode::Symbol
    )
    a = value_cache # evaluate residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # to Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
    # (a) J_F is n-by-d so we have to allocate – where could we maybe store something like that and pass it down?
    # (I - s*a*a')^2 = I + (-2s + s^2*||a||^2) * a*a'
    # so JF' * (ρ' * (I - s*a*a')^2) * JF
    #   = ρ' * (JF'JF) + ρ' * (-2s + s^2*||a||^2) * (JF'a) * (JF'a)'
    rank1_scaling = ρ_prime * (-2 * operator_scaling + operator_scaling^2 * F_sq)
    mul!(A, jacobian_cache', jacobian_cache, ρ_prime, true)
    if !iszero(rank1_scaling)
        JFa = jacobian_cache' * a
        mul!(A, JFa, JFa', rank1_scaling, true)
    end
    # damping term is added once after summing up all blocks, so we do not add it here
    return A
end

function linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearCoordinatesSurrogateObjective}, p, B::AbstractBasis;
        penalty::Real = slso.objective.penalty,
    ) where {E <: AbstractEvaluationType}
    return linear_normal_operator!(M, A, slso.objective, p, B; penalty = penalty)
end

function normal_vector_field_coord!(
        M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearCoordinatesSurrogateObjective, p,
    )
    nlso = lmsco.objective
    # For every block
    fill!(c, 0)
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len_o = length(o)
        add_normal_vector_field_coord!(
            M, c, o, r, p;
            value_cache = view(lmsco.value_cache, (start + 1):(start + len_o)), jacobian_cache = jc, ε = lmsco.ε, mode = lmsco.mode
        )
        start += len_o
    end
    return c
end

function add_normal_vector_field_coord!(
        M::AbstractManifold, c, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
        value_cache, jacobian_cache, ε::Real, mode::Symbol,
    )
    y = copy(value_cache) # evaluate residuals F(p)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # Compute y = ρ'(p) / (1-α)) F(p) and ...
    y .= residual_scaling .* sqrt(ρ_prime) * (I - operator_scaling * (y * y')) * y
    # ...apply the adjoint, i.e. compute  J_F^*(p)[C^T y] (adding it to c)
    JFt = jacobian_cache'
    mul!(c, jacobian_cache, y, true, true)
    return c
end

function normal_vector_field_coord!(
        M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearCoordinatesSurrogateObjective, p, B::AbstractBasis,
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(c, 0)
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len_o = length(o)
        add_normal_vector_field_coord!(
            M, c, o, r, p, B;
            value_cache = view(lmsco.value_cache, (start + 1):(start + len_o)),
            jacobian_cache = jc, ε = lmsco.ε, mode = lmsco.mode
        )
        start += len_o
    end
    return c
end

# for a single block – the actual formula
@doc "$(_doc_add_normal_vector_field)"
function add_normal_vector_field_coord!(
        M::AbstractManifold, c::AbstractVector, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, B::AbstractBasis;
        value_cache, jacobian_cache, ε::Real, mode::Symbol,
    )
    y = copy(value_cache) # evaluate residuals F(p)
    F_sq = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # Compute y = ρ'(p) / (1-α)) F(p) and ...
    y .= residual_scaling .* sqrt(ρ_prime) * (I - operator_scaling * (y * y')) * y
    # ...apply the adjoint, i.e. compute  J_F^*(p)[C^T y] (inplace of y)
    mul!(c, jacobian_cache', y, true, true)
    return c
end

function vector_field!(
        M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearCoordinatesSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    # Init to zero
    fill!(y, 0)
    start = 0
    # For every block
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        _vector_field!(M, view(y, (start + 1):(start + length(o))), o, r, p; ε = lmsco.ε, mode = lmsco.mode)
        start += length(o)
    end
    return y
end

function vector_field!(
        M::AbstractManifold, c, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearCoordinatesSurrogateObjective}, p, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    normal_vector_field_coord!(M, c, slso.objective, p, B)
    c .*= -1
    return c
end


function get_solver_result(
        dmp::DefaultManoptProblem{<:TangentSpace, <:SymmetricLinearSystem{<:AbstractEvaluationType, <:LevenbergMarquardtLinearCoordinatesSurrogateObjective}},
        cnss::CoordinatesNormalSystemState
    )
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_vector(M, p, cnss.c, cnss.basis)
end

function get_cost(
        TpM::TangentSpace, lnsco::SymmetricLinearSystem{<:AbstractEvaluationType, <:LevenbergMarquardtLinearCoordinatesSurrogateObjective},
        ::ZeroTangentVector
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    # TODO: optimize?
    n = sum(length(o) for o in lnsco.objective.objective.objective)
    vf = zeros(number_eltype(p), n)
    vector_field!(M, vf, lnsco.objective, p)
    if lnsco.objective.basis === DefaultOrthonormalBasis()
        # really should work with any orthonormal basis
        return 0.5 * norm(vf)^2
    else
        return 0.5 * norm(M, p, get_vector(M, p, vf))^2
    end
end

function get_cost(
        TpM::TangentSpace, lnsco::SymmetricLinearSystem{<:AbstractEvaluationType, <:LevenbergMarquardtLinearCoordinatesSurrogateObjective},
        X,
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    # TODO: optimize?
    if lnsco.objective.basis === DefaultOrthonormalBasis()
        cX = get_coordinates(M, p, X)
        n = sum(length(o) for o in lnsco.objective.objective.objective)
        vf = zeros(number_eltype(p), n)
        vector_field!(M, vf, lnsco.objective, p)
        add_linear_operator_coord!(TpM, vf, lnsco.objective, p, cX)
        cost = 0.5 * norm(vf)^2
        cost += (lnsco.objective.penalty / 2) * norm(M, p, X)^2
        return cost
    else
        return 0.5 * norm(M, p, linear_operator(TpM, lnsco.objective, p, X) + vector_field(TpM, lnsco.objective, p))^2
    end
end

function add_linear_normal_operator_coord!(
        M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearCoordinatesSurrogateObjective, p, cX;
        penalty::Real = lmsco.penalty,
    )
    nlso = get_objective(lmsco)
    # For every block
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        add_linear_normal_operator_coord!(
            M, c, o, r, p, cX;
            ε = lmsco.ε, mode = lmsco.mode, value_cache = value_cache, jacobian_cache = jc
        )
        start += len
    end
    # Finally add the damping term
    (penalty != 0) && (c .+= penalty .* cX)
    return c
end
# for a single block – the actual formula - but never with penalty
function add_linear_normal_operator_coord!(
        M::AbstractManifold, c::AbstractVector, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, cX::AbstractVector;
        value_cache, jacobian_cache, ε::Real, mode::Symbol
    )
    a = value_cache # residuals F(p)
    F_sq = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared indirectly
    # maybe TODO: maybe make this do a more generic conversion?
    b = convert(Vector, jacobian_cache * cX)
    # Compute C^TCb = C^2 b
    # The code below is mathematically equivalent to the following, but avoids allocating
    # the outer product a * a' and the matrix-vector product (a * a') * b
    # b .= ρ_prime .* (I - operator_scaling * (a * a'))^2 * b
    t = dot(a, b)
    aa = dot(a, a)
    coef = operator_scaling * t * (operator_scaling * aa - 2)

    @. b = ρ_prime * (b + coef * a)

    # Now apply the adjoint
    mul!(c, jacobian_cache, b, true, true)
    # penalty is added once after summing up all blocks, so we do not add it here
    return c
end

function add_linear_operator_coord!(
        M::AbstractManifold, y::AbstractVector, lmsco::LevenbergMarquardtLinearCoordinatesSurrogateObjective, p, cX::AbstractVector
    )
    nlso = get_objective(lmsco)
    # Init to zero
    start = 0
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        _add_linear_operator_coord!(
            M, view(y, (start + 1):(start + len)), o, r, p, cX, value_cache, jc;
            ε = lmsco.ε, mode = lmsco.mode
        )
        start += len
    end
    return y
end

function _add_linear_operator_coord!(
        M::AbstractManifold, y::AbstractVector, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, cX::AbstractVector,
        value_cache, jacobian_cache; ε::Real, mode::Symbol
    )
    F_sq = sum(abs2, value_cache)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_sq)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_sq, ε, mode)
    y_cache = jacobian_cache * cX
    # Compute C y
    α = sqrt(ρ_prime)
    t = dot(value_cache, y)
    @. y += α * (y_cache - operator_scaling * t * value_cache)
    return y
end
