# TODO (RB -> MB, 12/03): Order functions here alphabetically
# TODO (RB -> MB|RB, 12/03): All docs should be thoroughly written

@doc """
    LevenbergMarquardtLinearSurrogateCoordinatesObjective{E<:AbstractEvaluationType, VF<:AbstractManifoldFirstOrderObjective{E}, R} <: AbstractLevenbergMarquardtLinearSurrogateObjective{E}


A subobjective similar to `LevenbergMarquardtLinearSurrogateObjective` but which uses
coordinate-based Jacobians in a single, selected basis instead of being centered around
linear operators.
## Fields

* `objective`:     the [`NonlinearLeastSquaresObjective`](@ref) to penalize
* `penalty::Real`: the damping term ``λ``
* `ε::Real`:       stabilization for ``α ≤ 1-ε`` in the rescaling of the Jacobian, that
* `mode::Symbol`:  which mode to use to stabilize α, see the internal helper [`get_LevenbergMarquardt_scaling`](@ref)
* `value_cache`:   a vector to store the residuals ``F(p)`` at the current point `p` internally to avoid recomputations
* `jacobian_cache`: a vector to store the coordinate-based Jacobian of the residuals at the
  current point `p` internally to avoid recomputations. If the Jacobian is used as a linear
  operator, this is just a vector of `nothing`s.

## Constructor

    LevenbergMarquardtLinearSurrogateCoordinatesObjective(objective; penalty::Real = 1e-6, ε::Real = 1e-4, mode::Symbol = :Default )
"""
mutable struct LevenbergMarquardtLinearSurrogateCoordinatesObjective{
        E <: AbstractEvaluationType, R <: Real, TO <: NonlinearLeastSquaresObjective{E}, TVC <: AbstractVector{R}, TJC <: AbstractVector, TB <: AbstractBasis,
    } <: AbstractLevenbergMarquardtLinearSurrogateObjective{E}
    objective::TO
    penalty::R
    ε::R
    mode::Symbol
    value_cache::TVC
    jacobian_cache::TJC
    basis::TB
    function LevenbergMarquardtLinearSurrogateCoordinatesObjective(
            objective::NonlinearLeastSquaresObjective{E};
            penalty::R = 1.0e-6, ε::R = 1.0e-4, mode::Symbol = :Default,
            residuals::TVC = zeros(residuals_count(get_objective(objective))),
            jacobian_cache::TJC = fill(nothing, length(get_objective(objective).objective)),
            basis::TB = DefaultOrthonormalBasis(),
        ) where {E, R <: Real, TVC <: AbstractVector, TJC <: AbstractVector, TB <: AbstractBasis}
        return new{E, R, typeof(objective), TVC, TJC, TB}(objective, penalty, ε, mode, residuals, jacobian_cache, basis)
    end
end

function set_parameter!(lmlso::LevenbergMarquardtLinearSurrogateCoordinatesObjective, ::Val{:Penalty}, penalty::Real)
    lmlso.penalty = penalty
    return lmlso
end

# Adapt the get_normal_linear_operator! to also pass down the Jacobian cache.
# similar to nls_general line 1149 (second in case (b)) for the case with different bases.
function get_normal_linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco.objective)
    # For every block
    fill!(A, 0)
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len_o = length(o)
        add_normal_linear_operator_coord!(
            M, A, o, r, p, B; value_cache = view(lmsco.value_cache, (start + 1):(start + len_o)), jacobian_cache = jc,
            ε = lmsco.ε, mode = lmsco.mode
        )
        start += len_o
    end
    # Finally add the damping term
    (penalty != 0) && (LinearAlgebra.diagview(A) .+= penalty)
    return A
end

# adapted from nls_general line 1175 (first add case) to use the jacoban cache.
function add_normal_linear_operator_coord!(
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

function get_linear_operator!(
        M::AbstractManifold, A::AbstractMatrix, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateCoordinatesObjective}, p, B::AbstractBasis;
        penalty::Real = neo.objective.penalty,
    ) where {E <: AbstractEvaluationType}
    return get_normal_linear_operator!(M, A, neo.objective, p, B; penalty = penalty)
end

# same as above, pass down Jacobian cache, but otherwise similar to nls_general line 1135 (just that there the basis is passed down)
function get_normal_vector_field_coord!(
        M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p,
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
    mul!(c, JFt, y, true, true)
    return c
end

# similar to the single block ones in nls_general (l. 1348)
function get_normal_vector_field_coord!(
        M::AbstractManifold, c::AbstractVector, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p,
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(c, 0)
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len_o = length(o)
        add_normal_vector_field_coord!(
            M, c, o, r, p;
            value_cache = view(lmsco.value_cache, (start + 1):(start + len_o)),
            jacobian_cache = jc, ε = lmsco.ε, mode = lmsco.mode
        )
        start += len_o
    end
    return c
end

# for a single block – the actual formula cf. nls_general 1348
function add_normal_vector_field_coord!(
        M::AbstractManifold, c::AbstractVector, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
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

function get_vector_field!(
        M::AbstractManifold, c, neo::NormalEquationsObjective{E, <:LevenbergMarquardtLinearSurrogateCoordinatesObjective}, p, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    get_normal_vector_field_coord!(M, c, neo.objective, p)
    c .*= -1
    return c
end
function get_solver_result(
        dmp::DefaultManoptProblem{<:TangentSpace, <:NormalEquationsObjective{<:AbstractEvaluationType, <:LevenbergMarquardtLinearSurrogateCoordinatesObjective}},
        cnss::CoordinatesNormalSystemState
    )
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_vector(M, p, cnss.c, cnss.basis)
end

function get_cost(
        TpM::TangentSpace, lnsco::NormalEquationsObjective{<:AbstractEvaluationType, <:LevenbergMarquardtLinearSurrogateCoordinatesObjective},
        ::ZeroTangentVector
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    # TODO: optimize?
    n = residuals_count(lnsco.objective.objective)
    vf = zeros(number_eltype(p), n)
    get_vector_field!(M, vf, lnsco.objective, p)
    return 0.5 * norm(vf)^2
end

function get_cost(
        TpM::TangentSpace, lnsco::NormalEquationsObjective{<:AbstractEvaluationType, <:LevenbergMarquardtLinearSurrogateCoordinatesObjective},
        X,
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    # TODO: optimize?
    cX = get_coordinates(M, p, X)
    n = residuals_count(lnsco.objective.objective)
    vf = zeros(number_eltype(p), n)
    get_vector_field!(M, vf, lnsco.objective, p)
    add_linear_operator_coord!(TpM, vf, lnsco.objective, p, cX)
    cost = 0.5 * norm(vf)^2
    cost += (lnsco.objective.penalty / 2) * norm(M, p, X)^2
    return cost
end

function add_normal_linear_operator_coord!(
        M::AbstractManifold, c::AbstractVector,
        lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p, cX::AbstractVector;
        penalty::Real = lmsco.penalty,
    )
    nlso = get_objective(lmsco)
    # For every block
    # lmsco.value_cache has been filled in step_solver! of LevenbergMarquardt, so we can just use it here
    start = 0
    for (o, r, jc) in zip(nlso.objective, nlso.robustifier, lmsco.jacobian_cache)
        len = length(o)
        value_cache = view(lmsco.value_cache, (start + 1):(start + len))
        add_normal_linear_operator_coord!(
            M, c, o, r, p, cX;
            ε = lmsco.ε, mode = lmsco.mode, value_cache = value_cache, jacobian_cache = jc
        )
        start += len
    end
    # Finally add the damping term
    (penalty != 0) && (c .+= penalty .* cX)
    return c
end
function add_normal_linear_operator_coord!(
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
    mul!(c, jacobian_cache', b, true, true)
    # penalty is added once after summing up all blocks, so we do not add it here
    return c
end
"""
    add_linear_operator_coord!(
        M::AbstractManifold, y::AbstractVector, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p, cX::AbstractVector
    )

Add the (Triggs correction, residual-like) linear operator corresponding to the `lmsco`
surrogate to vector `y`. It is assumed that `lmsco.value_cache` has been filled in 
`step_solver!` of `LevenbergMarquardt``, so we can just use it here.
"""
function add_linear_operator_coord!(
        M::AbstractManifold, y::AbstractVector, lmsco::LevenbergMarquardtLinearSurrogateCoordinatesObjective, p, cX::AbstractVector
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
    t = dot(value_cache, y_cache)
    @. y += α * (y_cache - operator_scaling * t * value_cache)
    return y
end
