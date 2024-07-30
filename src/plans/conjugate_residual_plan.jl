#
#
# Objective.
@doc raw"""
    SymmetricLinearSystemObjective{E<:AbstractEvaluationType,TA,T} <: AbstractManifoldObjective{E}

Model the objective

```math
f(X) = \frac{1}{2} \lVert \mathcal A[X] + b \rVert_p^2,\qquad X ∈ T_p\mathcal M,
```

defined on the tangent space ``T_p\mathcal M`` at ``p`` on the manifold ``\mathcal M``.

In other words this is an objective to solve ``\mathcal A(p)[X] = -b(p)``
for some linear symmetric operator and a vector function.
Note the minus on the right hand side, which makes this objective especially taylored
for (iteratively) solving Newton-like equations.

# Fields

* `A!!`: a symmetric, linear operator on the tangent space
* `b!!`: a gradient function

where ``A!!`` can work as an allocating operator `(M, p, X) -> Y` or an in-place one `(M, Y, p, X) -> Y`,
and similarly ``b`` can either be a function `(M, p) -> X` or `(M, X, p) -> X`

# Constructor

    SymmetricLinearSystemObjective(A, b; evaluation=AllocatingEvaluation())

Generate the objective specifying whether the two parts work allocating or in-place.
"""
mutable struct SymmetricLinearSystemObjective{E<:AbstractEvaluationType,TA,T} <:
               AbstractManifoldObjective{E}
    A!!::TA
    b!!::T
end

function SymmetricLinearSystemObjective(
    A::TA, b::T; evaluation::E=AllocatingEvaluation(), kwargs...
) where {TA,T,E<:AbstractEvaluationType}
    return SymmetricLinearSystemObjective{E,TA,T}(A, b)
end

function set_manopt_parameter!(slso::SymmetricLinearSystemObjective, symbol::Symbol, value)
    set_manopt_parameter!(slso.A!!, symbol, value)
    set_manopt_parameter!(slso.b!!, symbol, value)
    return slso
end

function get_cost(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return 0.5 * norm(M, p, slso.A!!(M, p, X) + slso.b!!(M, p))^2
end
function get_cost(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    Y = zero_vector(M, p)
    W = copy(M, p, Y)
    slso.b!!(M, Y, p)
    slso.A!!(M, W, p, X)
    return 0.5 * norm(M, p, W + Y)^2
end

function get_b(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X
)
    return slso.b!!(base_manifold(TpM), base_point(TpM))
end
function get_b(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    Y = zero_vector(M, p)
    return slso.b!!(M, Y, p)
end

function get_gradient(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, X)
    p = base_point(TpM)
    return get_hessian(TpM, slso, p, X) + get_b(TpM, slso, X)
end
function get_gradient!(
    TpM::TangentSpace, Y, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    # Evaluate A[X] + b
    Y .= slso.A!!(M, p, X) + slso.b!!(M, p)
    return Y
end
function get_gradient!(
    TpM::TangentSpace, Y, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    W = copy(M, p, Y)
    slso.b!!(M, Y, p)
    slso.A!!(M, W, p, X)
    Y .+= W
    return Y
end

# evaluate Hessian: ∇²Q(X) = A[X]
function get_hessian(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X, V
)
    return slso.A!!(base_manifold(TpM), base_point(TpM), V)
end
function get_hessian(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X, V
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    W = copy(M, p, V)
    slso.A!!(M, W, p, V)
    return W
end
function get_hessian!(
    TpM::TangentSpace, W, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X, V
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    copyto!(M, W, p, slso.A!!(M, p, V))
    return W
end
function get_hessian!(
    TpM::TangentSpace, W, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X, V
)
    return slso.A!!(base_manifold(TpM), W, base_point(TpM), V)
end

#
#
# State
@doc raw"""
    ConjugateResidualState{T,R,TStop<:StoppingCriterion} <: AbstractManoptSolverState

A state for the [`conjugate_residual`](@ref) solver.

# Fields

* `X::T`: the iterate
* `r::T`: the residual ``r = -b(p) - \mathcal A(p)[X]``
* `d::T`: the conjugate direction
* `Ar::T`, `Ad::T`: storages for ``\mathcal A``
* `rAr::R`: internal field for storing ``⟨ r, \mathcal A(p)[r] ⟩``
* `α::R`: a step length
* `β::R`: the conjugate coefficient
* `stop::TStop`: a [`StoppingCriterion`] for the solver

# Constructor

        function ConjugateResidualState(
            TpM::TangentSpace,
            slso::SymmetricLinearSystemObjective;
            X=rand(TpM),
            r=-get_gradient(TpM, slso, X),
            d=copy(TpM, r),
            Ar=get_hessian(TpM, slso, X, r),
            Ad=copy(TpM, Ar),
            α::R=0.0,
            β::R=0.0,
            stopping_criterion=StopAfterIteration(manifold_dimension(TpM)) |
                               StopWhenGradientNormLess(1e-8),
            kwargs...,
    )

    Initialise the state with default values.
"""
mutable struct ConjugateResidualState{T,R,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    X::T
    r::T
    d::T
    Ar::T
    Ad::T
    rAr::R
    α::R
    β::R
    stop::TStop
    function ConjugateResidualState(
        TpM::TangentSpace,
        slso::SymmetricLinearSystemObjective;
        X::T=rand(TpM),
        r::T=-get_gradient(TpM, slso, X),
        d::T=copy(TpM, r),
        Ar::T=get_hessian(TpM, slso, X, r),
        Ad::T=copy(TpM, Ar),
        α::R=0.0,
        β::R=0.0,
        stopping_criterion::SC=StopAfterIteration(manifold_dimension(TpM)) |
                               StopWhenGradientNormLess(1e-8),
        kwargs...,
    ) where {T,R,SC<:StoppingCriterion}
        M = base_manifold(TpM)
        p = base_point(TpM)
        crs = new{T,R,SC}()
        crs.X = X
        crs.r = r
        crs.d = d
        crs.Ar = Ar
        crs.Ad = Ad
        crs.α = α
        crs.β = β
        crs.rAr = zero(R)
        crs.stop = stopping_criterion
        return crs
    end
end

get_iterate(crs::ConjugateResidualState) = crs.X
function set_iterate!(crs::ConjugateResidualState, ::AbstractManifold, X)
    crs.X = X
    return crs
end

get_gradient(crs::ConjugateResidualState) = crs.r
function set_gradient!(crs::ConjugateResidualState, ::AbstractManifold, r)
    crs.r = r
    return crs
end

function get_message(crs::ConjugateResidualState)
    return get_message(crs.α)
end

function show(io::IO, crs::ConjugateResidualState)
    i = get_count(crs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(crs.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Conjugate Residual Method
    $Iter
    ## Parameters
    * α: $(crs.α)
    * β: $(crs.β)

    ## Stopping criterion
    $(status_summary(crs.stop))

    This indicates convergence: $Conv
    """
    return print(io, s)
end

#
#
# Stopping Criterion
@doc raw"""
    StopWhenRelativeResidualLess <: StoppingCriterion

Stop when re relative residual in the [`conjugate_residual`](@ref)
is below a certain threshold, i.e.

```math
  \frac{\lVert r^{(k)} \rVert}{c} ≤ ε,
```

where ``c = \lVert b \rVert of the initial vector from the vector field in ``\mathcal A(p)[X] = -b(p)``,
from the [`conjugate_residual`](@ref)

# Fields

* `at_iteration` indicates at which iteration (including `i=0`) the stopping criterion
  was fulfilled and is `-1` while it is not fulfilled.
* `c`: the initial norm
* `ε`: the threshold
* `norm_rk`: the last computed norm of the residual

# Constructor

    StopWhenRelativeResidualLess(c, ε; norm_r = 2*c*ε)


"""
mutable struct StopWhenRelativeResidualLess{R} <: StoppingCriterion
    c::R
    ε::R
    norm_r::R
    at_iteration::Int
    function StopWhenRelativeResidualLess(c::R, ε::R; norm_r::R=2 * c * ε) where {R}
        return new{R}(c, ε, norm_r, 0.1)
    end
end
function (swrr::StopWhenRelativeResidualLess)(
    amp::AbstractManoptProblem{<:TangentSpace}, ips::ConjugateResidualState, k::Int
)
    TpM = get_manifold(amp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    swrr.norm_r = norm(M, p, ips.r)
    if k <= 0 #init -> reset r-norm
        swrr.norm_r = 2 * swrr.c * swrr.ε
        return true
    end
    # now k > 0
    if swrr.norm_r / swrr.c < swrr.ε #residual small enough
        swrr.at_iteration = k
        return true
    end
    return false
end
function get_reason(swrr::StopWhenRelativeResidualLess)
    if (swrr.at_iteration >= 0)
        return "After iteration #$(swrr.at_iteration) the algorithm stopped with a relative residual $(swrr.norm_r/swrr.c) < $(swrr.ε).\n"
    end
    return ""
end
function status_summary(swrr::StopWhenRelativeResidualLess)
    has_stopped = (swrr.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "‖r^(k)‖ / c < $(swrr.ε):\t$s"
end
indicates_convergence(::StopWhenRelativeResidualLess) = true
function show(io::IO, swrr::StopWhenRelativeResidualLess)
    return print(
        io, "StopWhenRelativeResidualLess($(swrr.c), $(swrr.ε))\n    $(status_summary(c))"
    )
end
