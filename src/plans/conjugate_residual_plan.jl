@doc """
    ConjugateResidualState{T,R,TStop<:StoppingCriterion,C<:AbstractDict{Symbol}} <: AbstractManoptSolverState

A state for the [`conjugate_residual`](@ref) solver.

# Fields

* `X::T`: the iterate
* `callbacks::C`: the callbacks dictionary
* `r::T`: the residual ``r = -b(p) - $(_tex(:Cal, "A"))(p)[X]``
* `d::T`: the conjugate direction
* `Ar::T`, `Ad::T`: storages for ``$(_tex(:Cal, "A"))(p)[d]``, ``$(_tex(:Cal, "A"))(p)[r]``
* `rAr::R`: internal field for storing ``⟨ r, $(_tex(:Cal, "A"))(p)[r] ⟩``
* `α::R`: a step length
* `β::R`: the conjugate coefficient
$(_fields(:stopping_criterion; name = "stop"))
* `warm_start`: whether to warm start or not when reusing this state, i.e.
  * `true` (default): means we reuse the values in `X` on initialization and set the remaining terms accordingly. This involved one call to the objectives linear system and right hand side.
  * `false`: Initialize `X` to the zero vector and hence `d=r=-b(p)`, but we avoid evaluating the linear operator.

# Constructor

    ConjugateResidualState(TpM::TangentSpace,slso::SymmetricLinearSystemObjective; kwargs...)

Initialise the state with default values.

## Keyword arguments

* `r=-`[`get_gradient`](@ref)`(TpM, slso, X)`
* `d=copy(TpM, r)`
* `Ar=`[`get_hessian`](@ref)`(TpM, slso, X, r)`
* `Ad=copy(TpM, Ar)`
* `α::R=0.0`
* `β::R=0.0`
$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(`$(_link(:manifold_dimension))`)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-8)"))
$(_kwargs(:X))

# See also

[`conjugate_residual`](@ref)
"""
mutable struct ConjugateResidualState{T, R, TStop <: StoppingCriterion, C <: AbstractDict{Symbol}} <:
    AbstractManoptSolverState
    callbacks::C
    X::T
    r::T
    d::T
    Ar::T
    Ad::T
    rAr::R
    α::R
    β::R
    stop::TStop
    warm_start::Bool
    function ConjugateResidualState(;
            callbacks::C, X::T, r::T, d::T, Ar::T, Ad::T, α::R, β::R, rAr::R, stopping_criterion::SC, warm_start::Bool
        ) where {T, R, SC <: StoppingCriterion, C <: AbstractDict{Symbol}}
        crs = new{T, R, SC, C}()
        crs.callbacks = callbacks; crs.X = X; crs.r = r; crs.d = d; crs.Ar = Ar; crs.Ad = Ad
        crs.α = α; crs.β = β; crs.rAr = rAr; crs.stop = stopping_criterion
        crs.warm_start = warm_start
        return crs
    end
    function ConjugateResidualState(
            TpM::TangentSpace,
            aslso::AbstractSymmetricLinearSystemObjective;
            callbacks::C = Dict{Symbol, Function}(),
            X::T = rand(TpM), r::T = (-get_gradient(TpM, aslso, X)), d::T = copy(TpM, r),
            Ar::T = get_hessian(TpM, aslso, X, r), Ad::T = copy(TpM, Ar), α::Real = 0.0, β::Real = 0.0,
            stopping_criterion::SC = StopAfterIteration(manifold_dimension(TpM)) | StopWhenGradientNormLess(1.0e-8),
            warm_start::Bool = true,
            kwargs...,
        ) where {T, SC <: StoppingCriterion, C <: AbstractDict{Symbol}}
        R = promote_type(typeof(α), typeof(β))
        return ConjugateResidualState(; callbacks = callbacks, X = X, r = r, d = d, Ar = Ar, Ad = Ad, α = α, β = β, rAr = zero(R), stopping_criterion = stopping_criterion, warm_start = warm_start)
    end
end

get_callbacks(crs::ConjugateResidualState) = crs.callbacks

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

function status_summary(crs::ConjugateResidualState; context::Symbol = :default)
    i = get_count(crs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(crs.stop) ? "Yes" : "No"
    _is_inline(context) && (return "$(repr(crs)) – $(Iter) $(has_converged(crs) ? "(converged)" : "")")
    as = _callbacks_summary(crs)
    s = """
    # Solver state for `Manopt.jl`s Conjugate Residual Method
    $Iter
    ## Parameters$(as)
    * α: $(crs.α)
    * β: $(crs.β)

    ## Stopping criterion
    $(_in_str(status_summary(crs.stop; context = context); indent = 0, headers = 1))
    This indicates convergence: $Conv
    """
    return s
end

function Base.show(io::IO, crs::ConjugateResidualState)
    print(io, "ConjugateResidualState(;")
    print(io, " X = ", crs.X, ", d = ", crs.d, ", r = ", crs.r, ", α = ", crs.α, ", β = ", crs.β)
    print(io, "Ar = ", crs.Ar, ", Ad = ", crs.Ad, ", rAr = ", crs.rAr)
    print(io, ", stopping_criterion = ", status_summary(crs.stop; context = :short))
    return print(io, ")")
end

#
#
# Stopping Criterion
@doc """
    StopWhenRelativeResidualLess <: StoppingCriterion

Stop when re relative residual in the [`conjugate_residual`](@ref)
is below a certain threshold, i.e.

```math
$(_tex(:displaystyle))$(_tex(:frac, _tex(:norm, "r^{(k)}"), "c")) ≤ ε,
```

where ``c = $(_tex(:norm, "b"))`` of the initial vector from the vector field in ``$(_tex(:Cal, "A"))(p)[X] + b(p) = 0_p``,
from the [`conjugate_residual`](@ref)

# Fields

$(_fields(:at_iteration))
* `c`: the initial norm
* `ε`: the threshold
* `norm_rk`: the last computed norm of the residual

# Constructor

    StopWhenRelativeResidualLess(c, ε; norm_r = 2*c*ε)

Initialise the stopping criterion.

!!! note

    The initial norm of the vector field ``c = $(_tex(:norm, "b"))``
    that is stored internally is updated on initialisation, that is,
    if this stopping criterion is called with `k<=0`.
"""
mutable struct StopWhenRelativeResidualLess{R} <: StoppingCriterion
    c::R
    ε::R
    norm_r::R
    at_iteration::Int
    function StopWhenRelativeResidualLess(c::R, ε::R; norm_r::R = 2 * c * ε) where {R}
        return new{R}(c, ε, norm_r, -1)
    end
end
function (swrr::StopWhenRelativeResidualLess)(
        amp::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState, k::Int
    )
    TpM = get_manifold(amp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    #compute current r-norm
    swrr.norm_r = norm(M, p, crs.r)
    if k <= 0
        # on init also update the right hand side norm
        swrr.c = norm(M, p, get_vector_field(M, get_objective(amp), p))
        return false # just init the norm, but do not stop
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
        return "After iteration #$(swrr.at_iteration) the algorithm stopped with a relative residual $(swrr.norm_r / swrr.c) < $(swrr.ε).\n"
    end
    return ""
end
function status_summary(swrr::StopWhenRelativeResidualLess; context::Symbol = :default)
    has_stopped = (swrr.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return _is_inline(context) ? "‖r^(k)‖ / c < ε:$(_MANOPT_INDENT)$s" : "A stopping criterion to stop when the relative residual is less than the threshold of $(swrr.ε)\n$(_MANOPT_INDENT)$s"
end
indicates_convergence(::StopWhenRelativeResidualLess) = true
function show(io::IO, swrr::StopWhenRelativeResidualLess)
    return print(io, "StopWhenRelativeResidualLess($(swrr.c), $(swrr.ε))")
end
