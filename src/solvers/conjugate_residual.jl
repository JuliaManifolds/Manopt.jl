@doc """
    ConjugateResidualState{T,R,TStop<:StoppingCriterion,C<:AbstractDict{Symbol}} <: AbstractManoptSolverState

A state for the [`conjugate_residual`](@ref) solver.

# Fields

* `X::T`: the iterate
* `callbacks::C`: the callbacks dictionary
* `r::T`: the residual ``r = -b(p) - $(_tex(:Cal, "A"))(p)[X]``
* `d::T`: the conjugate direction
* `Ar::T`, `Ad::T`: storages for ``$(_tex(:Cal, "A"))(p)[r]``, ``$(_tex(:Cal, "A"))(p)[d]``
* `rAr::R`: internal field for storing ``⟨ r, $(_tex(:Cal, "A"))(p)[r] ⟩``
* `α::R`: a step length
* `β::R`: the conjugate coefficient
$(_fields(:stopping_criterion; name = "stop"))
* `warm_start`: whether to warm start or not when reusing this state, i.e.
  * `true` (default): means we reuse the values in `X` on initialization and set the remaining terms accordingly. This involved one call to the objectives linear system and right hand side.
  * `false`: Initialize `X` to the zero vector and hence `d=r=-b(p)`, but we avoid evaluating the linear operator.

# Constructor

    ConjugateResidualState(TpM::TangentSpace,slso::SymmetricLinearSystemObjective; kwargs...)

Initialize the state with default values.

## Keyword arguments

* `r=-`[`get_gradient`](@ref)`(TpM, slso, X)`
* `d=copy(TpM, r)`
* `Ar=`[`get_hessian`](@ref)`(TpM, slso, X, r)`
* `Ad=copy(TpM, Ar)`
* `α::R=0.0`
* `β::R=0.0`
$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(`$(_link(:manifold_dimension))`)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-8)"))
$(_kwargs(:X; default = _open_link(:rand; M = "TpM")))

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
    Conv = has_converged(crs.stop) ? "Yes" : "No"
    (context === :short) && return repr(crs)
    conv_inl = (i > 0) ? (has_converged(crs.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the conjugate residual solver$(conv_inl)"
    as = _callbacks_summary(crs)
    s = """
    # Solver state for `Manopt.jl`s Conjugate Residual Method
    $Iter
    ## Parameters$(as)
    * α: $(crs.α)
    * β: $(crs.β)

    ## Stopping criterion
    $(_in_str(status_summary(crs.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv
    """
    return s
end
function Base.show(io::IO, crs::ConjugateResidualState)
    print(io, "ConjugateResidualState(;")
    print(io, " X = ", crs.X, ", d = ", crs.d, ", r = ", crs.r, ", α = ", crs.α, ", β = ", crs.β)
    print(io, ", Ar = ", crs.Ar, ", Ad = ", crs.Ad, ", rAr = ", crs.rAr)
    print(io, ", stopping_criterion = ", status_summary(crs.stop; context = :short))
    return print(io, ")")
end

#
# A specific Stopping Criterion
# ---
@doc """
    StopWhenRelativeResidualLess <: StoppingCriterion

Stop when the relative residual in the [`conjugate_residual`](@ref)
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
* `norm_r`: the last computed norm of the residual

# Constructor

    StopWhenRelativeResidualLess(c, ε; norm_r = 2*c*ε)

Initialize the stopping criterion.

!!! note

    The initial norm of the vector field ``c = $(_tex(:norm, "b"))``
    that is stored internally is updated on initialization, that is,
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
requires_update(::Type{<:StopWhenRelativeResidualLess}) = false
function show(io::IO, swrr::StopWhenRelativeResidualLess)
    return print(io, "StopWhenRelativeResidualLess($(swrr.c), $(swrr.ε))")
end

#
# The Conjugate Residual Method
# ---
_doc_conjugate_residual = """
    conjugate_residual(TpM::TangentSpace, A, b, X=zero_vector(TpM))
    conjugate_residual(TpM::TangentSpace, slso::AbstractSymmetricLinearSystemObjective, X=zero_vector(TpM))
    conjugate_residual!(TpM::TangentSpace, A, b, X)
    conjugate_residual!(TpM::TangentSpace, slso::AbstractSymmetricLinearSystemObjective, X)

Compute the solution of ``$(_tex(:Cal, "A"))(p)[X] + b(p) = 0_p ``, where

* ``$(_tex(:Cal, "A"))`` is a linear, symmetric operator on ``$(_math(:TangentSpace))``
* ``b`` is a vector field on the manifold
* ``X ∈ $(_math(:TangentSpace))`` is a tangent vector
* ``0_p`` is the zero vector ``$(_math(:TangentSpace))``.

This implementation follows Algorithm 3 in [LaiYoshise:2024](@cite) and
is initialized with ``X^{(0)}`` as the zero vector and

* the initial residual ``r^{(0)} = -b(p) - $(_tex(:Cal, "A"))(p)[X^{(0)}]``
* the initial conjugate direction ``d^{(0)} = r^{(0)}``
* initialize ``Y^{(0)} = $(_tex(:Cal, "A"))(p)[X^{(0)}]``

performed the following steps at iteration ``k=0,…`` until the `stopping_criterion` is fulfilled.

1. compute a step size ``α_k = $(_tex(:displaystyle))$(_tex(:frac, "⟨ r^{(k)}, $(_tex(:Cal, "A"))(p)[r^{(k)}] ⟩_p", "⟨ $(_tex(:Cal, "A"))(p)[d^{(k)}], $(_tex(:Cal, "A"))(p)[d^{(k)}] ⟩_p"))``
2. do a step ``X^{(k+1)} = X^{(k)} + α_kd^{(k)}``
2. update the residual ``r^{(k+1)} = r^{(k)} + α_k Y^{(k)}``
4. compute ``Z = $(_tex(:Cal, "A"))(p)[r^{(k+1)}]``
5. Update the conjugate coefficient ``β_k = $(_tex(:displaystyle))$(_tex(:frac, "⟨ r^{(k+1)}, $(_tex(:Cal, "A"))(p)[r^{(k+1)}] ⟩_p", "⟨ r^{(k)}, $(_tex(:Cal, "A"))(p)[r^{(k)}] ⟩_p"))``
6. Update the conjugate direction ``d^{(k+1)} = r^{(k+1)} + β_kd^{(k)}``
7. Update  ``Y^{(k+1)} = -Z + β_k Y^{(k)}``

Note that the right hand side of Step 7 is the same as evaluating ``$(_tex(:Cal, "A"))[d^{(k+1)}]``, but avoids the actual evaluation

# Input

* `TpM` the [`TangentSpace`](@extref `ManifoldsBase.TangentSpace`) as the domain
* `A` a symmetric linear operator on the tangent space `(M, p, X) -> Y`
* `b` a vector field on the tangent space `(M, p) -> X`
* `X` the initial tangent vector

# Keyword arguments

$(_kwargs(:evaluation))
$(_kwargs(:callbacks; add_properties = [:process_note]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(`$(_link(:manifold_dimension))`)`$(_sc(:Any))[`StopWhenRelativeResidualLess`](@ref)`(c,1e-8)"))
  where ``c = $(_tex(:norm, "b"))`` is the norm of the vector field `b` at `p`.
$(_note(:OutputSection))
"""

@doc "$_doc_conjugate_residual"
conjugate_residual(TpM::TangentSpace, args...; kwargs...)

function conjugate_residual(
        TpM::TangentSpace, A, b, X = zero_vector(TpM);
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = base_point(TpM), kwargs...,
    )
    slso = SymmetricLinearSystemObjective(A, b; evaluation = evaluation, p = p)
    return conjugate_residual(TpM, slso, X; kwargs...)
end
function conjugate_residual(
        TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X = zero_vector(TpM); kwargs...
    )
    keywords_accepted(conjugate_residual; kwargs...)
    Y = copy(TpM, X)
    return conjugate_residual!(TpM, aslso, Y; kwargs...)
end
calls_with_kwargs(::typeof(conjugate_residual)) = (conjugate_residual!,)

@doc "$_doc_conjugate_residual"
conjugate_residual!(TpM::TangentSpace, args...; kwargs...)

function conjugate_residual!(
        TpM::TangentSpace, A, b, X;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = base_point(TpM), kwargs...,
    )
    slso = SymmetricLinearSystemObjective(A, b; evaluation = evaluation, p = p)
    return conjugate_residual!(TpM, slso, X; kwargs...)
end

function conjugate_residual!(
        TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X;
        callbacks = Dict{Symbol, Function}(),
        stopping_criterion::SC = StopAfterIteration(manifold_dimension(TpM)) |
            StopWhenRelativeResidualLess(
            norm(base_manifold(TpM), base_point(TpM), get_vector_field(TpM, aslso)), 1.0e-8
        ),
        kwargs...,
    ) where {SC <: StoppingCriterion}
    keywords_accepted(conjugate_residual!; kwargs...)
    crs = ConjugateResidualState(
        TpM, aslso; X = X, callbacks = process_callbacks_arg(callbacks, ConjugateResidualState), stopping_criterion = stopping_criterion, kwargs...
    )
    dslso = decorate_objective!(TpM, aslso; kwargs...)
    dmp = DefaultManoptProblem(TpM, dslso)
    dcrs = decorate_state!(crs; kwargs...)
    solve!(dmp, dcrs)
    return get_solver_return(get_objective(dmp), dcrs)
end
calls_with_kwargs(::typeof(conjugate_residual!)) = (ConjugateResidualState, decorate_objective!, decorate_state!)

provided_callbacks(::Type{<:ConjugateResidualState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:Stepsize])

function initialize_solver!(
        amp::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState
    )
    TpM = get_manifold(amp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    if crs.warm_start
        get_linear_operator!(M, crs.r, get_objective(amp), p, crs.X)
        crs.r .*= -1
    else
        zero_vector!(M, crs.X, p)
        zero_vector!(M, crs.r, p)
    end
    crs.r .-= get_vector_field(M, get_objective(amp), p)
    copyto!(TpM, crs.d, crs.r)
    get_hessian!(amp, crs.Ar, crs.X, crs.r)
    copyto!(TpM, crs.Ad, crs.Ar)
    crs.α = 0.0
    crs.β = 0.0
    return crs
end

function step_solver!(
        amp::AbstractManoptProblem{<:TangentSpace}, crs::ConjugateResidualState, i
    )
    TpM = get_manifold(amp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    crs.α = inner(M, p, crs.r, crs.Ar) / inner(M, p, crs.Ad, crs.Ad)
    callback(:Stepsize, amp, crs, i)
    crs.X .+= crs.α .* crs.d
    crs.rAr = inner(M, p, crs.r, crs.Ar)
    crs.r .-= crs.α .* crs.Ad
    get_hessian!(amp, crs.Ar, crs.X, crs.r)
    crs.β = inner(M, p, crs.r, crs.Ar) / crs.rAr
    crs.d .= crs.r .+ crs.β .* crs.d
    crs.Ad .= crs.Ar .+ crs.β .* crs.Ad
    return crs
end

get_solver_result(crs::ConjugateResidualState) = crs.X
