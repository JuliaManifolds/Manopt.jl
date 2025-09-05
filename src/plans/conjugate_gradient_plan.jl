struct DirectionUpdateRuleStorage{TC <: DirectionUpdateRule, TStorage <: StoreStateAction}
    coefficient::TC
    storage::TStorage
end

function DirectionUpdateRuleStorage(
        M::AbstractManifold,
        dur::DirectionUpdateRule;
        p_init = rand(M),
        X_init = zero_vector(M, p_init),
    )
    ursp = update_rule_storage_points(dur)
    ursv = update_rule_storage_vectors(dur)
    # StoreStateAction makes a copy
    sa = StoreStateAction(
        M; store_points = ursp, store_vectors = ursv, p_init = p_init, X_init = X_init
    )
    return DirectionUpdateRuleStorage{typeof(dur), typeof(sa)}(dur, sa)
end

@doc """
    ConjugateGradientState <: AbstractGradientSolverState

specify options for a conjugate gradient descent algorithm, that solves a
[`DefaultManoptProblem`].

# Fields

$(_var(:Field, :p; add = [:as_Iterate]))
$(_var(:Field, :X))
* `δ`:                       the current descent direction, also a tangent vector
* `β`:                       the current update coefficient rule, see .
* `coefficient`:             function to determine the new `β`
* `restart_condition`:       an [`AbstractRestartCondition`](@ref) to determine how to handle non-descent directions.
$(_var(:Field, :stepsize))
$(_var(:Field, :stopping_criterion, "stop"))
$(_var(:Field, :retraction_method))
$(_var(:Field, :vector_transport_method))

# Constructor

    ConjugateGradientState(M::AbstractManifold; kwargs...)

where the last five fields can be set by their names as keyword and the
`X` can be set to a tangent vector type using the keyword `initial_gradient` which defaults to `zero_vector(M,p)`,
and `δ` is initialized to a copy of this vector.

## Keyword arguments

The following fields from above <re keyword arguments

$(_var(:Keyword, :X, "initial_gradient"))
$(_var(:Keyword, :p; add = :as_Initial))
* `coefficient=[`ConjugateDescentCoefficient`](@ref)`()`: specify a CG coefficient, see also the [`ManifoldDefaultsFactory`](@ref).
* `restart_condition=[`NeverRestart`](@ref)`()`: specify a [restart condition](@ref cg-restart). It defaults to never restart.
$(_var(:Keyword, :stepsize; default = "[`default_stepsize`](@ref)`(M, ConjugateGradientDescentState; retraction_method=retraction_method)`"))
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-8)`)"))
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :vector_transport_method))

# See also

[`conjugate_gradient_descent`](@ref), [`DefaultManoptProblem`](@ref), [`ArmijoLinesearch`](@ref)
"""
mutable struct ConjugateGradientDescentState{
        P,
        T,
        F,
        TStepsize <: Stepsize,
        TStop <: StoppingCriterion,
        TCoeff <: DirectionUpdateRuleStorage,
        TRC <: AbstractRestartCondition,
        TRetr <: AbstractRetractionMethod,
        VTM <: AbstractVectorTransportMethod,
    } <: AbstractGradientSolverState
    p::P
    p_old::P
    X::T
    δ::T
    β::F
    coefficient::TCoeff
    restart_condition::TRC
    stepsize::TStepsize
    stop::TStop
    retraction_method::TRetr
    vector_transport_method::VTM
    function ConjugateGradientDescentState(
            M::AbstractManifold,
            p::P,
            sC::TsC,
            s::TStep,
            dC::DirectionUpdateRule,
            res_cond::TRC = NeverRestart(),
            retr::TRetr = default_retraction_method(M, typeof(p)),
            vtr::VTM = default_vector_transport_method(M),
            initial_gradient::T = zero_vector(M, p),
        ) where {
            P,
            T,
            TsC <: StoppingCriterion,
            TStep <: Stepsize,
            TRC <: AbstractRestartCondition,
            TRetr <: AbstractRetractionMethod,
            VTM <: AbstractVectorTransportMethod,
        }
        coef = DirectionUpdateRuleStorage(M, dC; p_init = p, X_init = initial_gradient)
        βT = allocate_result_type(M, ConjugateGradientDescentState, (p, initial_gradient))
        cgs = new{P, T, βT, TStep, TsC, typeof(coef), TRC, TRetr, VTM}()
        cgs.p = p
        cgs.p_old = copy(M, p)
        cgs.X = initial_gradient
        cgs.δ = copy(M, p, initial_gradient)
        cgs.stop = sC
        cgs.retraction_method = retr
        cgs.stepsize = s
        cgs.coefficient = coef
        cgs.restart_condition = res_cond
        cgs.vector_transport_method = vtr
        cgs.β = zero(βT)
        return cgs
    end
end

function ConjugateGradientDescentState(
        M::AbstractManifold;
        p::P = rand(M),
        coefficient::Union{DirectionUpdateRule, ManifoldDefaultsFactory} = ConjugateDescentCoefficient(),
        restart_condition::TRC = NeverRestart(),
        retraction_method::TRetr = default_retraction_method(M, typeof(p)),
        stepsize::TStep = default_stepsize(
            M, ConjugateGradientDescentState; retraction_method = retraction_method
        ),
        stopping_criterion::TsC = StopAfterIteration(500) | StopWhenGradientNormLess(1.0e-8),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        initial_gradient::T = zero_vector(M, p),
    ) where {
        P,
        T,
        TsC <: StoppingCriterion,
        TStep <: Stepsize,
        TRC <: AbstractRestartCondition,
        TRetr <: AbstractRetractionMethod,
        VTM <: AbstractVectorTransportMethod,
    }
    return ConjugateGradientDescentState(
        M,
        p,
        stopping_criterion,
        stepsize,
        _produce_type(coefficient, M),
        restart_condition,
        retraction_method,
        vector_transport_method,
        initial_gradient,
    )
end

function get_message(cgs::ConjugateGradientDescentState)
    # for now only step size is quipped with messages
    return get_message(cgs.stepsize)
end

function get_gradient(cgs::ConjugateGradientDescentState)
    return cgs.X
end

_doc_CG_notaion = """
Denote the last iterate and gradient by ``p_k,X_k``,
the current iterate and gradient by ``p_{k+1}, X_{k+1}``, respectively,
as well as the last update direction by ``δ_k``.
"""

@doc """
    ConjugateDescentCoefficientRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient adapted to manifolds

See also [`conjugate_gradient_descent`](@ref)

# Constructor

    ConjugateDescentCoefficientRule()

Construct the conjugate descent coefficient update rule, a new storage is created by default.

# See also

[`ConjugateDescentCoefficient`](@ref), [`conjugate_gradient_descent`](@ref)
"""
struct ConjugateDescentCoefficientRule <: DirectionUpdateRule end

"""
    ConjugateDescentCoefficient()
    ConjugateDescentCoefficient(M::AbstractManifold)

Compute the (classical) conjugate gradient coefficient based on [Fletcher:1987](@cite) adapted to manifolds

$(_doc_CG_notaion)

Then the coefficient reads
```math
β_k = $(_tex(:frac, "$(_tex(:diff))f(p_{k+1})[X_{k+1}]", "$(_tex(:diff))f(p_k)[-δ_k]"))
 = $(_tex(:frac, "$(_tex(:norm, "X_{k+1}"; index = "p_{k+1}") * "^2")", "$(_tex(:inner, "-δ_k", "X_k"; index = "p_k"))"))
```

The second one it the one usually stated, while the first one avoids to use the metric `inner`.
The first one is implemented here, but falls back to calling `inner` if there is no dedicated differential available.

$(_note(:ManifoldDefaultFactory, "ConjugateDescentCoefficientRule"))
"""
function ConjugateDescentCoefficient()
    return ManifoldDefaultsFactory(
        Manopt.ConjugateDescentCoefficientRule; requires_manifold = false
    )
end

update_rule_storage_points(::ConjugateDescentCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::ConjugateDescentCoefficientRule) = Tuple{:Gradient}

function (u::DirectionUpdateRuleStorage{ConjugateDescentCoefficientRule})(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
    )
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
            !has_storage(u.storage, VectorStorageKey(:Gradient))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    # previously
    # coeff = inner(M, cgs.p, cgs.X, cgs.X) / inner(M, p_old, -cgs.δ, X_old)
    # now via differential, but also provide gradients for the fallbacks
    nom = get_differential(amp, cgs.p, cgs.X; gradient = cgs.X, evaluated = true)
    denom = get_differential(amp, p_old, -cgs.δ; gradient = X_old, evaluated = true)
    coeff = nom / denom
    update_storage!(u.storage, amp, cgs)
    return coeff
end
function show(io::IO, ::ConjugateDescentCoefficientRule)
    return print(io, "Manopt.ConjugateDescentCoefficientRule()")
end

@doc """
    DaiYuanCoefficientRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient based on [DaiYuan:1999](@cite) adapted to manifolds

# Fields

$(_var(:Field, :vector_transport_method))

# Constructor

    DaiYuanCoefficientRule(M::AbstractManifold; kwargs...)

Construct the Dai—Yuan coefficient update rule.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

# See also

[`DaiYuanCoefficient`](@ref), [`conjugate_gradient_descent`](@ref)
"""
struct DaiYuanCoefficientRule{VTM <: AbstractVectorTransportMethod} <: DirectionUpdateRule
    vector_transport_method::VTM
end
function DaiYuanCoefficientRule(
        M::AbstractManifold; vector_transport_method::VTM = default_vector_transport_method(M)
    ) where {VTM <: AbstractVectorTransportMethod}
    return DaiYuanCoefficientRule{VTM}(vector_transport_method)
end

update_rule_storage_points(::DaiYuanCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::DaiYuanCoefficientRule) = Tuple{:Gradient, :δ}

function (u::DirectionUpdateRuleStorage{<:DaiYuanCoefficientRule})(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
    )
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
            !has_storage(u.storage, VectorStorageKey(:Gradient)) ||
            !has_storage(u.storage, VectorStorageKey(:δ))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    δ_old = get_storage(u.storage, VectorStorageKey(:δ))

    gradienttr = vector_transport_to(
        M, p_old, X_old, cgs.p, u.coefficient.vector_transport_method
    )
    ν = cgs.X - gradienttr #notation y from [HZ06]
    δtr = vector_transport_to(M, p_old, δ_old, cgs.p, u.coefficient.vector_transport_method)
    # previously: nominator = inner(M, cgs.p, cgs.X, cgs.X)
    nominator = get_differential(amp, cgs.p, cgs.X; gradient = cgs.X, evaluated = true)
    β = nominator / inner(M, p_old, δtr, ν)
    update_storage!(u.storage, amp, cgs)
    return β
end
function show(io::IO, u::DaiYuanCoefficientRule)
    return print(
        io,
        "Manopt.DaiYuanCoefficientRule(; vector_transport_method=$(u.vector_transport_method))",
    )
end

@doc """
    DaiYuanCoefficient(; kwargs...)
    DaiYuanCoefficient(M::AbstractManifold; kwargs...)

Computes an update coefficient for the [`conjugate_gradient_descent`](@ref) algorithm based on [DaiYuan:1999](@cite) adapted to
Riemannian manifolds.

$(_doc_CG_notaion)
Let ``ν_k = X_{k+1} - $(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))X_k``,
where ``$(_math(:vector_transport, :symbol))`` denotes a vector transport.

Then the coefficient reads
````math
β_k =
=
$(_tex(:frac, "$(_tex(:diff))f(p_{k+1})[X_{k+1}]", "$(_tex(:inner, "δ_k", "ν_k"; index = "p_{k+1}"))"))
=
$(
    _tex(
        :frac,
        _tex(:norm, "X_{k+1}"; index = "p_{k+1}") * "^2",
        "⟨$(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))δ_k, ν_k⟩_{p_{k+1}}"
    )
)
````

The second one it the one usually stated, while the first one avoids to use the metric `inner`.
The first one is implemented here, but falls back to calling `inner` if there is no dedicated differential available.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

$(_note(:ManifoldDefaultFactory, "DaiYuanCoefficientRule"))
"""
function DaiYuanCoefficient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.DaiYuanCoefficientRule, args...; kwargs...)
end

@doc """
    FletcherReevesCoefficientRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient based on [FletcherReeves:1964](@cite) adapted to manifolds

# Constructor

    FletcherReevesCoefficientRule()

Construct the Fletcher—Reeves coefficient update rule.

# See also
[`FletcherReevesCoefficient`](@ref), [`conjugate_gradient_descent`](@ref)
"""
struct FletcherReevesCoefficientRule <: DirectionUpdateRule end

update_rule_storage_points(::FletcherReevesCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::FletcherReevesCoefficientRule) = Tuple{:Gradient}

function (u::DirectionUpdateRuleStorage{FletcherReevesCoefficientRule})(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
    )
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
            !has_storage(u.storage, VectorStorageKey(:Gradient))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    # old version:
    # inner(M, cgs.p, cgs.X, cgs.X) / inner(M, p_old, X_old, X_old)
    nominator = get_differential(amp, cgs.p, cgs.X; gradient = cgs.X, evaluated = true)
    denominator = get_differential(amp, p_old, X_old; gradient = X_old, evaluated = true)
    coeff = nominator / denominator
    update_storage!(u.storage, amp, cgs)
    return coeff
end
function show(io::IO, ::FletcherReevesCoefficientRule)
    return print(io, "Manopt.FletcherReevesCoefficientRule()")
end

@doc """
    FletcherReevesCoefficient()
    FletcherReevesCoefficient(M::AbstractManifold)

Computes an update coefficient for the [`conjugate_gradient_descent`](@ref) algorithm based on [FletcherReeves:1964](@cite) adapted to manifolds

$(_doc_CG_notaion)

Then the coefficient reads
```math
β_k = $(_tex(:frac, "$(_tex(:diff))f(p_{k+1})[X_{k+1}]", "$(_tex(:diff))f(p_k)[X_k]"))
 = $(_tex(:frac, _tex(:norm, "X_{k+1}"; index = "p_{k+1}") * "^2", _tex(:norm, "X_k"; index = "p_k") * "^2"))
```

The second one it the one usually stated, while the first one avoids to use the metric `inner`.
The first one is implemented here, but falls back to calling `inner` if there is no dedicated differential available.

$(_note(:ManifoldDefaultFactory, "FletcherReevesCoefficientRule"))
"""
function FletcherReevesCoefficient()
    return ManifoldDefaultsFactory(
        Manopt.FletcherReevesCoefficientRule; requires_manifold = false
    )
end

@doc """
    HagerZhangCoefficientRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient based on [HagerZhang:2005](@cite) adapted to manifolds

# Fields

$(_var(:Field, :vector_transport_method))

# Constructor

    HagerZhangCoefficientRule(M::AbstractManifold; kwargs...)

Construct the Hager-Zhang coefficient update rule based on [HagerZhang:2005](@cite) adapted to manifolds.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

# See also

[`HagerZhangCoefficient`](@ref), [`conjugate_gradient_descent`](@ref)
"""
mutable struct HagerZhangCoefficientRule{VTM <: AbstractVectorTransportMethod} <:
    DirectionUpdateRule
    vector_transport_method::VTM
end
function HagerZhangCoefficientRule(
        M::AbstractManifold; vector_transport_method::VTM = default_vector_transport_method(M)
    ) where {VTM <: AbstractVectorTransportMethod}
    return HagerZhangCoefficientRule{VTM}(vector_transport_method)
end

update_rule_storage_points(::HagerZhangCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::HagerZhangCoefficientRule) = Tuple{:Gradient, :δ}

function (u::DirectionUpdateRuleStorage{<:HagerZhangCoefficientRule})(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
    )
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
            !has_storage(u.storage, VectorStorageKey(:Gradient)) ||
            !has_storage(u.storage, VectorStorageKey(:δ))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    δ_old = get_storage(u.storage, VectorStorageKey(:δ))

    gradienttr = vector_transport_to(
        M, p_old, X_old, cgs.p, u.coefficient.vector_transport_method
    )
    ν = cgs.X - gradienttr #notation y from [HZ06]
    δtr = vector_transport_to(M, p_old, δ_old, cgs.p, u.coefficient.vector_transport_method)
    denom = inner(M, cgs.p, δtr, ν)
    νknormsq = inner(M, cgs.p, ν, ν)
    β =
        inner(M, cgs.p, ν, cgs.X) / denom -
        2 * νknormsq * inner(M, cgs.p, δtr, cgs.X) / denom^2
    # Numerical stability from Manopt / Hager-Zhang paper
    ξn = norm(M, cgs.p, cgs.X)
    η = -1 / (ξn * min(0.01, norm(M, p_old, X_old)))
    coef = max(β, η)
    update_storage!(u.storage, amp, cgs)
    return coef
end
function show(io::IO, u::HagerZhangCoefficientRule)
    return print(
        io,
        "Manopt.HagerZhangCoefficientRule(; vector_transport_method=$(u.vector_transport_method))",
    )
end

@doc """
    HagerZhangCoefficient(; kwargs...)
    HagerZhangCoefficient(M::AbstractManifold; kwargs...)

Computes an update coefficient for the [`conjugate_gradient_descent`](@ref) algorithm based on [FletcherReeves:1964](@cite) adapted to manifolds

$(_doc_CG_notaion)
Let ``ν_k = X_{k+1} - $(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))X_k``,
where ``$(_math(:vector_transport, :symbol))`` denotes a vector transport.

Then the coefficient reads
```math
β_k = $(_tex(:Bigl))⟨ν_k - $(
    _tex(
        :frac,
        "2$(_tex(:norm, "ν_k"; index = "p_{k+1}"))^2",
        "⟨$(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))δ_k, ν_k⟩_{p_{k+1}}",
    )
)
  $(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))δ_k,
  $(_tex(:frac, "X_{k+1}", "⟨$(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))δ_k, ν_k⟩_{p_{k+1}}"))
$(_tex(:Bigr))⟩_{p_{k+1}}.
```

This method includes a numerical stability proposed by those authors.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

$(_note(:ManifoldDefaultFactory, "HagerZhangCoefficientRule"))
"""
function HagerZhangCoefficient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.HagerZhangCoefficientRule, args...; kwargs...)
end

@doc """
    HestenesStiefelCoefficientRuleRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient based on [HestenesStiefel:1952](@cite) adapted to manifolds

# Fields

$(_var(:Field, :vector_transport_method))

# Constructor

    HestenesStiefelCoefficientRuleRule(M::AbstractManifold; kwargs...)

Construct the Hestenes-Stiefel coefficient update rule based on [HestenesStiefel:1952](@cite) adapted to manifolds.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

# See also

[`HestenesStiefelCoefficient`](@ref), [`conjugate_gradient_descent`](@ref)
"""
struct HestenesStiefelCoefficientRule{VTM <: AbstractVectorTransportMethod} <:
    DirectionUpdateRule
    vector_transport_method::VTM
end
function HestenesStiefelCoefficientRule(
        M::AbstractManifold; vector_transport_method::VTM = default_vector_transport_method(M)
    ) where {VTM <: AbstractVectorTransportMethod}
    return HestenesStiefelCoefficientRule{VTM}(vector_transport_method)
end

update_rule_storage_points(::HestenesStiefelCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::HestenesStiefelCoefficientRule) = Tuple{:Gradient, :δ}

function (u::DirectionUpdateRuleStorage{<:HestenesStiefelCoefficientRule})(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
    )
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
            !has_storage(u.storage, VectorStorageKey(:Gradient)) ||
            !has_storage(u.storage, VectorStorageKey(:δ))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    δ_old = get_storage(u.storage, VectorStorageKey(:δ))

    gradienttr = vector_transport_to(
        M, p_old, X_old, cgs.p, u.coefficient.vector_transport_method
    )
    δtr = vector_transport_to(M, p_old, δ_old, cgs.p, u.coefficient.vector_transport_method)
    ν = cgs.X - gradienttr #notation from [HZ06]
    # old with inners:
    # β = inner(M, cgs.p, cgs.X, ν) / inner(M, cgs.p, δtr, ν)
    nominator = get_differential(amp, cgs.p, ν; gradient = cgs.X, evaluated = true)
    denominator =
        get_differential(amp, cgs.p, δtr; gradient = cgs.X, evaluated = true) -
        get_differential(amp, p_old, δ_old; gradient = X_old, evaluated = true)
    β = nominator / denominator
    update_storage!(u.storage, amp, cgs)
    return max(0, β)
end
function show(io::IO, u::HestenesStiefelCoefficientRule)
    return print(
        io,
        "Manopt.HestenesStiefelCoefficientRule(; vector_transport_method=$(u.vector_transport_method))",
    )
end

"""
    HestenesStiefelCoefficient(; kwargs...)
    HestenesStiefelCoefficient(M::AbstractManifold; kwargs...)

Computes an update coefficient for the [`conjugate_gradient_descent`](@ref) algorithm based on [HestenesStiefel:1952](@cite) adapted to manifolds


$(_doc_CG_notaion)
Let ``ν_k = X_{k+1} - $(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))X_k``,
where ``$(_math(:vector_transport, :symbol))`` denotes a vector transport.

Then the coefficient reads

```math
\\begin{aligned}
β_k
&= $(
    _tex(
        :frac,
        "$(_tex(:diff))f(p_{k+1})[ν_k]",
        "$(_tex(:diff))f(p_{k+1})[$(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))δ_k] - $(_tex(:diff))f(p_k)[δ_k]",
    )
)
\\\\&= $(
    _tex(
        :frac,
        "$(_tex(:inner, "X_{k+1}", "ν_k"; index = "p_{k+1}"))",
        "$(_tex(:inner, "$(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))δ_k", "X_{k+1}"; index = "p_{k+1}")) - $(_tex(:inner, "δ_k", "X_k"; index = "p_{k}"))",
    )
)
\\\\&= $(
    _tex(
        :frac,
        "$(_tex(:inner, "X_{k+1}", "ν_k"; index = "p_{k+1}"))",
        "$(_tex(:inner, "$(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))δ_k", "ν_k"; index = "p_{k+1}"))",
    )
),
\\end{aligned}
```

The third one is the one usually stated, while the first one avoids to use the metric `inner`.
The first one is implemented here, but falls back to calling `inner` if there is no dedicated differential available.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

$(_note(:ManifoldDefaultFactory, "HestenesStiefelCoefficientRule"))
"""
function HestenesStiefelCoefficient(args...; kwargs...)
    return ManifoldDefaultsFactory(
        Manopt.HestenesStiefelCoefficientRule, args...; kwargs...
    )
end

@doc """
    LiuStoreyCoefficientRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient based on [LiuStorey:1991](@cite) adapted to manifolds

# Fields

$(_var(:Field, :vector_transport_method))

# Constructor

    LiuStoreyCoefficientRule(M::AbstractManifold; kwargs...)

Construct the Lui-Storey coefficient update rule based on [LiuStorey:1991](@cite) adapted to manifolds.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

# See also

[`LiuStoreyCoefficient`](@ref), [`conjugate_gradient_descent`](@ref)
"""
struct LiuStoreyCoefficientRule{VTM <: AbstractVectorTransportMethod} <: DirectionUpdateRule
    vector_transport_method::VTM
end
function LiuStoreyCoefficientRule(
        M::AbstractManifold; vector_transport_method::VTM = default_vector_transport_method(M)
    ) where {VTM <: AbstractVectorTransportMethod}
    return LiuStoreyCoefficientRule{VTM}(vector_transport_method)
end

update_rule_storage_points(::LiuStoreyCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::LiuStoreyCoefficientRule) = Tuple{:Gradient, :δ}

function (u::DirectionUpdateRuleStorage{<:LiuStoreyCoefficientRule})(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
    )
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
            !has_storage(u.storage, VectorStorageKey(:Gradient)) ||
            !has_storage(u.storage, VectorStorageKey(:δ))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    δ_old = get_storage(u.storage, VectorStorageKey(:δ))
    gradienttr = vector_transport_to(
        M, p_old, X_old, cgs.p, u.coefficient.vector_transport_method
    )
    ν = cgs.X - gradienttr # notation y from [HZ06]
    # old:
    # β = inner(M, cgs.p, cgs.X, ν) / inner(M, p_old, -δ_old, X_old)
    nominator = get_differential(amp, cgs.p, ν; gradient = cgs.X, evaluated = true)
    denominator = get_differential(amp, p_old, δ_old; gradient = X_old, evaluated = true)
    β = -nominator / denominator
    update_storage!(u.storage, amp, cgs)
    return β
end
function show(io::IO, u::LiuStoreyCoefficientRule)
    return print(
        io,
        "Manopt.LiuStoreyCoefficientRule(; vector_transport_method=$(u.vector_transport_method))",
    )
end

"""
    LiuStoreyCoefficient(; kwargs...)
    LiuStoreyCoefficient(M::AbstractManifold; kwargs...)

Computes an update coefficient for the [`conjugate_gradient_descent`](@ref) algorithm based on [LiuStorey:1991](@cite) adapted to manifolds

$(_doc_CG_notaion)
Let ``ν_k = X_{k+1} - $(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))X_k``,
where ``$(_math(:vector_transport, :symbol))`` denotes a vector transport.

Then the coefficient reads
```math
β_k
= - $(_tex(:frac, "$(_tex(:diff))f(p_{k+1})[ν_k]", "$(_tex(:diff))f(p_k)[δ_k]"))
= - $(_tex(:frac, "$(_tex(:inner, "X_{k+1}", "ν_k"; index = "p_{k+1}"))", "$(_tex(:inner, "δ_k", "X_k"; index = "p_k"))")).
```

The second one it the one usually stated, while the first one avoids to use the metric `inner`.
The first one is implemented here, but falls back to calling `inner` if there is no dedicated differential available.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

$(_note(:ManifoldDefaultFactory, "LiuStoreyCoefficientRule"))
"""
function LiuStoreyCoefficient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.LiuStoreyCoefficientRule, args...; kwargs...)
end

@doc """
    PolakRibiereCoefficientRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient based on [PolakRibiere:1969](@cite) adapted to manifolds

# Fields

$(_var(:Field, :vector_transport_method))

# Constructor

    PolakRibiereCoefficientRule(M::AbstractManifold; kwargs...)

Construct the Dai—Yuan coefficient update rule.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

# See also
[`PolakRibiereCoefficient`](@ref), [`conjugate_gradient_descent`](@ref)
"""
struct PolakRibiereCoefficientRule{VTM <: AbstractVectorTransportMethod} <:
    DirectionUpdateRule
    vector_transport_method::VTM
end
function PolakRibiereCoefficientRule(
        M::AbstractManifold; vector_transport_method::VTM = default_vector_transport_method(M)
    ) where {VTM <: AbstractVectorTransportMethod}
    return PolakRibiereCoefficientRule{VTM}(vector_transport_method)
end

update_rule_storage_points(::PolakRibiereCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::PolakRibiereCoefficientRule) = Tuple{:Gradient}

function (u::DirectionUpdateRuleStorage{<:PolakRibiereCoefficientRule})(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
    )
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
            !has_storage(u.storage, VectorStorageKey(:Gradient))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))

    gradienttr = vector_transport_to(
        M, p_old, X_old, cgs.p, u.coefficient.vector_transport_method
    )
    ν = cgs.X - gradienttr
    # old
    # β = real(inner(M, cgs.p, cgs.X, ν)) / real(inner(M, p_old, X_old, X_old))
    nominator = get_differential(amp, cgs.p, ν; gradient = cgs.X, evaluated = true)
    denominator = get_differential(amp, p_old, X_old; gradient = X_old, evaluated = true)
    β = nominator / denominator
    # numerical stability from Manopt
    update_storage!(u.storage, amp, cgs)
    return max(zero(β), β)
end
function show(io::IO, u::PolakRibiereCoefficientRule)
    return print(
        io,
        "Manopt.PolakRibiereCoefficientRule(; vector_transport_method=$(u.vector_transport_method))",
    )
end

"""
    PolakRibiereCoefficient(; kwargs...)
    PolakRibiereCoefficient(M::AbstractManifold; kwargs...)

Computes an update coefficient for the [`conjugate_gradient_descent`](@ref) algorithm based
on [PolakRibiere:1969](@cite) adapted to Riemannian manifolds.

$(_doc_CG_notaion)
Let ``ν_k = X_{k+1} - $(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))X_k``,
where ``$(_math(:vector_transport, :symbol))`` denotes a vector transport.

Then the coefficient reads

````math
β_k
= $(_tex(:frac, "$(_tex(:diff))f(p_{k+1})[ν_k]", "$(_tex(:diff))f(p_k)[X_k]"))
= $(_tex(:frac, _tex(:inner, "X_{k+1}", "ν_k"; index = "p_{k+1}"), _tex(:norm, "X_k"; index = "{p_k}") * "^2")).
````

The second one is the one usually stated, while the first one avoids to use the metric `inner`.
The first one is implemented here, but falls back to calling `inner` if there is no dedicated differential available.

# Keyword arguments

$(_var(:Keyword, :vector_transport_method))

$(_note(:ManifoldDefaultFactory, "PolakRibiereCoefficientRule"))
"""
function PolakRibiereCoefficient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.PolakRibiereCoefficientRule, args...; kwargs...)
end

@doc """
    SteepestDescentCoefficientRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient
to obtain the steepest direction, that is ``β_k=0``.

# Constructor

    SteepestDescentCoefficientRule()

Construct the steepest descent coefficient update rule.

# See also
[`SteepestDescentCoefficient`](@ref), [`conjugate_gradient_descent`](@ref)
"""
struct SteepestDescentCoefficientRule <: DirectionUpdateRule end

update_rule_storage_points(::SteepestDescentCoefficientRule) = Tuple{}
update_rule_storage_vectors(::SteepestDescentCoefficientRule) = Tuple{}

function (u::DirectionUpdateRuleStorage{SteepestDescentCoefficientRule})(
        ::DefaultManoptProblem, ::ConjugateGradientDescentState, i
    )
    return 0.0
end
@doc """
    SteepestDescentCoefficient()
    SteepestDescentCoefficient(M::AbstractManifold)

Computes an update coefficient for the [`conjugate_gradient_descent`](@ref) algorithm
so that is falls back to a [`gradient_descent`](@ref) method, that is
````math
β_k = 0
````

$(_note(:ManifoldDefaultFactory, "SteepestDescentCoefficient"))
"""
function SteepestDescentCoefficient()
    return ManifoldDefaultsFactory(
        Manopt.SteepestDescentCoefficientRule; requires_manifold = false
    )
end

@doc """
    ConjugateGradientBealeRestartRule <: DirectionUpdateRule

A functor `(problem, state, k) -> β_k` to compute the conjugate gradient update coefficient
based on a restart idea of [Beale:1972](@cite), following [HagerZhang:2006; page 12](@cite)
adapted to manifolds.

# Fields

* `direction_update::DirectionUpdateRule`: the actual rule, that is restarted
* `threshold::Real`: a threshold for the restart check.
$(_var(:Field, :vector_transport_method))

# Constructor

    ConjugateGradientBealeRestartRule(
        direction_update::Union{DirectionUpdateRule,ManifoldDefaultsFactory};
        kwargs...
    )
    ConjugateGradientBealeRestartRule(
        M::AbstractManifold=DefaultManifold(),
        direction_update::Union{DirectionUpdateRule,ManifoldDefaultsFactory};
        kwargs...
    )

Construct the Beale restart coefficient update rule adapted to manifolds.

## Input

$(_var(:Argument, :M; type = true))
  If this is not provided, the `DefaultManifold()` from $(_link(:ManifoldsBase)) is used.
* `direction_update`: a [`DirectionUpdateRule`](@ref) or a corresponding
  [`ManifoldDefaultsFactory`](@ref) to produce such a rule.

## Keyword arguments

$(_var(:Keyword, :vector_transport_method))
* `threshold=0.2`

# See also

[`ConjugateGradientBealeRestart`](@ref), [`conjugate_gradient_descent`](@ref)
"""
mutable struct ConjugateGradientBealeRestartRule{
        DUR <: DirectionUpdateRule, VT <: AbstractVectorTransportMethod, F <: Real,
    } <: DirectionUpdateRule
    direction_update::DUR
    threshold::F
    vector_transport_method::VT
end
function ConjugateGradientBealeRestartRule(
        M::AbstractManifold,
        direction_update::Union{DirectionUpdateRule, ManifoldDefaultsFactory};
        threshold::F = 0.2,
        vector_transport_method::V = default_vector_transport_method(M),
    ) where {V <: AbstractVectorTransportMethod, F <: Real}
    dir = _produce_type(direction_update, M)
    return ConjugateGradientBealeRestartRule{typeof(dir), V, F}(
        dir, threshold, vector_transport_method
    )
end
function ConjugateGradientBealeRestartRule(
        direction_update::Union{DirectionUpdateRule, ManifoldDefaultsFactory}; kwargs...
    )
    return ConjugateGradientBealeRestartRule(DefaultManifold(), direction_update; kwargs...)
end

@inline function update_rule_storage_points(dur::ConjugateGradientBealeRestartRule)
    dur_p = update_rule_storage_points(dur.direction_update)
    return :Iterate in dur_p.parameters ? dur_p : Tuple{:Iterate, dur_p.parameters...}
end
@inline function update_rule_storage_vectors(dur::ConjugateGradientBealeRestartRule)
    dur_X = update_rule_storage_vectors(dur.direction_update)
    return :Gradient in dur_X.parameters ? dur_X : Tuple{:Gradient, dur_X.parameters...}
end

function (u::DirectionUpdateRuleStorage{<:ConjugateGradientBealeRestartRule})(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, k
    )
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
            !has_storage(u.storage, VectorStorageKey(:Gradient))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))

    # call actual rule
    β = u.coefficient.direction_update(amp, cgs, k)

    denom = norm(M, cgs.p, cgs.X)
    Xoldpk = vector_transport_to(
        M, p_old, X_old, cgs.p, u.coefficient.vector_transport_method
    )
    num = inner(M, cgs.p, cgs.X, Xoldpk)
    # update storage only after that in case they share
    update_storage!(u.storage, amp, cgs)
    return real(num / denom) > u.coefficient.threshold ? zero(β) : β
end
function show(io::IO, u::ConjugateGradientBealeRestartRule)
    return print(
        io,
        "Manopt.ConjugateGradientBealeRestartRule($(repr(u.direction_update)); threshold=$(u.threshold), vector_transport_method=$(u.vector_transport_method))",
    )
end

"""
    ConjugateGradientBealeRestart(direction_update::Union{DirectionUpdateRule,ManifoldDefaultsFactory}; kwargs...)
    ConjugateGradientBealeRestart(M::AbstractManifold, direction_update::Union{DirectionUpdateRule,ManifoldDefaultsFactory}; kwargs...)

Compute a conjugate gradient coefficient with a potential restart, when two directions are
nearly orthogonal. See [HagerZhang:2006; page 12](@cite) (in the preprint, page 46 in Journal page numbers).
This method is named after E. Beale from his proceedings paper in 1972 [Beale:1972](@cite).
This method acts as a _decorator_ to any existing [`DirectionUpdateRule`](@ref) `direction_update`.

$(_doc_CG_notaion)

Then a restart is performed, hence ``β_k = 0`` returned if

```math
  $(
    _tex(
        :frac,
        "⟨X_{k+1}, $(_math(:vector_transport, :symbol, "p_{k+1}", "p_k"))X_k⟩",
        _tex(:norm, "X_k", index = "p_k")
    )
) > ε,
```
where ``ε`` is the `threshold`, which is set by default to `0.2`, see [Powell:1977](@cite)

## Input

* `direction_update`: a [`DirectionUpdateRule`](@ref) or a corresponding
  [`ManifoldDefaultsFactory`](@ref) to produce such a rule.

## Keyword arguments

$(_var(:Keyword, :vector_transport_method))
* `threshold=0.2`

$(_note(:ManifoldDefaultFactory, "ConjugateGradientBealeRestartRule"))
"""
function ConjugateGradientBealeRestart(args...; kwargs...)
    return ManifoldDefaultsFactory(
        Manopt.ConjugateGradientBealeRestartRule, args...; kwargs...
    )
end

@doc """
    NeverRestart <: AbstractRestartCondition

A restart strategy that indicates to never restart.
"""
struct NeverRestart <: AbstractRestartCondition end

function (corr::NeverRestart)(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, k
    )
    return false
end

@doc """
    RestartOnNonDescent <: AbstractRestartCondition

A restart strategy that restarts, whenever the search direction `δ` is not a descent direction,
i.e. when

```math
    ⟨$(_tex(:grad))f(p), δ⟩ > 0,
```

at the current iterate ``p``.
"""
struct RestartOnNonDescent <: AbstractRestartCondition end
function (corr::RestartOnNonDescent)(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, k
    )
    return get_differential(amp, cgs.p, cgs.δ; gradient = cgs.X, evaluated = true) >= 0
end

@doc """
RestartOnNonSufficientDescent <: AbstractRestartCondition

## Fields
* `κ`: the sufficient decrease factor

A restart strategy that indicates to restart whenever the search direction `δ` is not a sufficient descent direction, i.e.
```math
    ⟨$(_tex(:grad))f(p), δ⟩ ≤ - κ $(_tex(:norm, "X"))^2.
```

at the current iterate ``p``.
"""
struct RestartOnNonSufficientDescent{F <: Real} <: AbstractRestartCondition
    κ::F
end
function (corr::RestartOnNonSufficientDescent)(
        amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, k
    )
    return (
        get_differential(amp, cgs.p, cgs.δ; gradient = cgs.X, evaluated = true) >
            -corr.κ * get_differential(amp, cgs.p, cgs.X; gradient = cgs.X, evaluated = true)
    )
end
