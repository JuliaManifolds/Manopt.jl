function reflect end
@doc """
    reflect(M, f, x; kwargs...)
    reflect!(M, q, f, x; kwargs...)

reflect the point `x` from the manifold `M` at the point `f(x)` of the
function ``f: $(_math(:Manifold)) → $(_math(:Manifold))``, given by

````math
    $(_tex(:reflect))_f(x) = $(_tex(:reflect))_{f(x)}(x),
````

Compute the result in `q`.

see also [`reflect`](@ref reflect(M::AbstractManifold, p, x))`(M,p,x)`, to which the keywords are also passed to.
"""
reflect(M::AbstractManifold, pr::Function, x; kwargs...)

@doc """
    reflect(M, p, x, kwargs...)
    reflect!(M, q, p, x, kwargs...)

Reflect the point `x` from the manifold `M` at point `p`, given by

```math
$(_tex(:reflect))_p(q) = $(_tex(:retr))_p(-$(_tex(:invretr))_p q),
```
where ``$(_tex(:retr))`` and ``$(_tex(:invretr))`` denote a retraction and an inverse retraction, respectively.

This can also be done in place of `q`.

## Keyword Arguments

$(_kwargs([:retraction_method, :inverse_retraction_method]))

and for the `reflect!` additionally

* `X=zero_vector(M,p)`: a temporary memory to compute the inverse retraction in place.
  otherwise this is the memory that would be allocated anyways.
"""
reflect(M::AbstractManifold, p::Any, x; kwargs...)

@doc """
    DouglasRachfordState <: AbstractManoptSolverState

Store all options required for the DouglasRachford algorithm,

# Fields

* `α`:                         relaxation of the step from old to new iterate, to be precise
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result of the double
  reflection involved in the DR algorithm
$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:inverse_retraction_method))
* `λ`:                         function to provide the value for the proximal parameter during the calls
* `parallel`:                  indicate whether to use a parallel Douglas-Rachford or not.
* `R!`:                          method employed in the iteration to perform the reflection of `x` at the prox `p`.
$(_fields(:p; add_properties = [:as_Iterate]))
  For the parallel Douglas-Rachford, this is not a value from the `PowerManifold` manifold but the mean.
* `reflection!`:     whether `R` works in-place or allocating
$(_fields(:retraction_method))
* `s`:                         the last result of the double reflection at the proximal maps relaxed by `α`.
$(_fields(:stopping_criterion; name = "stop"))

# Constructor

    DouglasRachfordState(M::AbstractManifold; kwargs...)

# Input

$(_args(:M))

# Keyword arguments

* `α= k -> 0.9`: relaxation of the step from old to new iterate, to be precise
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result of the double reflection involved in the DR algorithm
$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
$(_kwargs(:inverse_retraction_method))
* `λ= k -> 1.0`: function to provide the value for the proximal parameter
  during the calls
$(_kwargs(:p; add_properties = [:as_Initial]))
* `R!= Manopt.`[`reflect!`](@ref)`(!)`: method employed in the iteration to perform the reflection of `p` at
  the prox of `p`, which always works in-place.
* `reflection_evaluation=`[`InplaceEvaluation`](@ref)`()`) specify whether the reflection works in-place (default) or allocating
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(300)"))
* `parallel=false`: indicate whether to use a parallel Douglas-Rachford or not.
"""
mutable struct DouglasRachfordState{
        P, C <: AbstractDict{Symbol}, Tλ, Tα, TR, S, TM <: AbstractRetractionMethod, ITM <: AbstractInverseRetractionMethod,
    } <: AbstractManoptSolverState
    α::Tα
    callbacks::C
    inverse_retraction_method::ITM
    λ::Tλ
    p::P
    parallel::Bool
    p_tmp::P
    R!::TR
    retraction_method::TM
    s::P
    s_tmp::P
    stop::S
    function DouglasRachfordState(
            M::AbstractManifold; p::P = rand(M), λ::Fλ = i -> 1.0, α::Fα = i -> 0.9,
            callbacks::C = Dict{Symbol, Function}(),
            reflection_evaluation::E = InplaceEvaluation(),
            R!::FR = Manopt.reflect!,
            stopping_criterion::S = StopAfterIteration(300),
            parallel = false,
            retraction_method::TM = default_retraction_method(M, typeof(p)),
            inverse_retraction_method::ITM = default_inverse_retraction_method(M, typeof(p)),
        ) where {
            P, C <: AbstractDict{Symbol}, Fλ, Fα, FR, S <: StoppingCriterion, E <: AbstractEvaluationType,
            TM <: AbstractRetractionMethod, ITM <: AbstractInverseRetractionMethod,
        }
        R! = maybe_wrap_function(R!, p, reflection_evaluation; result = :Point)
        return DouglasRachfordState(;
            p = p, p_tmp = copy(M, p), s = copy(M, p), s_tmp = copy(M, p),
            λ = λ, α = α, (R!) = R!, callbacks = callbacks,
            retraction_method = retraction_method, inverse_retraction_method = inverse_retraction_method,
            stopping_criterion = stopping_criterion, parallel = parallel,
        )
    end
    function DouglasRachfordState(;
            p::P, p_tmp::P, s::P, s_tmp::P, λ::Fλ, α::Fα, R!::FR,
            callbacks::C, retraction_method::TM, inverse_retraction_method::ITM,
            stopping_criterion::S, parallel::Bool
        ) where {
            P, C <: AbstractDict{Symbol}, Fλ, Fα, FR, S <: StoppingCriterion,
            TM <: AbstractRetractionMethod, ITM <: AbstractInverseRetractionMethod,
        }
        return new{P, C, Fλ, Fα, FR, S, TM, ITM}(
            α, callbacks, inverse_retraction_method, λ, p, parallel, p_tmp,
            R!, retraction_method, s, s_tmp, stopping_criterion,
        )
    end
end
provided_callbacks(::Type{DouglasRachfordState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:FirstReflection, :ProximalMap, :SecondReflection])
get_callbacks(drs::DouglasRachfordState) = drs.callbacks
function Base.show(io::IO, drs::DouglasRachfordState)
    print(io, "DouglasRachfordState(; ")
    print(io, "α = "); print(io, drs.α); print(io, ", ")
    print(io, "callbacks = "); print(io, drs.callbacks); print(io, ", ")
    print(io, "inverse_retraction_method = "); print(io, drs.inverse_retraction_method); print(io, ", ")
    print(io, "λ = "); print(io, drs.λ); print(io, ", ")
    print(io, "p = "); print(io, drs.p); print(io, ", ")
    print(io, "parallel = "); print(io, drs.parallel); print(io, ", ")
    print(io, "p_tmp = "); print(io, drs.p_tmp); print(io, ", ")
    print(io, "(R!) = "); print(io, drs.R!); print(io, ", ")
    print(io, "retraction_method = "); print(io, drs.retraction_method); print(io, ", ")
    print(io, "s = "); print(io, drs.s); print(io, ", ")
    print(io, "s_tmp = "); print(io, drs.s_tmp); print(io, ", ")
    print(io, "stopping_criterion = "); print(io, status_summary(drs.stop; context = :short))
    return print(io, ")")
end
function status_summary(drs::DouglasRachfordState; context::Symbol = :default)
    (context === :short) && return repr(drs)
    i = get_count(drs, :Iterations)
    conv_inl = (i > 0) ? (indicates_convergence(drs.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the Douglas Rachford solver$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(drs.stop) ? "Yes" : "No"
    _is_inline(context) && (return "$(repr(drs)) – $(Iter) $(has_converged(drs) ? "(converged)" : "")")
    as = _callbacks_summary(drs)
    P = drs.parallel ? "Parallel " : ""
    s = """
    # Solver state for `Manopt.jl`s $(P)Douglas Rachford Algorithm
    $Iter

    ## Parameters$(as)
    * `R! = ` $(drs.R!)

    ## Stopping criterion
    $(_in_str(status_summary(drs.stop; context = context); indent = 0, headers = 1))
    This indicates convergence: $Conv"""
    return s
end
get_iterate(drs::DouglasRachfordState) = drs.p
function set_iterate!(drs::DouglasRachfordState, p)
    drs.p = p
    return drs
end

function (d::DebugProximalParameter)(
        ::AbstractManoptProblem, cpps::DouglasRachfordState, k::Int
    )
    (k >= (d.at_init ? 0 : 1)) && Printf.format(d.io, Printf.Format(d.format), cpps.λ(k))
    return nothing
end
function (r::RecordProximalParameter)(
        ::AbstractManoptProblem, cpps::DouglasRachfordState, k::Int
    )
    return record_or_reset!(r, cpps.λ(k), k)
end
_doc_Douglas_Rachford = """
    DouglasRachford(M, f, proxes_f, p)
    DouglasRachford(M, mpo, p)
    DouglasRachford!(M, f, proxes_f, p)
    DouglasRachford!(M, mpo, p)

Compute the Douglas-Rachford algorithm on the manifold ``$(_math(:Manifold))``, starting from `p`
given the (two) proximal maps `proxes_f`, see [BergmannPerschSteidl:2016](@cite).

For ``k>2`` proximal maps, the problem is reformulated using the parallel Douglas Rachford:
a vectorial proximal map on the power manifold ``$(_math(:Manifold))^k`` is introduced as the first
proximal map and the second proximal map of the is set to the [`mean`](@extref Statistics.mean-Tuple{AbstractManifold, Vararg{Any}}) (Riemannian center of mass).
This hence also boils down to two proximal maps, though each evaluates proximal maps in parallel,
that is, component wise in a vector.

!!! note
    The parallel Douglas Rachford does not work in-place for now, since
    while creating the new staring point `p'` on the power manifold, a copy of `p`
    Is created

If you provide a [`ManifoldProximalMapObjective`](@ref) `mpo` instead, the proximal maps are kept unchanged.

# Input

$(_args([:M, :f]))
* `proxes_f`: functions of the form `(M, λ, p)-> q` performing a proximal maps,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `F`.
  These can also be given in the [`InplaceEvaluation`](@ref) variants `(M, q, λ p) -> q`
  computing in place of `q`.
$(_args(:p))

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
* `α= k -> 0.9`: relaxation of the step from old to new iterate, to be precise
  ``p^{(k+1)} = g(α_k; p^{(k)}, q^{(k)})``, where ``q^{(k)}`` is the result of the double reflection
  involved in the DR algorithm and ``g`` is a curve induced by the retraction and its inverse.
$(_kwargs([:evaluation, :inverse_retraction_method]))
  This is used both in the relaxation step as well as in the reflection, unless you set `R` yourself.
* `λ= k -> 1.0`: function to provide the value for the proximal parameter ``λ_k``
* `R=reflect!`: method employed in the iteration to perform the reflection of `p` at the prox of `p`.
  This uses by default [`reflect`](@ref) or `reflect!` depending on `reflection_evaluation` and
  the retraction and inverse retraction specified by `retraction_method` and `inverse_retraction_method`, respectively.
* `reflection_evaluation`: ([`AllocatingEvaluation`](@ref) whether `R` works in-place or allocating
$(_kwargs(:retraction_method))
  This is used both in the relaxation step as well as in the reflection, unless you set `R` yourself.
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-5)"))
* `parallel=false`: indicate whether to use a parallel Douglas-Rachford or not.

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_Douglas_Rachford)"
DouglasRachford(::AbstractManifold, args...; kwargs...)
function DouglasRachford(
        M::AbstractManifold, f::TF, proxes_f::Vector{<:Any}, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), parallel = 0, kwargs...,
    ) where {TF}
    p_ = maybe_wrap_variable(p)
    proxes_f_ = [maybe_wrap_function(prox_f, p, evaluation; result = :Point) for prox_f in proxes_f]
    N, f__, (prox1, prox2), parallel_, q = parallel_to_alternating_DR(M, f, proxes_f_, p_, parallel)
    # we are inplace, so no need to pass it further down here
    mpo = ManifoldProximalMapObjective(f__, (prox1, prox2); evaluation = InplaceEvaluation())
    rs = DouglasRachford(N, mpo, q; evaluation = evaluation, parallel = parallel_, kwargs...)
    return maybe_unwrap_variable(p, rs)
end
function DouglasRachford(
        M::AbstractManifold, mpo::O, p; kwargs...
    ) where {O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(DouglasRachford; kwargs...)
    p_ = maybe_wrap_variable(p)
    q = copy(M, p_)
    rs = DouglasRachford!(M, mpo, q; kwargs...)
    return maybe_unwrap_variable(p, rs)
end
calls_with_kwargs(::typeof(DouglasRachford)) = (DouglasRachford!,)

@doc "$(_doc_Douglas_Rachford)"
DouglasRachford!(::AbstractManifold, args...; kwargs...)
function DouglasRachford!(
        M::AbstractManifold, f::TF, proxes_f::Vector{<:Any}, p;
        evaluation = AllocatingEvaluation(), parallel::Int = 0, kwargs...,
    ) where {TF}
    proxes_f_ = [maybe_wrap_function(prox_f, p, evaluation; result = :Point) for prox_f in proxes_f]
    N, f_, (prox1, prox2), parallel_, p0 = parallel_to_alternating_DR(M, f, proxes_f_, p, parallel)
    # we are inplace, so no need to pass it further down here
    mpo = ManifoldProximalMapObjective(f_, (prox1, prox2); evaluation = InplaceEvaluation())
    return DouglasRachford!(
        N, mpo, p0; evaluation = evaluation, parallel = parallel_, kwargs...
    )
end
function DouglasRachford!(
        M::AbstractManifold, mpo::O, p;
        callbacks = Dict{Symbol, Function}(),
        λ::Tλ = (iter) -> 1.0,
        α::Tα = (iter) -> 0.9,
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        inverse_retraction_method::AbstractInverseRetractionMethod = default_inverse_retraction_method(
            M, typeof(p)
        ),
        reflection_evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        # Adapt to evaluation type
        R::TR = if reflection_evaluation == InplaceEvaluation()
            (M, r, p, q) -> Manopt.reflect!(
                M, r, p, q;
                retraction_method = retraction_method,
                inverse_retraction_method = inverse_retraction_method,
            )
        else
            (M, p, q) -> Manopt.reflect(
                M, p, q;
                retraction_method = retraction_method,
                inverse_retraction_method = inverse_retraction_method,
            )
        end,
        parallel::Int = 0,
        stopping_criterion::StoppingCriterion = StopAfterIteration(200) |
            StopWhenChangeLess(M, 1.0e-5),
        kwargs..., #especially may contain decorator options
    ) where {
        Tλ, Tα, TR,
        O <: Union{ManifoldProximalMapObjective, AbstractDecoratedManifoldObjective},
    }
    keywords_accepted(DouglasRachford!; kwargs...)
    dmpo = decorate_objective!(M, mpo; kwargs...)
    dmp = DefaultManoptProblem(M, dmpo)
    drs = DouglasRachfordState(
        M;
        callbacks = process_callbacks_arg(callbacks, DouglasRachfordState),
        p = p, λ = λ, α = α, (R!) = R,
        reflection_evaluation = reflection_evaluation,
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
        stopping_criterion = stopping_criterion,
        parallel = parallel > 0,
    )
    ddrs = decorate_state!(drs; kwargs...)
    solve!(dmp, ddrs)
    return get_solver_return(get_objective(dmp), ddrs)
end
calls_with_kwargs(::typeof(DouglasRachford!)) = (decorate_objective!, decorate_state!)

#
# An internal function that turns more than 2 proximal maps into a parallel variant
# on the power manifold
function parallel_to_alternating_DR(
        M, f, proxes_f, p, parallel
    )
    prox1, prox2, parallel_ = prepare_proxes(proxes_f, parallel)
    if parallel_ > 0
        N = PowerManifold(M, NestedPowerRepresentation(), parallel_)
        p0 = [p]
        for _ in 2:parallel_
            push!(p0, copy(M, p))
        end
        f_ = (M, p) -> f(M.manifold, p[1])
    else
        N = M
        f_ = f
        p0 = p
    end
    return N, f_, (prox1, prox2), parallel_, p0
end #
# An internal function that turns more than 2 proximal maps into a parallel variant
function prepare_proxes(proxes_f, parallel)
    parallel_ = parallel
    if length(proxes_f) < 2
        throw(
            ErrorException(
                "Less than two proximal maps provided, the (parallel) Douglas Rachford requires (at least) two proximal maps.",
            ),
        )
    elseif length(proxes_f) == 2
        prox1 = proxes_f[1]
        prox2 = proxes_f[2]
    else # more than 2 -> parallelDouglasRachford
        parallel_ = length(proxes_f)
        prox1 = function (M, q, λ, p)
            [proxes_f[i](M.manifold, q[i], λ, p[i]) for i in 1:parallel_]
            return q
        end
        prox2 = (M, q, λ, p) -> fill!(q, mean(M.manifold, p))
    end
    return prox1, prox2, parallel_
end
function initialize_solver!(::AbstractManoptProblem, ::DouglasRachfordState) end
function step_solver!(amp::AbstractManoptProblem, drs::DouglasRachfordState, k)
    M = get_manifold(amp)
    get_proximal_map!(amp, drs.p_tmp, drs.λ(k), drs.s, 1)
    #dispatch on allocation type for the reflection, see below.
    drs.R!(M, drs.s_tmp, drs.p_tmp, drs.s)
    callback(:FirstReflection, amp, drs, k)
    get_proximal_map!(amp, drs.p, drs.λ(k), drs.s_tmp, 2)
    callback(:ProximalMap, amp, drs, k)
    drs.R!(M, drs.s_tmp, drs.p, drs.s_tmp)
    callback(:SecondReflection, amp, drs, k)
    # relaxation
    drs.s = ManifoldsBase.retract_fused(
        M, drs.s,
        inverse_retract(M, drs.s, drs.s_tmp, drs.inverse_retraction_method),
        drs.α(k), drs.retraction_method,
    )
    return drs
end
get_solver_result(drs::DouglasRachfordState) = drs.parallel ? drs.p[1] : drs.p
