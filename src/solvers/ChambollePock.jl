#
#
# ---
@doc """
    PrimalDualManifoldObjective <: AbstractPrimalDualManifoldObjective

Describes an Objective linearized or exact Chambolle-Pock algorithm, cf. [BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021](@cite), [ChambollePock:2011](@cite)

# Fields

* `cost`:                          ``F + G(Λ(⋅))`` to evaluate interim cost function values
* `linearized_forward_operator!`: linearized operator for the forward operation in the algorithm ``DΛ``
* `adjoint_linearized_operator!`: the adjoint differential ``(DΛ)^* : $(_math(:Manifold; M = "N")) → T$(_math(:Manifold))``
* `prox_f!`:                      the proximal map belonging to ``f``
* `prox_g_dual!`:                 the proximal map belonging to ``g_n^*``
* `Λ!`:                           the  forward operator (if given) ``Λ: $(_math(:Manifold)) → $(_math(:Manifold; M = "N"))``

Either the linearized operator ``DΛ`` or ``Λ`` are required usually.

# Constructor

    PrimalDualManifoldObjective(cost, prox_f, prox_G_dual, adjoint_linearized_operator;
        linearized_forward_operator::Union{Function,Missing}=missing,
        Λ::Union{Function,Missing}=missing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    )

Using the `evaluation=` keyword can be used to specify that all functions work in-place instead
of the default allocating one.
"""
mutable struct PrimalDualManifoldObjective{
        TC, TP, TDP, LFO, ALFO, L,
    } <: AbstractPrimalDualManifoldObjective{TC, TP}
    cost::TC
    prox_f!::TP
    prox_g_dual!::TDP
    linearized_forward_operator!::LFO
    adjoint_linearized_operator!::ALFO
    Λ!::L
end
function PrimalDualManifoldObjective(
        cost, prox_f, prox_g_dual, adjoint_linearized_operator;
        linearized_forward_operator::Union{Function, Missing} = missing,
        Λ::Union{Function, Missing} = missing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    )
    cost_ = maybe_wrap_function(cost, evaluation; result = :Number)
    prox_f_ = maybe_wrap_function(prox_f, evaluation; result = :Point)
    prox_g_dual_ = maybe_wrap_function(prox_g_dual, evaluation; result = :TangentVector)
    linearized_forward_operator_ = maybe_wrap_function(linearized_forward_operator, evaluation; result = :SecondManifoldPoint)
    adjoint_linearized_operator_ = maybe_wrap_function(adjoint_linearized_operator, evaluation; result = :Vector)
    Λ_ = ismissing(Λ) ? missing : maybe_wrap_function(Λ, evaluation; result = :SecondManifoldPoint)
    return PrimalDualManifoldObjective{
        typeof(cost_), typeof(prox_f_), typeof(prox_g_dual_), typeof(linearized_forward_operator_), typeof(adjoint_linearized_operator_), typeof(Λ_),
    }(
        cost_, prox_f_, prox_g_dual_, linearized_forward_operator_, adjoint_linearized_operator_, Λ_,
    )
end

function status_summary(pdmo::PrimalDualManifoldObjective; context::Symbol = :default)
    both_missing = ismissing(pdmo.Λ!) && ismissing(pdmo.linearized_forward_operator!)
    _is_inline(context) && ("A primal dual objective with a cost of f+g, a prox for f, a prox for the dual of g, as well as $(!ismissing(pdmo.Λ!) ? "an operator Λ," : "") $(!ismissing(pdmo.linearized_forward_operator!) ? "DΛ, " : "")$(!both_missing ? "and " : "")an adjoint D^*Λ")

    maybe_line1 = ismissing(pdmo.Λ!) ? "" : "\n* Λ:       $(pdmo.Λ!)"
    maybe_line2 = ismissing(pdmo.linearized_forward_operator!) ? "" : "\n* DΛ:      $(pdmo.linearized_forward_operator!)"
    return """
    A primal dual objective with

    * cost:    $(pdmo.cost)
    * prox_f:  $(pdmo.prox_f!)
    * prox_g*: $(pdmo.prox_g_dual!)
    * D^*Λ:    $(pdmo.adjoint_linearized_operator!)$(maybe_line1)$(maybe_line2)"""
end
function show(io::IO, pdmo::PrimalDualManifoldObjective)
    print(io, "PrimalDualManifoldObjective(", pdmo.cost, ", ", pdmo.prox_f!, ", ")
    print(io, pdmo.prox_g_dual!, ", ", pdmo.adjoint_linearized_operator!, "; ")
    !ismissing(pdmo.Λ!) && (print(io, ", Λ = ", pdmo.Λ!))
    !ismissing(pdmo.linearized_forward_operator!) && (print(io, ", linearized_forward_operator = ", pdmo.linearized_forward_operator!))
    return print(io, ")")
end

@doc """
    ChambollePockState <: AbstractPrimalDualSolverState

stores all options and variables within a linearized or exact Chambolle Pock.

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
* `acceleration::R`:    acceleration factor
* `dual_stepsize::R`:   proximal parameter of the dual prox
$(_fields(:inverse_retraction_method))
$(_fields(:inverse_retraction_method; name = "inverse_retraction_method_dual", M = "N", p = "n"))
* `m::P`:               base point on ``$(_math(:Manifold))``
* `n::Q`:               base point on ``$(_math(:Manifold; M = "N"))``
* `p::P`:               an initial point on ``p^{(0)} ∈ $(_math(:Manifold))``
* `pbar::P`:            the relaxed iterate used in the next dual update step (when using `:primal` relaxation)
* `primal_stepsize::R`: proximal parameter of the primal prox
* `X::T`:               an initial tangent vector ``X^{(0)} ∈ $(_math(:TangentSpace; p = "p^{(0)}"))``
* `Xbar::T`:            the relaxed iterate used in the next primal update step (when using `:dual` relaxation)
* `relaxation::R`:      relaxation in the primal relaxation step (to compute `pbar`:
* `relax::Symbol`:      which variable to relax, `:primal` or `:dual`
$(_fields(:retraction_method))
$(_fields(:stopping_criterion; name = "stop"))
* `variant`:            whether to perform an `:exact` or `:linearized` Chambolle-Pock
* `update_primal_base`: function `(pr, st, k) -> m` to update the primal base
* `update_dual_base`:  function `(pr, st, k) -> n` to update the dual base
$(_fields(:vector_transport_method))
$(_fields(:vector_transport_method; name = "vector_transport_method_dual", M = "N"))

Here, `P` is a point type on ``$(_math(:Manifold))``, `T` its tangent vector type, `Q` a point type on ``$(_math(:Manifold; M = "N"))``,
and `R<:Real` is a real number type

where for the last two the functions a [`AbstractManoptProblem`](@ref)` p`,
[`AbstractManoptSolverState`](@ref)` o` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing` in the linearized case).

# Constructor

    ChambollePockState(M::AbstractManifold, N::AbstractManifold;
        kwargs...
    ) where {P, Q, T, R <: Real}

# Keyword arguments

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `n=`$(Manopt._link(:rand; M = "N"))
* `p=`$(Manopt._link(:rand))
* `m=`$(Manopt._link(:rand))
* `X=`$(Manopt._link(:zero_vector))
* `acceleration=0.0`
* `dual_stepsize=1/sqrt(8)`
* `primal_stepsize=1/sqrt(8)`
$(_kwargs(:inverse_retraction_method))
$(_kwargs(:inverse_retraction_method; name = "inverse_retraction_method_dual", M = "N", p = "n"))
* `relaxation=1.0`
* `relax=:primal`: relax the primal variable by default
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(300)"))
* `variant=:exact`: run the exact Chambolle Pock by default
* `update_primal_base=missing`
* `update_dual_base=missing`
$(_kwargs(:vector_transport_method))
$(_kwargs(:vector_transport_method; name = "vector_transport_method_dual", M = "N", p = "n"))

if `Manifolds.jl` is loaded, `N` is also a keyword argument and set to `TangentBundle(M)` by default.

!!! note
    Constructing the default `n` on a `TangentBundle` requires
    [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl) to be loaded as well.
"""
mutable struct ChambollePockState{
        P, Q, T, R, C <: AbstractDict{Symbol}, SC <: StoppingCriterion,
        RM <: AbstractRetractionMethod, IRM <: AbstractInverseRetractionMethod,
        IRM_Dual <: AbstractInverseRetractionMethod,
        VTM <: AbstractVectorTransportMethod, VTM_Dual <: AbstractVectorTransportMethod,
    } <: AbstractPrimalDualSolverState
    callbacks::C
    m::P
    n::Q
    p::P
    pbar::P
    X::T
    Xbar::T
    primal_stepsize::R
    dual_stepsize::R
    acceleration::R
    relaxation::R
    relax::Symbol
    stop::SC
    variant::Symbol
    update_primal_base::Union{Function, Missing}
    update_dual_base::Union{Function, Missing}
    retraction_method::RM
    inverse_retraction_method::IRM
    inverse_retraction_method_dual::IRM_Dual
    vector_transport_method::VTM
    vector_transport_method_dual::VTM_Dual
end
function Manopt.ChambollePockState(
        M::AbstractManifold,
        N::AbstractManifold;
        m::P = rand(M),
        n::Q = rand(N),
        p::P = rand(M),
        X::T = zero_vector(M, p),
        callbacks::C = Dict{Symbol, Function}(),
        primal_stepsize::R = 1 / sqrt(8),
        dual_stepsize::R = 1 / sqrt(8),
        acceleration::R = 0.0,
        relaxation::R = 1.0,
        relax::Symbol = :primal,
        stopping_criterion::SC = StopAfterIteration(300),
        variant::Symbol = :exact,
        update_primal_base::Union{Function, Missing} = missing,
        update_dual_base::Union{Function, Missing} = missing,
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
        inverse_retraction_method_dual::IRM_Dual = default_inverse_retraction_method(N, typeof(p)),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(n)),
        vector_transport_method_dual::VTM_Dual = default_vector_transport_method(N, typeof(n)),
    ) where {
        P, Q, T, R, C <: AbstractDict{Symbol}, SC <: StoppingCriterion,
        RM <: AbstractRetractionMethod, IRM <: AbstractInverseRetractionMethod,
        IRM_Dual <: AbstractInverseRetractionMethod,
        VTM <: AbstractVectorTransportMethod, VTM_Dual <: AbstractVectorTransportMethod,
    }
    return ChambollePockState{P, Q, T, R, C, SC, RM, IRM, IRM_Dual, VTM, VTM_Dual}(
        callbacks,
        m,
        n,
        p,
        copy(M, p),
        X,
        copy(N, X),
        primal_stepsize,
        dual_stepsize,
        acceleration,
        relaxation,
        relax,
        stopping_criterion,
        variant,
        update_primal_base,
        update_dual_base,
        retraction_method,
        inverse_retraction_method,
        inverse_retraction_method_dual,
        vector_transport_method,
        vector_transport_method_dual,
    )
end
provided_callbacks(::Type{<:ChambollePockState}) = _MANOPT_DEFAULT_CALLBACKS
get_callbacks(state::ChambollePockState) = state.callbacks
function status_summary(cps::ChambollePockState; context::Symbol = :default)
    (context === :short) && return repr(cps)
    i = get_count(cps, :Iterations)
    conv_inl = (i > 0) ? (has_converged(cps.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for Chambolle-Pock algorithm$(conv_inl)"
    i = get_count(cps, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(cps.stop) ? "Yes" : "No"
    as = _callbacks_summary(cps)
    s = """
    # Solver state for `Manopt.jl`s Chambolle-Pock Algorithm
    $Iter
    ## Parameters$(as)
    * primal_stepsize:  $(cps.primal_stepsize)
    * dual_stepsize:    $(cps.dual_stepsize)
    * acceleration:     $(cps.acceleration)
    * relaxation:       $(cps.relaxation)
    * relax:            $(cps.relax)
    * variant:          :$(cps.variant)
    * retraction_method:              $(cps.retraction_method)
    * inverse_retraction_method:      $(cps.inverse_retraction_method)
    * vector_transport_method:        $(cps.vector_transport_method)
    * inverse_retraction_method_dual: $(cps.inverse_retraction_method_dual)
    * vector_transport_method_dual:   $(cps.vector_transport_method_dual)

    ## Stopping criterion
    $(_in_str(status_summary(cps.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv"""
    return s
end
get_solver_result(apds::AbstractPrimalDualSolverState) = get_iterate(apds)
get_iterate(apds::AbstractPrimalDualSolverState) = apds.p
function set_iterate!(apds::AbstractPrimalDualSolverState, p)
    apds.p = p
    return apds
end

_tex_DΛ = "DΛ: $(_math(:TangentSpace; p = "m")) → $(_math(:TangentSpace; p = "Λ(m)", M = "N"))"

_doc_ChambollePock_formula = """
Given a `cost` function ``$(_tex(:Cal, "E")): $(_math(:Manifold)) → ℝ`` of the form

    ```math
$(_tex(:Cal, "E"))(p) = F(p) + G( Λ(p) ),
```

where ``F:$(_math(:Manifold)) → ℝ``, ``G:$(_math(:Manifold; M = "N")) → ℝ``,
and ``Λ:$(_math(:Manifold)) → $(_math(:Manifold; M = "N"))``.
"""

_doc_ChambollePock = """
    ChambollePock(M, N, f, p, X, m, n, prox_G, prox_G_dual, get_adjoint_linear_operator; kwargs...)
    ChambollePock!(M, N, f, p, X, m, n, prox_G, prox_G_dual, get_adjoint_linear_operator; kwargs...)


Perform the Riemannian Chambolle-Pock algorithm.

$_doc_ChambollePock_formula

This can be done inplace of ``p``.

 # Input parameters

$(_args(:M))
$(_args(:M; name = "N"))
$(_args(:p))
$(_args(:X))
$(_args(:p; name = "m"))
$(_args(:p; name = "n", M = "N"))
* `adjoint_linearized_operator`:  the adjoint ``DΛ^*`` of the linearized operator ``$(_tex_DΛ)``
* `prox_F, prox_G_Dual`:          the proximal maps of ``F`` and ``G^$(_tex(:ast))_n``

By default, this performs the exact Riemannian Chambolle Pock algorithm, see the optional parameter
`DΛ` for their linearized variant.

For more details on the algorithm, see [BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021](@cite).

# Keyword Arguments

* `acceleration=0.05`: acceleration parameter
* `dual_stepsize=1/sqrt(8)`: proximal parameter of the dual prox
$(_kwargs([:evaluation, :inverse_retraction_method]))
$(_kwargs(:inverse_retraction_method; name = "inverse_retraction_method_dual", M = "N", p = "n"))
* `Λ=missing`: the (forward) operator ``Λ(⋅)`` (required for the `:exact` variant)
* `linearized_forward_operator=missing`: its linearization ``DΛ(⋅)[⋅]`` (required for the `:linearized` variant)
* `primal_stepsize=1/sqrt(8)`: proximal parameter of the dual prox
* `relaxation=1.`: the relaxation parameter ``γ``
* `relax=:primal`: whether to relax the primal or dual
* `variant=:linearized` if `Λ` is missing, otherwise `:exact`: variant to use.
  Note that this changes the arguments the `forward_operator` is called with.
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)"))
* `update_primal_base=missing`: function to update `m` (identity by default/missing)
* `update_dual_base=missing`: function to update `n` (identity by default/missing)
$(_kwargs([:retraction_method, :vector_transport_method]))
$(_kwargs(:vector_transport_method; name = "vector_transport_method_dual", M = "N", p = "n"))

$(_note(:OutputSection))
"""

@doc "$(_doc_ChambollePock)"
function ChambollePock(
        M::AbstractManifold, N::AbstractManifold, cost::TF, p::P, X::T, m::P, n::Q,
        prox_F::Function, prox_G_dual::Function, adjoint_linear_operator::Function;
        Λ::Union{Function, Missing} = missing, linearized_forward_operator::Union{Function, Missing} = missing,
        kwargs...,
    ) where {TF, P, T, Q}
    q = copy(M, p)
    Y = copy(N, n, X)
    m2 = copy(M, m)
    n2 = copy(N, n)
    keywords_accepted(ChambollePock; kwargs...)
    return ChambollePock!(
        M, N, cost, q, Y, m2, n2, prox_F, prox_G_dual, adjoint_linear_operator;
        Λ = Λ, linearized_forward_operator = linearized_forward_operator, kwargs...,
    )
end
calls_with_kwargs(::typeof(ChambollePock)) = (ChambollePock!,)

@doc "$(_doc_ChambollePock)"
function ChambollePock!(
        M::AbstractManifold, N::AbstractManifold, cost::TF, p::P, X::T, m::P, n::Q,
        prox_F::Function, prox_G_dual::Function, adjoint_linear_operator::Function;
        callbacks = Dict{Symbol, Function}(),
        Λ::Union{Function, Missing} = missing,
        linearized_forward_operator::Union{Function, Missing} = missing,
        acceleration = 0.05,
        dual_stepsize = 1 / sqrt(8),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        primal_stepsize = 1 / sqrt(8),
        relaxation = 1.0,
        relax::Symbol = :primal,
        stopping_criterion::StoppingCriterion = StopAfterIteration(200),
        update_primal_base::Union{Function, Missing} = missing,
        update_dual_base::Union{Function, Missing} = missing,
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        variant = ismissing(Λ) ? :linearized : :exact,
        kwargs...,
    ) where {
        TF, P, Q, T, RM <: AbstractRetractionMethod, IRM <: AbstractInverseRetractionMethod,
        VTM <: AbstractVectorTransportMethod,
    }
    pdmo = PrimalDualManifoldObjective(
        cost, prox_F, prox_G_dual, adjoint_linear_operator;
        linearized_forward_operator = linearized_forward_operator, Λ = Λ,
        evaluation = evaluation
    )
    keywords_accepted(ChambollePock!; kwargs...)
    (variant === :exact && ismissing(Λ)) &&
        throw(ArgumentError("The `:exact` variant requires the forward operator `Λ`."))
    (variant === :linearized && ismissing(linearized_forward_operator)) &&
        throw(ArgumentError("The `:linearized` variant requires the `linearized_forward_operator`."))
    dpdmo = decorate_objective!(M, pdmo; kwargs...)
    tmp = TwoManifoldProblem(M, N, dpdmo)
    cps = ChambollePockState(
        M, N;
        m = m, n = n, p = p, X = X,
        callbacks = process_callbacks_arg(callbacks, ChambollePockState),
        primal_stepsize = primal_stepsize,
        dual_stepsize = dual_stepsize,
        acceleration = acceleration,
        relaxation = relaxation,
        stopping_criterion = stopping_criterion,
        relax = relax,
        update_primal_base = update_primal_base,
        update_dual_base = update_dual_base,
        variant = variant,
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
        vector_transport_method = vector_transport_method,
    )
    dcps = decorate_state!(cps; kwargs...)
    solve!(tmp, dcps)
    return get_solver_return(get_objective(tmp), dcps)
end
calls_with_kwargs(::typeof(ChambollePock!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(::TwoManifoldProblem, ::ChambollePockState) end

function step_solver!(tmp::TwoManifoldProblem, cps::ChambollePockState, iter)
    N = get_manifold(tmp, 2)
    primal_dual_step!(tmp, cps, Val(cps.relax))
    cps.m =
        ismissing(cps.update_primal_base) ? cps.m : cps.update_primal_base(tmp, cps, iter)
    if !ismissing(cps.update_dual_base)
        n_old = deepcopy(cps.n)
        cps.n = cps.update_dual_base(tmp, cps, iter)
        vector_transport_to!(
            N, cps.X, n_old, cps.X, cps.n, cps.vector_transport_method_dual
        )
        vector_transport_to!(
            N, cps.Xbar, n_old, cps.Xbar, cps.n, cps.vector_transport_method_dual
        )
    end
    return cps
end
#
# Variant 1: primal relax
#
function primal_dual_step!(tmp::TwoManifoldProblem, cps::ChambollePockState, ::Val{:primal})
    dual_update!(tmp, cps, cps.pbar, Val(cps.variant))
    obj = get_objective(tmp)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    if !hasproperty(obj, :Λ!) || ismissing(obj.Λ!)
        ptXn = cps.X
    else
        ptXn = vector_transport_to(
            N, cps.n, cps.X, forward_operator(tmp, cps.m), cps.vector_transport_method_dual
        )
    end
    p_old = cps.p
    cps.p = get_primal_prox!(
        tmp,
        cps.p,
        cps.primal_stepsize,
        retract(
            M,
            cps.p,
            vector_transport_to(
                M, cps.m,
                -cps.primal_stepsize *
                    (adjoint_linearized_operator(tmp, cps.m, cps.n, ptXn)),
                cps.p, cps.vector_transport_method,
            ),
            cps.retraction_method,
        ),
    )
    update_prox_parameters!(cps)
    retract!(
        M, cps.pbar, cps.p,
        -cps.relaxation * inverse_retract(M, cps.p, p_old, cps.inverse_retraction_method),
        cps.retraction_method,
    )
    return cps
end
#
# Variant 2: dual relax
#
function primal_dual_step!(tmp::TwoManifoldProblem, cps::ChambollePockState, ::Val{:dual})
    obj = get_objective(tmp)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    if !hasproperty(obj, :Λ!) || ismissing(obj.Λ!)
        ptXbar = cps.Xbar
    else
        ptXbar = vector_transport_to(
            N, cps.n, cps.Xbar, forward_operator(tmp, cps.m), cps.vector_transport_method_dual,
        )
    end
    get_primal_prox!(
        tmp, cps.p, cps.primal_stepsize,
        retract(
            M, cps.p,
            vector_transport_to(
                M, cps.m,
                -cps.primal_stepsize * (adjoint_linearized_operator(tmp, cps.m, cps.n, ptXbar)),
                cps.p, cps.vector_transport_method,
            ),
            cps.retraction_method,
        ),
    )
    X_old = deepcopy(cps.X)
    dual_update!(tmp, cps, cps.p, Val(cps.variant))
    update_prox_parameters!(cps)
    cps.Xbar = cps.X + cps.relaxation * (cps.X - X_old)
    return cps
end
#
# Dual step: linearized
# depending on whether its primal relaxed or dual relaxed, start from start=o.x or start=o.xbar
#
function dual_update!(
        tmp::TwoManifoldProblem, cps::ChambollePockState, start::P, ::Val{:linearized}
    ) where {P}
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    obj = get_objective(tmp)
    # (1) compute update direction
    X_update = linearized_forward_operator(
        tmp, cps.m, inverse_retract(M, cps.m, start, cps.inverse_retraction_method), cps.n
    )
    # (2) if p.Λ is missing, if n = Λ(m) and do not PT, otherwise do
    (hasproperty(obj, :Λ!) && !ismissing(obj.Λ!)) && vector_transport_to!(
        N, X_update, forward_operator(tmp, cps.m), X_update, cps.n, cps.vector_transport_method_dual,
    )
    # (3) to the dual update
    get_dual_prox!(
        tmp, cps.X, cps.n, cps.dual_stepsize, cps.X + cps.dual_stepsize * X_update
    )
    return cps
end
#
# Dual step: exact
# depending on whether its primal relaxed or dual relaxed start from start=o.x or start=o.xbar here
#
function dual_update!(
        tmp::TwoManifoldProblem, cps::ChambollePockState, start::P, ::Val{:exact}
    ) where {P}
    N = get_manifold(tmp, 2)
    ξ_update = inverse_retract(
        N, cps.n, forward_operator(tmp, start), cps.inverse_retraction_method_dual
    )
    get_dual_prox!(
        tmp, cps.X, cps.n, cps.dual_stepsize, cps.X + cps.dual_stepsize * ξ_update
    )
    return cps
end

@doc """
    update_prox_parameters!(o)
update the prox parameters as described in Algorithm 2 of [ChambollePock:2011](@cite),

1. ``θ_{n} = $(_tex(:frac, "1", "$(_tex(:sqrt, "1+2γτ_n"))"))``
2. ``τ_{n+1} = θ_nτ_n``
3. ``σ_{n+1} = $(_tex(:frac, "σ_n", "θ_n"))``
"""
function update_prox_parameters!(pds::S) where {S <: AbstractPrimalDualSolverState}
    if pds.acceleration > 0
        pds.relaxation = 1 / sqrt(1 + 2 * pds.acceleration * pds.primal_stepsize)
        pds.primal_stepsize = pds.primal_stepsize * pds.relaxation
        pds.dual_stepsize = pds.dual_stepsize / pds.relaxation
    end
    return pds
end
