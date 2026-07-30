@doc """
    PrimalDualManifoldSemismoothNewtonObjective{TC, LO, TALO, PF, DPF, PG, DPG, L} <: AbstractPrimalDualManifoldObjective{TC, PF}

Describes a Problem for the Primal-dual Riemannian semismooth Newton algorithm. [DiepeveenLellmann:2021](@cite)

# Fields

* `cost`:                        ``F + G(Λ(⋅))`` to evaluate interim cost function values
* `linearized_operator`:         the linearization ``DΛ(⋅)[⋅]`` of the operator ``Λ(⋅)``.
* `linearized_adjoint_operator`: the adjoint differential ``(DΛ)^* : $(_math(:Manifold; M = "N")) → $(_math(:TangentBundle))``
* `prox_F`:                      the proximal map belonging to ``F``
* `diff_prox_F`:                 the (Clarke Generalized) differential of the proximal maps of ``F``
* `prox_G_dual`:                 the proximal map belonging to `G^$(_tex(:ast))_n``
* `diff_prox_dual_G`:            the (Clarke Generalized) differential of the proximal maps of ``G^$(_tex(:ast))_n``
* `Λ`:                           the exact forward operator. This operator is required if `Λ(m)=n` does not hold.

# Constructor

    PrimalDualManifoldSemismoothNewtonObjective(cost, prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator,Λ)
"""
mutable struct PrimalDualManifoldSemismoothNewtonObjective{
        TC, PF, DPF, PG, DPG, LFO, TALO, L,
    } <: AbstractPrimalDualManifoldObjective{TC, PF}
    cost::TC
    prox_f!::PF
    diff_prox_f!::DPF
    prox_g_dual!::PG
    diff_prox_g_dual!::DPG
    linearized_forward_operator!::LFO
    adjoint_linearized_operator!::TALO
    Λ!::L
end
function PrimalDualManifoldSemismoothNewtonObjective(
        cost::C, prox_F::PF, diff_prox_F::DPF, prox_G_dual::PG, diff_prox_G_dual::DPG,
        linearized_forward_operator::LFO, adjoint_linearized_operator::AL;
        Λ::L = missing, evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    ) where {C, PF, DPF, PG, DPG, LFO, AL, L}
    cost_ = maybe_wrap_function(cost, evaluation; result = :Number)
    prox_F_ = maybe_wrap_function(prox_F, evaluation)
    diff_prox_F_ = maybe_wrap_function(diff_prox_F, evaluation)
    prox_G_dual_ = maybe_wrap_function(prox_G_dual, evaluation)
    diff_prox_G_dual_ = maybe_wrap_function(diff_prox_G_dual, evaluation)
    linearized_forward_operator_ = maybe_wrap_function(linearized_forward_operator, evaluation; result = :Vector)
    adjoint_linearized_operator_ = maybe_wrap_function(adjoint_linearized_operator, evaluation)
    Λ_ = ismissing(Λ) ? missing : maybe_wrap_function(Λ, evaluation)
    return PrimalDualManifoldSemismoothNewtonObjective{typeof(cost_), typeof(prox_F_), typeof(diff_prox_F_), typeof(prox_G_dual_), typeof(diff_prox_G_dual_), typeof(linearized_forward_operator_), typeof(adjoint_linearized_operator_), typeof(Λ_)}(
        cost_, prox_F_, diff_prox_F_, prox_G_dual_, diff_prox_G_dual_, linearized_forward_operator_, adjoint_linearized_operator_, Λ_,
    )
end

@doc """
    PrimalDualSemismoothNewtonState <: AbstractPrimalDualSolverState

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
* `dual_stepsize::Float64`:    proximal parameter of the dual prox
$(_fields(:inverse_retraction_method))
$(_fields(:p; name = "m"))
$(_fields(:p; type = "Q", name = "n", M = "N"))
$(_fields(:p; add_properties = [:as_Iterate]))
* `primal_stepsize::Float64`:  proximal parameter of the primal prox
* `reg_param::Float64`:        regularisation parameter for the Newton matrix
$(_fields(:retraction_method))
$(_fields(:stopping_criterion; name = "stop"))
* `update_dual_base`:          function to update the dual base
* `update_primal_base`:        function to update the primal base
$(_fields(:vector_transport_method))
$(_fields(:X))

where for the update functions a [`AbstractManoptProblem`](@ref) `amp`,
[`AbstractManoptSolverState`](@ref) `ams` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing`).

# Constructor

    PrimalDualSemismoothNewtonState(M::AbstractManifold; kwargs...)

Generate a state for the [`primal_dual_semismooth_Newton`](@ref).

## Keyword arguments

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `dual_stepsize=1/sqrt(8)`
$(Manopt._kwargs([:inverse_retraction_method]))
* `m=`$(Manopt._link(:rand))
* `n=`$(Manopt._link(:rand; M = "N"))
* `p=`$(Manopt._link(:rand))
* `primal_stepsize=1/sqrt(8)`
* `reg_param=1e-5`
$(Manopt._kwargs([:retraction_method]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(50)`"))
* `update_dual_base=(amp, ams, k) -> o.n`
* `update_primal_base=(amp, ams, k) -> o.m`
$(_kwargs(:vector_transport_method))
* `X=`$(Manopt._link(:zero_vector))
"""
mutable struct PrimalDualSemismoothNewtonState{
        P, Q, T, C <: AbstractDict{Symbol}, RM <: AbstractRetractionMethod,
        IRM <: AbstractInverseRetractionMethod, VTM <: AbstractVectorTransportMethod,
    } <: AbstractPrimalDualSolverState
    callbacks::C
    dual_stepsize::Float64
    inverse_retraction_method::IRM
    m::P
    n::Q
    p::P
    primal_stepsize::Float64
    regularization_parameter::Float64
    retraction_method::RM
    stop::StoppingCriterion
    update_dual_base::Union{Function, Missing}
    update_primal_base::Union{Function, Missing}
    vector_transport_method::VTM
    X::T
    function PrimalDualSemismoothNewtonState(
            M::AbstractManifold;
            callbacks::C = Dict{Symbol, Function}(),
            dual_stepsize::Float64 = 1 / sqrt(8),
            m::P = rand(M), n::Q = rand(N), p::P = rand(M),
            primal_stepsize::Float64 = 1 / sqrt(8),
            regularization_parameter::Float64 = 1.0e-5,
            stopping_criterion::StoppingCriterion = StopAfterIteration(50),
            update_dual_base::Union{Function, Missing} = missing,
            update_primal_base::Union{Function, Missing} = missing,
            # the following defaults depend on `p`, so they have to be keyword arguments
            # listed after `p` is bound above
            inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
            retraction_method::RM = default_retraction_method(M, typeof(p)),
            vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
            X::T = zero_vector(M, p),
        ) where {
            P, Q, T, C <: AbstractDict{Symbol}, RM <: AbstractRetractionMethod,
            IRM <: AbstractInverseRetractionMethod, VTM <: AbstractVectorTransportMethod,
        }
        return new{P, Q, T, C, RM, IRM, VTM}(
            callbacks, dual_stepsize, inverse_retraction_method, m, n, p,
            primal_stepsize, regularization_parameter, retraction_method,
            stopping_criterion, update_dual_base, update_primal_base,
            vector_transport_method, X,
        )
    end
end
get_callbacks(pdsn::PrimalDualSemismoothNewtonState) = pdsn.callbacks

@doc """
    y = get_differential_primal_prox(M::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective σ, x)
    get_differential_primal_prox!(p::TwoManifoldProblem, y, σ, x)

Evaluate the differential proximal map of ``F`` stored within [`AbstractPrimalDualManifoldObjective`](@ref)

```math
D$(_tex(:prox))_{σF}(x)[X]
```

which can also be computed in place of `y`.
"""
get_differential_primal_prox(
    M::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective, ::Any...
)

function get_differential_primal_prox(tmo::TwoManifoldProblem, σ, p, X)
    M = get_manifold(tmo, 1)
    pdsno = get_objective(tmo)
    return get_differential_primal_prox(M, pdsno, σ, p, X)
end
function get_differential_primal_prox!(tmo::TwoManifoldProblem, Y, σ, p, X)
    M = get_manifold(tmo, 1)
    pdsno = get_objective(tmo)
    get_differential_primal_prox!(M, Y, pdsno, σ, p, X)
    return Y
end

function get_differential_primal_prox(
        M::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective, σ, p, X,
    )
    Y = allocate_result(M, get_differential_primal_prox, p, X)
    pdsno.diff_prox_f!(M, Y, σ, p, X)
    return Y
end
function get_differential_primal_prox(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, σ, p, X)
    return get_differential_primal_prox(M, get_objective(admo, false), σ, p, X)
end
function get_differential_primal_prox!(
        M::AbstractManifold, Y, pdsno::PrimalDualManifoldSemismoothNewtonObjective, σ, p, X,
    )
    pdsno.diff_prox_f!(M, Y, σ, p, X)
    return Y
end
function get_differential_primal_prox!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, σ, p, X
    )
    return get_differential_primal_prox!(M, Y, get_objective(admo, false), σ, p, X)
end

@doc """
    η = get_differential_dual_prox(N::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective, n, τ, X, ξ)
    get_differential_dual_prox!(N::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective, η, n, τ, X, ξ)

Evaluate the differential proximal map of ``G_n^*`` stored within [`PrimalDualManifoldSemismoothNewtonObjective`](@ref)

```math
D$(_tex(:prox))_{τG_n^*}(X)[ξ]
```

which can also be computed in place of `η`.
"""
get_differential_dual_prox(
    ::AbstractManifold, ::PrimalDualManifoldSemismoothNewtonObjective, Any...,
)

function get_differential_dual_prox(tmo::TwoManifoldProblem, n, τ, X, ξ)
    N = get_manifold(tmo, 2)
    pdsno = get_objective(tmo)
    return get_differential_dual_prox(N, pdsno, n, τ, X, ξ)
end
function get_differential_dual_prox!(tmo::TwoManifoldProblem, η, n, τ, X, ξ)
    N = get_manifold(tmo, 2)
    pdsno = get_objective(tmo)
    get_differential_dual_prox!(N, η, pdsno, n, τ, X, ξ)
    return η
end
function get_differential_dual_prox(
        N::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective, n, τ, X, ξ,
    )
    η = allocate_result(N, get_differential_dual_prox, X, ξ)
    pdsno.diff_prox_g_dual!(N, η, n, τ, X, ξ)
    return η
end
function get_differential_dual_prox(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, n, τ, X, ξ
    )
    return get_differential_dual_prox(M, get_objective(admo, false), n, τ, X, ξ)
end
function get_differential_dual_prox!(
        N::AbstractManifold, η, pdsno::PrimalDualManifoldSemismoothNewtonObjective, n, τ, X, ξ,
    )
    pdsno.diff_prox_g_dual!(N, η, n, τ, X, ξ)
    return η
end
function get_differential_dual_prox!(
        M::AbstractManifold, η, admo::AbstractDecoratedManifoldObjective, n, τ, X, ξ
    )
    return get_differential_dual_prox!(M, η, get_objective(admo, false), n, τ, X, ξ)
end

get_iterate(pdsn::PrimalDualSemismoothNewtonState) = pdsn.p

function set_iterate!(pdsn::PrimalDualSemismoothNewtonState, p)
    pdsn.p = p
    return pdsn
end

function Base.show(io::IO, pdsns::PrimalDualSemismoothNewtonState)
    print(io, "PrimalDualSemismoothNewtonState(; ")
    print(io, "callbacks = ", pdsns.callbacks, ", ")
    print(io, "dual_stepsize = ", pdsns.dual_stepsize, ", ")
    print(io, "inverse_retraction_method = ", pdsns.inverse_retraction_method, ", ")
    print(io, "m = ", pdsns.m, ", n = ", pdsns.n, ", p = ", pdsns.p, ", ")
    print(io, "primal_stepsize = ", pdsns.primal_stepsize, ", ")
    print(io, "regularization_parameter = ", pdsns.regularization_parameter, ", ")
    print(io, "retraction_method = ", pdsns.retraction_method, ", ")
    print(io, "stopping_criterion = ", status_summary(pdsns.stop; context = :short), ", ")
    print(io, "update_dual_base = ", pdsns.update_dual_base, ", update_primal_base = ", pdsns.update_primal_base, ", ")
    print(io, "vector_transport_method = ", pdsns.vector_transport_method, ", X = ", pdsns.X)
    return print(io, ")")
end

function status_summary(pdsns::PrimalDualSemismoothNewtonState; context::Symbol = :default)
    (context === :short) && return repr(pdsns)
    i = get_count(pdsns, :Iterations)
    conv_inl = (i > 0) ? (indicates_convergence(pdsns.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the primal dual semismooth Newton solver$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pdsns.stop) ? "Yes" : "No"
    as = _callbacks_summary(pdsns)
    s = """
    # Solver state for `Manopt.jl`s primal dual semismooth Newton
    $Iter
    ## Parameters$(as)
    * primal_stepsize:          $(_MANOPT_INDENT)$(pdsns.primal_stepsize)
    * dual_stepsize:            $(_MANOPT_INDENT)$(pdsns.dual_stepsize)
    * regularization_parameter: $(_MANOPT_INDENT)$(pdsns.regularization_parameter)
    * retraction_method:        $(_MANOPT_INDENT)$(pdsns.retraction_method)
    * inverse_retraction_method:$(_MANOPT_INDENT)$(pdsns.inverse_retraction_method)
    * vector_transport_method:  $(_MANOPT_INDENT)$(pdsns.vector_transport_method)

    ## Stopping criterion
    $(_in_str(status_summary(pdsns.stop; context = context); indent = 0, headers = 1))
    This indicates convergence: $Conv"""
    return s
end
function Base.show(io::IO, pdmssno::PrimalDualManifoldSemismoothNewtonObjective)
    print(io, "PrimalDualManifoldSemismoothNewtonObjective(")
    print(io, pdmssno.cost); print(io, ", ")
    print(io, pdmssno.prox_f!); print(io, ", ")
    print(io, pdmssno.diff_prox_f!); print(io, ", ")
    print(io, pdmssno.prox_g_dual!); print(io, ", ")
    print(io, pdmssno.diff_prox_g_dual!); print(io, ", ")
    print(io, pdmssno.linearized_forward_operator!); print(io, ", ")
    print(io, pdmssno.adjoint_linearized_operator!); print(io, "; ")
    if !ismissing(pdmssno.Λ!)
        print(io, "Λ = "); print(io, pdmssno.Λ!); print(io, ", ")
    end
    return print(io, ")")
end
function status_summary(pdmssno::PrimalDualManifoldSemismoothNewtonObjective; context::Symbol = :default)
    (context === :short) && return repr(pdmssno)
    (context === :inline) && return "A primal dual semismooth Newton objective"
    Λs = ismissing(pdmssno.Λ!) ? "" : "\n* Λ:                $(_MANOPT_INDENT)$(pdmssno.Λ!)"
    return """
    A primal dual semismooth Newton objective

    ## Functions
    * cost:             $(_MANOPT_INDENT)$(pdmssno.cost)
    * prox_f:           $(_MANOPT_INDENT)$(pdmssno.prox_f!)
    * D prox_f:         $(_MANOPT_INDENT)$(pdmssno.diff_prox_f!)
    * prox_g*:          $(_MANOPT_INDENT)$(pdmssno.prox_g_dual!)
    * D prox_g*:        $(_MANOPT_INDENT)$(pdmssno.diff_prox_g_dual!)
    * lin. forward Op:  $(_MANOPT_INDENT)$(pdmssno.linearized_forward_operator!)
    * adj. lin. fw. Op.:$(_MANOPT_INDENT)$(pdmssno.adjoint_linearized_operator!)$(Λs)"""
end


_doc_PDSN_formula = raw"""
Given a `cost` function ``\mathcal E: \mathcal M → \overline{ℝ}`` of the form
```math
\mathcal E(p) = F(p) + G( Λ(p) ),
```
where ``F: \mathcal M → \overline{ℝ}``, ``G: \mathcal N → \overline{ℝ}``,
and ``Λ: \mathcal M → \mathcal N``. The remaining input parameters are
"""

_doc_PDSN = """
    primal_dual_semismooth_Newton(M, N, cost, p, X, m, n, prox_F, diff_prox_F, prox_G_dual, diff_prox_dual_G, linearized_operator, adjoint_linearized_operator)

Perform the Primal-Dual Riemannian semismooth Newton algorithm.

$(_doc_PDSN_formula)

* `p, X`:                          primal and dual start points ``p∈$(_math(:Manifold))`` and ``X ∈ T_n$(_math(:Manifold, M = "N"))``
* `m,n`:                           base points on ``$(_math(:Manifold))`` and ``$(_math(:Manifold, M = "N"))``, respectively.
* `linearized_forward_operator`:   the linearization ``DΛ(⋅)[⋅]`` of the operator ``Λ(⋅)``.
* `adjoint_linearized_operator`:   the adjoint ``DΛ^*`` of the linearized operator ``DΛ(m):  $(_math(:TangentSpace; p = "m")) → $(_math(:TangentSpace; M = "N", p = "Λ(m)"))``
* `prox_F, prox_G_Dual`:           the proximal maps of ``F`` and ``G^$(_tex(:ast))_n``
* `diff_prox_F, diff_prox_dual_G`: the (Clarke Generalized) differentials of the proximal maps of ``F`` and ``G^$(_tex(:ast))_n``

For more details on the algorithm, see [DiepeveenLellmann:2021](@cite).

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
* `dual_stepsize=1/sqrt(8)`: proximal parameter of the dual prox
$(_kwargs([:evaluation, :inverse_retraction_method]))
* `Λ=missing`: the exact operator, that is required if `Λ(m)=n` does not hold;
  `missing` indicates, that the forward operator is exact.
* `primal_stepsize=1/sqrt(8)`: proximal parameter of the primal prox
* `reg_param=1e-5`: regularisation parameter for the Newton matrix
  Note that this changes the arguments the `forward_operator` is called.
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(50)"))
* `update_primal_base=missing`: function to update `m` (identity by default/missing)
* `update_dual_base=missing`: function to update `n` (identity by default/missing)
$(_kwargs(:vector_transport_method))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_PDSN)"
function primal_dual_semismooth_Newton(
        M::AbstractManifold, N::AbstractManifold, cost::TF, p::P, X::T, m::P, n::Q,
        prox_F::Function, diff_prox_F::Function, prox_G_dual::Function, diff_prox_G_dual::Function,
        linearized_forward_operator::Function, adjoint_linearized_operator::Function;
        Λ::Union{Function, Missing} = missing,
        kwargs...,
    ) where {TF, P, T, Q}
    keywords_accepted(primal_dual_semismooth_Newton; kwargs...)
    x_res = copy(M, p)
    ξ_res = copy(N, n, X)
    m_res = copy(M, m)
    n_res = copy(N, n)
    return primal_dual_semismooth_Newton!(
        M, N, cost, x_res, ξ_res, m_res, n_res,
        prox_F, diff_prox_F, prox_G_dual, diff_prox_G_dual,
        linearized_forward_operator, adjoint_linearized_operator;
        Λ = Λ, kwargs...,
    )
end
calls_with_kwargs(::typeof(primal_dual_semismooth_Newton)) = (primal_dual_semismooth_Newton!,)

@doc "$(_doc_PDSN)"
function primal_dual_semismooth_Newton!(
        M::mT, N::nT, cost::Function, p::P, X::T, m::P, n::Q,
        prox_F::Function, diff_prox_F::Function, prox_G_dual::Function, diff_prox_G_dual::Function,
        linearized_forward_operator::Function, adjoint_linearized_operator::Function;
        callbacks = Dict{Symbol, Function}(),
        dual_stepsize = 1 / sqrt(8),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        Λ::Union{Function, Missing} = missing,
        primal_stepsize = 1 / sqrt(8),
        reg_param = 1.0e-5,
        stopping_criterion::StoppingCriterion = StopAfterIteration(50),
        update_primal_base::Union{Function, Missing} = missing,
        update_dual_base::Union{Function, Missing} = missing,
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        kwargs...,
    ) where {
        mT <: AbstractManifold, nT <: AbstractManifold, P, Q, T,
        RM <: AbstractRetractionMethod, IRM <: AbstractInverseRetractionMethod, VTM <: AbstractVectorTransportMethod,
    }
    keywords_accepted(primal_dual_semismooth_Newton!; kwargs...)
    pdmsno = PrimalDualManifoldSemismoothNewtonObjective(
        cost, prox_F, diff_prox_F, prox_G_dual, diff_prox_G_dual,
        linearized_forward_operator, adjoint_linearized_operator;
        Λ = Λ,
        evaluation = evaluation,
    )
    dpdmsno = decorate_objective!(M, pdmsno; kwargs...)
    tmp = TwoManifoldProblem(M, N, dpdmsno)
    pdsn = PrimalDualSemismoothNewtonState(
        M;
        callbacks = process_callbacks_arg(callbacks, PrimalDualSemismoothNewtonState),
        m = m,
        n = n,
        p = p,
        X = X,
        primal_stepsize = primal_stepsize,
        dual_stepsize = dual_stepsize,
        regularization_parameter = reg_param,
        stopping_criterion = stopping_criterion,
        update_primal_base = update_primal_base,
        update_dual_base = update_dual_base,
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
        vector_transport_method = vector_transport_method,
    )
    dpdsn = decorate_state!(pdsn; kwargs...)
    solve!(tmp, dpdsn)
    return get_solver_return(get_objective(tmp), dpdsn)
end
calls_with_kwargs(::typeof(primal_dual_semismooth_Newton!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(::TwoManifoldProblem, ::PrimalDualSemismoothNewtonState) end

function step_solver!(tmp::TwoManifoldProblem, pdsn::PrimalDualSemismoothNewtonState, iter)
    N = get_manifold(tmp, 2)
    # do step
    primal_dual_step!(tmp, pdsn)
    pdsn.m = if ismissing(pdsn.update_primal_base)
        pdsn.m
    else
        pdsn.update_primal_base(tmp, pdsn, iter)
    end
    if !ismissing(pdsn.update_dual_base)
        n_old = deepcopy(pdsn.n)
        pdsn.n = pdsn.update_dual_base(tmp, pdsn, iter)
        pdsn.X = vector_transport_to(N, n_old, pdsn.X, pdsn.n, pdsn.vector_transport_method)
    end
    return pdsn
end

function primal_dual_step!(tmp::TwoManifoldProblem, pdsn::PrimalDualSemismoothNewtonState)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    # construct X
    X = construct_primal_dual_residual_vector(tmp, pdsn)
    # construct matrix
    ∂X = construct_primal_dual_residual_covariant_derivative_matrix(tmp, pdsn)
    ∂X += pdsn.regularization_parameter * sparse(I, size(∂X))  # prevent singular matrix at solution
    # solve matrix -> find coordinates
    d_coords = ∂X \ -X
    dims = manifold_dimension(M)
    dx_coords = d_coords[1:dims]
    dξ_coords = d_coords[(dims + 1):end]
    # compute step
    dx = get_vector(M, pdsn.p, dx_coords, DefaultOrthonormalBasis())
    dξ = get_vector(N, pdsn.n, dξ_coords, DefaultOrthonormalBasis())
    # do step
    pdsn.p = retract(M, pdsn.p, dx, pdsn.retraction_method)
    return pdsn.X = pdsn.X + dξ
end

raw"""
    construct_primal_dual_residual_vector(p, o)

Constructs the vector representation of ``X(p^{(k)}, ξ_{n}^{(k)}) ∈ \mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}``
"""
function construct_primal_dual_residual_vector(
        tmp::TwoManifoldProblem, pdsn::PrimalDualSemismoothNewtonState
    )
    obj = get_objective(tmp)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    # Compute primal vector
    p_update = get_primal_prox(
        tmp,
        pdsn.primal_stepsize,
        retract(
            M,
            pdsn.p,
            vector_transport_to(
                M,
                pdsn.m,
                -pdsn.primal_stepsize * (adjoint_linearized_operator(tmp, pdsn.m, pdsn.n, pdsn.X)),
                pdsn.p,
                pdsn.vector_transport_method,
            ),
            pdsn.retraction_method,
        ),
    )

    primal_vector = -inverse_retract(M, pdsn.p, p_update, pdsn.inverse_retraction_method)

    X₁ = get_coordinates(M, pdsn.p, primal_vector, DefaultOrthonormalBasis())

    # Compute dual vector
    # (1) compute update direction
    ξ_update = linearized_forward_operator(
        tmp,
        pdsn.m,
        inverse_retract(M, pdsn.m, pdsn.p, pdsn.inverse_retraction_method),
        pdsn.n,
    )
    # (2) if p.Λ is missing, assume that n = Λ(m) and do not PT
    noPT = !hasproperty(obj, :Λ!) || ismissing(obj.Λ!)
    ξ_update = noPT ? ξ_update : vector_transport_to(
            N, forward_operator(tmp, pdsn.m), ξ_update, pdsn.n, pdsn.vector_transport_method,
        )
    # (3) the dual update
    ξ_update = get_dual_prox(
        tmp, pdsn.n, pdsn.dual_stepsize, pdsn.X + pdsn.dual_stepsize * ξ_update
    )
    dual_vector = pdsn.X - ξ_update
    X₂ = get_coordinates(N, pdsn.n, dual_vector, DefaultOrthonormalBasis())
    return [X₁; X₂]
end

raw"""
onstruct_primal_dual_residual_covariant_derivative_matrix(p, o)

Constructs the matrix representation of ``V^{(k)}:\mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}\rightarrow \mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}``
"""
function construct_primal_dual_residual_covariant_derivative_matrix(
        tmp::TwoManifoldProblem, pdsn::PrimalDualSemismoothNewtonState
    )
    obj = get_objective(tmp)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    # construct bases
    Θ = get_basis(M, pdsn.p, DefaultOrthonormalBasis())
    Ξ = get_basis(N, pdsn.n, DefaultOrthonormalBasis())

    dims = manifold_dimension(M)
    dualdims = manifold_dimension(N)

    # assume here that a parallel transport is already in the next operator
    qξ = -pdsn.primal_stepsize * adjoint_linearized_operator(tmp, pdsn.m, pdsn.n, pdsn.X)
    qₚ = shortest_geodesic(M, pdsn.m, pdsn.p, 1 / 2)
    qb = retract(M, pdsn.m, qξ, pdsn.retraction_method)
    q₅ = 2 * inverse_retract(M, qb, qₚ, pdsn.inverse_retraction_method)
    q₄ = retract(M, qb, q₅, pdsn.retraction_method)
    q₃ = -inverse_retract(M, pdsn.p, q₄, pdsn.inverse_retraction_method)
    q₂ = retract(M, pdsn.p, q₃, pdsn.retraction_method)
    q₁ = get_primal_prox(tmp, pdsn.primal_stepsize, q₂)

    # (1) compute update direction
    η₁ = linearized_forward_operator(
        tmp,
        pdsn.m,
        inverse_retract(M, pdsn.m, pdsn.p, pdsn.inverse_retraction_method),
        pdsn.n,
    )
    # (2) if p.Λ is missing, assume that n = Λ(m) and do  not PT
    noPT = !hasproperty(obj, :Λ!) || ismissing(obj.Λ!)

    η₁ = noPT ? η₁ : vector_transport_to(
            N, forward_operator(tmp, pdsn.m), η₁, pdsn.n, pdsn.vector_transport_method
        )
    # (3) to the dual update
    η₁ = pdsn.X + pdsn.dual_stepsize * η₁
    # construct ∂X₁₁ and ∂X₂₁
    ∂X₁₁ = spzeros(dims, dims)
    ∂X₂₁ = spzeros(dualdims, dims)

    Mdims = prod(manifold_dimension(M))
    for j in 1:Mdims
        eⱼ = zeros(Mdims)
        eⱼ[j] = 1
        Θⱼ = get_vector(M, pdsn.m, eⱼ, Θ)
        Gⱼ = differential_shortest_geodesic_endpoint(M, pdsn.m, pdsn.p, 1 / 2, Θⱼ)
        Fⱼ = 2 * differential_log_argument(M, qb, qₚ, Gⱼ)
        Eⱼ = differential_exp_argument(M, qb, q₅, Fⱼ)
        D₂ⱼ = -differential_log_argument(M, pdsn.p, q₄, Eⱼ)
        D₁ⱼ = -differential_log_basepoint(M, pdsn.p, q₄, Θⱼ)
        Dⱼ = D₁ⱼ + D₂ⱼ
        C₂ⱼ = differential_exp_argument(M, pdsn.p, q₃, Dⱼ)

        C₁ⱼ = differential_exp_basepoint(M, pdsn.p, q₃, Θⱼ)
        Cⱼ = C₁ⱼ + C₂ⱼ
        Bⱼ = get_differential_primal_prox(tmp, pdsn.primal_stepsize, q₂, Cⱼ)
        A₂ⱼ = -differential_log_argument(M, pdsn.p, q₁, Bⱼ)
        A₁ⱼ = -differential_log_basepoint(M, pdsn.p, q₁, Θⱼ)
        Aⱼ = A₁ⱼ + A₂ⱼ

        ∂X₁₁j = get_coordinates(M, pdsn.p, Aⱼ, DefaultOrthonormalBasis())
        sp_∂X₁₁j = sparsevec(∂X₁₁j)
        dropzeros!(sp_∂X₁₁j)
        ∂X₁₁[:, j] = sp_∂X₁₁j

        Mⱼ = differential_log_argument(M, pdsn.m, pdsn.p, Θⱼ)
        noPT = !hasproperty(obj, :Λ!) || ismissing(obj.Λ!)
        Kⱼ = pdsn.dual_stepsize * (
            noPT ? linearized_forward_operator(tmp, pdsn.m, Mⱼ, pdsn.n) : vector_transport_to(
                    N, forward_operator(tmp, pdsn.m), linearized_forward_operator(tmp, pdsn.m, Mⱼ, pdsn.n), pdsn.n,
                    pdsn.vector_transport_method,
                )
        )
        Jⱼ = get_differential_dual_prox(tmp, pdsn.n, pdsn.dual_stepsize, η₁, Kⱼ)
        ∂X₂₁j = get_coordinates(N, pdsn.n, -Jⱼ, DefaultOrthonormalBasis())

        sp_∂X₂₁j = sparsevec(∂X₂₁j)
        dropzeros!(sp_∂X₂₁j)
        ∂X₂₁[:, j] = sp_∂X₂₁j
    end

    # construct ∂X₁₂ and ∂X₂₂
    ∂X₁₂ = spzeros(dims, dualdims)
    ∂X₂₂ = spzeros(dualdims, dualdims)

    Ndims = prod(manifold_dimension(N))
    for j in 1:Ndims
        eⱼ = zeros(Ndims)
        eⱼ[j] = 1
        Ξⱼ = get_vector(N, pdsn.n, eⱼ, Ξ)
        hⱼ = -pdsn.primal_stepsize * adjoint_linearized_operator(tmp, pdsn.m, pdsn.n, Ξⱼ) # officially ∈ T*mM, but embedded in TmM
        Hⱼ = vector_transport_to(M, pdsn.m, hⱼ, pdsn.p)
        C₂ⱼ = differential_exp_argument(M, pdsn.p, q₃, Hⱼ)
        Bⱼ = get_differential_primal_prox(tmp, pdsn.primal_stepsize, q₂, C₂ⱼ)
        A₂ⱼ = -differential_log_argument(M, pdsn.p, q₁, Bⱼ)

        ∂X₁₂j = get_coordinates(M, pdsn.m, A₂ⱼ, DefaultOrthonormalBasis())

        sp_∂X₁₂j = sparsevec(∂X₁₂j)
        dropzeros!(sp_∂X₁₂j)
        ∂X₁₂[:, j] = sp_∂X₁₂j

        Jⱼ = get_differential_dual_prox(tmp, pdsn.n, pdsn.dual_stepsize, η₁, Ξⱼ)
        Iⱼ = Ξⱼ - Jⱼ

        ∂X₂₂j = get_coordinates(N, pdsn.n, Iⱼ, DefaultOrthonormalBasis())

        sp_∂X₂₂j = sparsevec(∂X₂₂j)
        dropzeros!(sp_∂X₂₂j)
        ∂X₂₂[:, j] = sp_∂X₂₂j
    end

    return [∂X₁₁ ∂X₁₂; ∂X₂₁ ∂X₂₂]
end
