@doc raw"""
    AdaptiveRegularizationState{P,T} <: AbstractHessianSolverState

A state for the [`adaptive_regularization_with_cubics`](@ref) solver.

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `η1`, `η2`:           (`0.1`, `0.9`) bounds for evaluating the regularization parameter
* `γ1`, `γ2`:           (`0.1`, `2.0`) shrinking and expansion factors for regularization parameter `σ`
* `p`:                  (`rand(M)` the current iterate
* `X`:                  (`zero_vector(M,p)`) the current gradient ``\operatorname{grad}f(p)``
* `s`:                  (`zero_vector(M,p)`) the tangent vector step resulting from minimizing the model
  problem in the tangent space ``\mathcal T_{p} \mathcal M``
* `σ`:                  the current cubic regularization parameter
* `σmin`:               (`1e-7`) lower bound for the cubic regularization parameter
* `ρ_regularization`:   (`1e3`) regularization parameter for computing ρ.
 When approaching convergence ρ may be difficult to compute with numerator and denominator approaching zero.
 Regularizing the ratio lets ρ go to 1 near convergence.
* `evaluation`:         (`AllocatingEvaluation()`) if you provide a
* `retraction_method`:  (`default_retraction_method(M)`) the retraction to use
* `stopping_criterion`: ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `sub_problem`:        sub problem solved in each iteration
* `sub_state`:          sub state for solving the sub problem, either a solver state if
  the problem is an [`AbstractManoptProblem`](@ref) or an [`AbstractEvaluationType`](@ref) if it is a function,
  where it defaults to [`AllocatingEvaluation`](@ref).

Furthermore the following integral fields are defined

* `q`:                  (`copy(M,p)`) a point for the candidates to evaluate model and ρ
* `H`:                  (`copy(M, p, X)`) the current Hessian, ``\operatorname{Hess}F(p)[⋅]``
* `S`:                  (`copy(M, p, X)`) the current solution from the subsolver
* `ρ`:                  the current regularized ratio of actual improvement and model improvement.
* `ρ_denominator`:      (`one(ρ)`) a value to store the denominator from the computation of ρ
  to allow for a warning or error when this value is non-positive.

# Constructor

    AdaptiveRegularizationState(M, p=rand(M); X=zero_vector(M, p); kwargs...)

Construct the solver state with all fields stated as keyword arguments.
"""
mutable struct AdaptiveRegularizationState{
    P,
    T,
    Pr<:Union{AbstractManoptProblem,<:Function},
    St<:Union{AbstractManoptSolverState,<:AbstractEvaluationType},
    TStop<:StoppingCriterion,
    R,
    TRTM<:AbstractRetractionMethod,
} <: AbstractManoptSolverState
    p::P
    X::T
    sub_problem::Pr
    sub_state::St
    q::P
    H::T
    S::T
    σ::R
    ρ::R
    ρ_denominator::R
    ρ_regularization::R
    stop::TStop
    retraction_method::TRTM
    σmin::R
    η1::R
    η2::R
    γ1::R
    γ2::R
end

function AdaptiveRegularizationState(
    M::AbstractManifold,
    p::P=rand(M),
    X::T=zero_vector(M, p);
    sub_objective=nothing,
    sub_problem::Pr=if isnothing(sub_objective)
        nothing
    else
        DefaultManoptProblem(TangentSpace(M, copy(M, p)), sub_objective)
    end,
    sub_state::St=if sub_problem isa Function
        AllocatingEvaluation()
    else
        LanczosState(TangentSpace(M, copy(M, p)))
    end,
    σ::R=100.0 / sqrt(manifold_dimension(M)),# Had this to initial value of 0.01. However try same as in MATLAB: 100/sqrt(dim(M))
    ρ_regularization::R=1e3,
    stopping_criterion::SC=StopAfterIteration(100),
    retraction_method::RTM=default_retraction_method(M),
    σmin::R=1e-10,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
) where {
    P,
    T,
    R,
    Pr<:Union{<:AbstractManoptProblem,<:Function,Nothing},
    St<:Union{<:AbstractManoptSolverState,<:AbstractEvaluationType},
    SC<:StoppingCriterion,
    RTM<:AbstractRetractionMethod,
}
    isnothing(sub_problem) && error("No sub_problem provided,")

    return AdaptiveRegularizationState{P,T,Pr,St,SC,R,RTM}(
        p,
        X,
        sub_problem,
        sub_state,
        copy(M, p),
        copy(M, p, X),
        copy(M, p, X),
        σ,
        one(σ),
        one(σ),
        ρ_regularization,
        stopping_criterion,
        retraction_method,
        σmin,
        η1,
        η2,
        γ1,
        γ2,
    )
end

get_iterate(s::AdaptiveRegularizationState) = s.p
function set_iterate!(s::AdaptiveRegularizationState, p)
    s.p = p
    return s
end
get_gradient(s::AdaptiveRegularizationState) = s.X
function set_gradient!(s::AdaptiveRegularizationState, X)
    s.X = X
    return s
end

function show(io::IO, arcs::AdaptiveRegularizationState)
    i = get_count(arcs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(arcs.stop) ? "Yes" : "No"
    sub = repr(arcs.sub_state)
    sub = replace(sub, "\n" => "\n    | ")
    s = """
    # Solver state for `Manopt.jl`s Adaptive Regularization with Cubics (ARC)
    $Iter
    ## Parameters
    * η1 | η2              : $(arcs.η1) | $(arcs.η2)
    * γ1 | γ2              : $(arcs.γ1) | $(arcs.γ2)
    * σ (σmin)             : $(arcs.σ) ($(arcs.σmin))
    * ρ (ρ_regularization) : $(arcs.ρ) ($(arcs.ρ_regularization))
    * retraction method    : $(arcs.retraction_method)
    * sub solver state     :
        | $(sub)

    ## Stopping criterion

    $(status_summary(arcs.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    adaptive_regularization_with_cubics(M, f, grad_f, Hess_f, p=rand(M); kwargs...)
    adaptive_regularization_with_cubics(M, f, grad_f, p=rand(M); kwargs...)
    adaptive_regularization_with_cubics(M, mho, p=rand(M); kwargs...)

Solve an optimization problem on the manifold `M` by iteratively minimizing

```math
  m_k(X) = f(p_k) + ⟨X, \operatorname{grad} f(p_k)⟩ + \frac{1}{2}⟨X, \operatorname{Hess} f(p_k)[X]⟩ + \frac{σ_k}{3}\lVert X \rVert^3
```

on the tangent space at the current iterate ``p_k``, where ``X ∈ T_{p_k}\mathcal M`` and
``σ_k > 0`` is a regularization parameter.

Let ``X_k`` denote the minimizer of the model ``m_k`` and use the model improvement

```math
  ρ_k = \frac{f(p_k) - f(\operatorname{retr}_{p_k}(X_k))}{m_k(0) - m_k(X_k) + \frac{σ_k}{3}\lVert X_k\rVert^3}.
```

With two thresholds ``η_2 ≥ η_1 > 0``
set ``p_{k+1} = \operatorname{retr}_{p_k}(X_k)`` if ``ρ ≥ η_1``
and reject the candidate otherwise, that is, set ``p_{k+1} = p_k``.

Further update the regularization parameter using factors ``0 < γ_1 < 1 < γ_2``

```math
σ_{k+1} =
\begin{cases}
    \max\{σ_{\min}, γ_1σ_k\} & \text{ if } ρ \geq η_2 &\text{   (the model was very successful)},\\
    σ_k & \text{ if } ρ ∈ [η_1, η_2)&\text{   (the model was successful)},\\
    γ_2σ_k & \text{ if } ρ < η_1&\text{   (the model was unsuccessful)}.
\end{cases}
```

For more details see [AgarwalBoumalBullinsCartis:2020](@cite).

# Input

* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad}F: \mathcal M → T \mathcal M`` of ``F``
* `Hess_f`: (optional) the Hessian ``H( \mathcal M, x, ξ)`` of ``F``
* `p`:      an initial value ``p ∈ \mathcal M``

For the case that no Hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

the cost `f` and its gradient and Hessian might also be provided as a [`ManifoldHessianObjective`](@ref)

# Keyword arguments

the default values are given in brackets

* `σ`:                      (`100.0 / sqrt(manifold_dimension(M)`) initial regularization parameter
* `σmin`:                   (`1e-10`) minimal regularization value ``σ_{\min}``
* `η1`:                     (`0.1`) lower model success threshold
* `η2`:                     (`0.9`) upper model success threshold
* `γ1`:                     (`0.1`) regularization reduction factor (for the success case)
* `γ2`:                     (`2.0`) regularization increment factor (for the non-success case)
* `evaluation`:             ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `grad_f(M, p)`
  or [`InplaceEvaluation`](@ref) in place, that is of the form `grad_f!(M, X, p)` and analogously for the Hessian.
* `retraction_method`:      (`default_retraction_method(M, typeof(p))`) a retraction to use
* `initial_tangent_vector`: (`zero_vector(M, p)`) initialize any tangent vector data,
* `maxIterLanczos`:         (`200`) a shortcut to set the stopping criterion in the sub solver,
* `ρ_regularization`:       (`1e3`) a regularization to avoid dividing by zero for small values of cost and model
* `stopping_criterion`:     ([`StopAfterIteration`](@ref)`(40) | `[`StopWhenGradientNormLess`](@ref)`(1e-9) | `[`StopWhenAllLanczosVectorsUsed`](@ref)`(maxIterLanczos)`)
* `sub_state`:              [`LanczosState`](@ref)`(M, copy(M, p); maxIterLanczos=maxIterLanczos, σ=σ)
  a state for the subproblem or an [`AbstractEvaluationType`](@ref) if the problem is a function.
* `sub_objective`:          a shortcut to modify the objective of the subproblem used within in the
* `sub_problem`:            [`DefaultManoptProblem`](@ref)`(M, sub_objective)` the problem (or a function) for the sub problem

All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.
If you provide the [`ManifoldGradientObjective`](@ref) directly, these decorations can still be specified

By default the `debug=` keyword is set to [`DebugIfEntry`](@ref)`(:ρ_denominator, >(0); message="Denominator nonpositive", type=:error)`
to avoid that by rounding errors the denominator in the computation of `ρ` gets nonpositive.
"""
adaptive_regularization_with_cubics(M::AbstractManifold, args...; kwargs...)

function adaptive_regularization_with_cubics(
    M::AbstractManifold, f, grad_f, Hess_f::TH; kwargs...
) where {TH<:Function}
    return adaptive_regularization_with_cubics(M, f, grad_f, Hess_f, rand(M); kwargs...)
end
function adaptive_regularization_with_cubics(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    Hess_f::THF,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF,THF}
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    Hess_f_ = _ensure_mutating_hessian(Hess_f, p, evaluation)
    mho = ManifoldHessianObjective(f_, grad_f_, Hess_f_; evaluation=evaluation)
    rs = adaptive_regularization_with_cubics(M, mho, p_; evaluation=evaluation, kwargs...)
    return _ensure_matching_output(p, rs)
end
function adaptive_regularization_with_cubics(M::AbstractManifold, f, grad_f; kwargs...)
    return adaptive_regularization_with_cubics(M, f, grad_f, rand(M); kwargs...)
end
function adaptive_regularization_with_cubics(
    M::AbstractManifold,
    f::TF,
    grad_f::TdF,
    p;
    evaluation=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
) where {TF,TdF}
    Hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation=evaluation, retraction_method=retraction_method
    )
    return adaptive_regularization_with_cubics(
        M,
        f,
        grad_f,
        Hess_f,
        p;
        evaluation=evaluation,
        retraction_method=retraction_method,
        kwargs...,
    )
end
function adaptive_regularization_with_cubics(
    M::AbstractManifold, mho::O, p=rand(M); kwargs...
) where {O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return adaptive_regularization_with_cubics!(M, mho, q; kwargs...)
end

@doc raw"""
    adaptive_regularization_with_cubics!(M, f, grad_f, Hess_f, p; kwargs...)
    adaptive_regularization_with_cubics!(M, f, grad_f, p; kwargs...)
    adaptive_regularization_with_cubics!(M, mho, p; kwargs...)

evaluate the Riemannian adaptive regularization with cubics solver in place of `p`.

# Input
* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad}F: \mathcal M → T \mathcal M`` of ``F``
* `Hess_f`: (optional) the Hessian ``H( \mathcal M, x, ξ)`` of ``F``
* `p`:      an initial value ``p  ∈  \mathcal M``

For the case that no Hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

the cost `f` and its gradient and Hessian might also be provided as a [`ManifoldHessianObjective`](@ref)

for more details and all options, see [`adaptive_regularization_with_cubics`](@ref).
"""
adaptive_regularization_with_cubics!(M::AbstractManifold, args...; kwargs...)
function adaptive_regularization_with_cubics!(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
)
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation=evaluation, retraction_method=retraction_method
    )
    return adaptive_regularization_with_cubics!(
        M,
        f,
        grad_f,
        hess_f,
        p;
        evaluation=evaluation,
        retraction_method=retraction_method,
        kwargs...,
    )
end
function adaptive_regularization_with_cubics!(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f::TH,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TH<:Function}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation)
    return adaptive_regularization_with_cubics!(M, mho, p; evaluation=evaluation, kwargs...)
end
function adaptive_regularization_with_cubics!(
    M::AbstractManifold,
    mho::O,
    p=rand(M);
    debug=DebugIfEntry(
        :ρ_denominator, >(-1e-8); message="denominator nonpositive", type=:error
    ),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    initial_tangent_vector::T=zero_vector(M, p),
    maxIterLanczos=min(300, manifold_dimension(M)),
    objective_type=:Riemannian,
    ρ_regularization::R=1e3,
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    σmin::R=1e-10,
    σ::R=100.0 / sqrt(manifold_dimension(M)),
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
    θ::R=0.5,
    sub_kwargs=(;),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(maxIterLanczos) |
                                              StopWhenFirstOrderProgress(θ),
    sub_state::Union{<:AbstractManoptSolverState,<:AbstractEvaluationType}=decorate_state!(
        LanczosState(
            TangentSpace(M, copy(M, p));
            maxIterLanczos=maxIterLanczos,
            σ=σ,
            θ=θ,
            stopping_criterion=sub_stopping_criterion,
            sub_kwargs...,
        );
        sub_kwargs,
    ),
    sub_objective=nothing,
    sub_problem=nothing,
    stopping_criterion::StoppingCriterion=if sub_state isa LanczosState
        StopAfterIteration(40) |
        StopWhenGradientNormLess(1e-9) |
        StopWhenAllLanczosVectorsUsed(maxIterLanczos - 1)
    else
        StopAfterIteration(40) | StopWhenGradientNormLess(1e-9)
    end,
    kwargs...,
) where {T,R,O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective}}
    dmho = decorate_objective!(M, mho; objective_type=objective_type, kwargs...)
    if isnothing(sub_objective)
        sub_objective = decorate_objective!(
            M, AdaptiveRagularizationWithCubicsModelObjective(dmho, σ); sub_kwargs...
        )
    end
    if isnothing(sub_problem)
        sub_problem = DefaultManoptProblem(TangentSpace(M, copy(M, p)), sub_objective)
    end
    X = copy(M, p, initial_tangent_vector)
    dmp = DefaultManoptProblem(M, dmho)
    arcs = AdaptiveRegularizationState(
        M,
        p,
        X;
        sub_state=sub_state,
        sub_problem=sub_problem,
        σ=σ,
        ρ_regularization=ρ_regularization,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        σmin=σmin,
        η1=η1,
        η2=η2,
        γ1=γ1,
        γ2=γ2,
    )
    darcs = decorate_state!(arcs; debug, kwargs...)
    solve!(dmp, darcs)
    return get_solver_return(get_objective(dmp), darcs)
end

function initialize_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState)
    get_gradient!(dmp, arcs.X, arcs.p)
    return arcs
end
function step_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState, i)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    # Update sub state
    # Set point also in the sub problem (eventually the tangent space)
    get_gradient!(M, arcs.X, mho, arcs.p)
    # Update base point in manifold
    set_manopt_parameter!(arcs.sub_problem, :Manifold, :p, copy(M, arcs.p))
    set_manopt_parameter!(arcs.sub_problem, :Objective, :σ, arcs.σ)
    set_iterate!(arcs.sub_state, M, copy(M, arcs.p, arcs.X))
    set_manopt_parameter!(arcs.sub_state, :σ, arcs.σ)
    #Solve the `sub_problem` via dispatch depending on type
    solve_arc_subproblem!(M, arcs.S, arcs.sub_problem, arcs.sub_state, arcs.p)
    # Compute ρ
    retract!(M, arcs.q, arcs.p, arcs.S, arcs.retraction_method)
    cost = get_cost(M, mho, arcs.p)
    ρ_num = cost - get_cost(M, mho, arcs.q)
    ρ_vec = arcs.X + 0.5 * get_hessian(M, mho, arcs.p, arcs.S)
    ρ_den = -inner(M, arcs.p, arcs.S, ρ_vec)
    ρ_reg = arcs.ρ_regularization * eps(Float64) * max(abs(cost), 1)
    arcs.ρ_denominator = ρ_den + ρ_reg # <= 0 -> the default debug kicks in
    arcs.ρ = (ρ_num + ρ_reg) / (ρ_den + ρ_reg)
    #Update iterate
    if arcs.ρ >= arcs.η1
        copyto!(M, arcs.p, arcs.q)
        get_gradient!(dmp, arcs.X, arcs.p) # only compute gradient when updating the point
    end
    # Update regularization parameter, for the last case between η1 and η2 keep it as is
    if arcs.ρ >= arcs.η2 #very successful, reduce
        arcs.σ = max(arcs.σmin, arcs.γ1 * arcs.σ)
    elseif arcs.ρ < arcs.η1 # unsuccessful
        arcs.σ = arcs.γ2 * arcs.σ
    end
    return arcs
end

# Dispatch on different forms of `sub_solvers`
function solve_arc_subproblem!(
    M, s, problem::P, state::S, p
) where {P<:AbstractManoptProblem,S<:AbstractManoptSolverState}
    solve!(problem, state)
    copyto!(M, s, p, get_solver_result(state))
    return s
end
function solve_arc_subproblem!(
    M, s, problem::P, ::AllocatingEvaluation, p
) where {P<:Function}
    copyto!(M, s, p, problem(M, p))
    return s
end
function solve_arc_subproblem!(
    M, s, problem!::P, ::InplaceEvaluation, p
) where {P<:Function}
    problem!(M, s, p)
    return s
end
