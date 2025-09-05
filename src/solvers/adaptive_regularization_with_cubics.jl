@doc """
    AdaptiveRegularizationState{P,T} <: AbstractHessianSolverState

A state for the [`adaptive_regularization_with_cubics`](@ref) solver.

# Fields

* `η1`, `η1`: bounds for evaluating the regularization parameter
* `γ1`, `γ2`:  shrinking and expansion factors for regularization parameter `σ`
* `H`: the current Hessian evaluation
* `s`: the current solution from the subsolver
$(_var(:Field, :p; add = [:as_Iterate]))
* `q`: a point for the candidates to evaluate model and ρ
$(_var(:Field, :X; add = [:as_Gradient]))
* `s`: the tangent vector step resulting from minimizing the model
  problem in the tangent space ``$(_math(:TpM))``
* `σ`: the current cubic regularization parameter
* `σmin`: lower bound for the cubic regularization parameter
* `ρ_regularization`: regularization parameter for computing ρ.
  When approaching convergence ρ may be difficult to compute with numerator and denominator approaching zero.
  Regularizing the ratio lets ρ go to 1 near convergence.
* `ρ`: the current regularized ratio of actual improvement and model improvement.
* `ρ_denominator`: a value to store the denominator from the computation of ρ
  to allow for a warning or error when this value is non-positive.
$(_var(:Field, :retraction_method))
$(_var(:Field, :stopping_criterion, "stop"))
$(_var(:Field, :sub_problem))
$(_var(:Field, :sub_state))

Furthermore the following integral fields are defined

# Constructor

    AdaptiveRegularizationState(M, sub_problem, sub_state; kwargs...)

Construct the solver state with all fields stated as keyword arguments and the following defaults

## Keyword arguments

* `η1=0.1`
* `η2=0.9`
* `γ1=0.1`
* `γ2=2.0`
* `σ=100/manifold_dimension(M)`
* `σmin=1e-7
* `ρ_regularization=1e3`
$(_var(:Keyword, :evaluation))
$(_var(:Keyword, :p))
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(100)`"))
$(_var(:Keyword, :X))
"""
mutable struct AdaptiveRegularizationState{
        P,
        T,
        Pr,
        St <: AbstractManoptSolverState,
        TStop <: StoppingCriterion,
        R,
        TRTM <: AbstractRetractionMethod,
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
        sub_problem::Pr,
        sub_state::St;
        p::P = rand(M),
        X::T = zero_vector(M, p),
        σ::R = 100.0 / sqrt(manifold_dimension(M)), # Had this to initial value of 0.01. However try same as in MATLAB: 100/sqrt(dim(M))
        ρ_regularization::R = 1.0e3,
        stopping_criterion::SC = StopAfterIteration(100),
        retraction_method::RTM = default_retraction_method(M, typeof(p)),
        σmin::R = 1.0e-10,
        η1::R = 0.1,
        η2::R = 0.9,
        γ1::R = 0.1,
        γ2::R = 2.0,
    ) where {
        P,
        T,
        R,
        Pr <: Union{<:AbstractManoptProblem, F} where {F},
        St <: AbstractManoptSolverState,
        SC <: StoppingCriterion,
        RTM <: AbstractRetractionMethod,
    }
    return AdaptiveRegularizationState{P, T, Pr, St, SC, R, RTM}(
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
function AdaptiveRegularizationState(
        M, sub_problem; evaluation::E = AllocatingEvaluation(), kwargs...
    ) where {E <: AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation = evaluation)
    return AdaptiveRegularizationState(M, sub_problem, cfs; kwargs...)
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

_doc_ARC_mdoel = raw"""
```math
m_k(X) = f(p_k) + ⟨X, \operatorname{grad} f(p^{(k)})⟩ + \frac{1}{2}⟨X, $(_tex(:Hess)) f(p^{(k)})[X]⟩ + \frac{σ_k}{3}\lVert X \rVert^3
```
"""

_doc_ARC_improvement = raw"""
```math
  ρ_k = \frac{f(p_k) - f(\operatorname{retr}_{p_k}(X_k))}{m_k(0) - m_k(X_k) + \frac{σ_k}{3}\lVert X_k\rVert^3}.
```
"""
_doc_ARC_regularization_update = raw"""
```math
σ_{k+1} =
\begin{cases}
    \max\{σ_{\min}, γ_1σ_k\} & \text{ if } ρ \geq η_2 &\text{   (the model was very successful)},\\
    σ_k & \text{ if } ρ ∈ [η_1, η_2)&\text{   (the model was successful)},\\
    γ_2σ_k & \text{ if } ρ < η_1&\text{   (the model was unsuccessful)}.
\end{cases}
```
"""

_doc_ARC = """
    adaptive_regularization_with_cubics(M, f, grad_f, Hess_f, p=rand(M); kwargs...)
    adaptive_regularization_with_cubics(M, f, grad_f, p=rand(M); kwargs...)
    adaptive_regularization_with_cubics(M, mho, p=rand(M); kwargs...)
    adaptive_regularization_with_cubics!(M, f, grad_f, Hess_f, p; kwargs...)
    adaptive_regularization_with_cubics!(M, f, grad_f, p; kwargs...)
    adaptive_regularization_with_cubics!(M, mho, p; kwargs...)

Solve an optimization problem on the manifold `M` by iteratively minimizing

$_doc_ARC_mdoel

on the tangent space at the current iterate ``p_k``, where ``X ∈ $(_math(:TpM; p = "p_k"))`` and
``σ_k > 0`` is a regularization parameter.

Let ``Xp^{(k)}`` denote the minimizer of the model ``m_k`` and use the model improvement

$_doc_ARC_improvement

With two thresholds ``η_2 ≥ η_1 > 0``
set ``p_{k+1} = $(_tex(:retr))_{p_k}(X_k)`` if ``ρ ≥ η_1``
and reject the candidate otherwise, that is, set ``p_{k+1} = p_k``.

Further update the regularization parameter using factors ``0 < γ_1 < 1 < γ_2`` reads

$_doc_ARC_regularization_update

For more details see [AgarwalBoumalBullinsCartis:2020](@cite).

# Input

$(_var(:Argument, :M; type = true))
$(_var(:Argument, :f))
$(_var(:Argument, :grad_f))
$(_var(:Argument, :Hess_f))
$(_var(:Argument, :p))

the cost `f` and its gradient and Hessian might also be provided as a [`ManifoldHessianObjective`](@ref)

# Keyword arguments

* `σ=100.0 / sqrt(manifold_dimension(M)`: initial regularization parameter
* `σmin=1e-10`: minimal regularization value ``σ_{\\min}``
* `η1=0.1`: lower model success threshold
* `η2=0.9`: upper model success threshold
* `γ1=0.1`: regularization reduction factor (for the success case)
* `γ2=2.0`: regularization increment factor (for the non-success case)
$(_var(:Keyword, :evaluation))
* `initial_tangent_vector=zero_vector(M, p)`: initialize any tangent vector data,
* `maxIterLanczos=200`: a shortcut to set the stopping criterion in the sub solver,
* `ρ_regularization=1e3`: a regularization to avoid dividing by zero for small values of cost and model
$(_var(:Keyword, :retraction_method)):
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(40)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-9)`$(_sc(:Any))[`StopWhenAllLanczosVectorsUsed`](@ref)`(maxIterLanczos)`"))
$(_var(:Keyword, :sub_kwargs))
* `sub_objective=nothing`: a shortcut to modify the objective of the subproblem used within in the `sub_problem=` keyword
  By default, this is initialized as a [`AdaptiveRagularizationWithCubicsModelObjective`](@ref), which can further be decorated by using the `sub_kwargs=` keyword.
$(_var(:Keyword, :sub_state; default = "[`LanczosState`](@ref)`(M, copy(M,p))`"))
$(_var(:Keyword, :sub_problem; default = "[`DefaultManoptProblem`](@ref)`(M, sub_objective)`"))

$(_note(:OtherKeywords))

If you provide the [`ManifoldFirstOrderObjective`](@ref) directly, the `evaluation=` keyword is ignored.
The decorations are still applied to the objective.

$(_note(:TutorialMode))

$(_note(:OutputSection))
"""

@doc "$_doc_ARC"
adaptive_regularization_with_cubics(M::AbstractManifold, args...; kwargs...)

function adaptive_regularization_with_cubics(
        M::AbstractManifold, f, grad_f, Hess_f::TH; kwargs...
    ) where {TH <: Function}
    return adaptive_regularization_with_cubics(M, f, grad_f, Hess_f, rand(M); kwargs...)
end
function adaptive_regularization_with_cubics(
        M::AbstractManifold,
        f::TF,
        grad_f::TDF,
        Hess_f::THF,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    ) where {TF, TDF, THF}
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    Hess_f_ = _ensure_mutating_hessian(Hess_f, p, evaluation)
    mho = ManifoldHessianObjective(f_, grad_f_, Hess_f_; evaluation = evaluation)
    rs = adaptive_regularization_with_cubics(M, mho, p_; evaluation = evaluation, kwargs...)
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
        evaluation = AllocatingEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        kwargs...,
    ) where {TF, TdF}
    Hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation = evaluation, retraction_method = retraction_method
    )
    return adaptive_regularization_with_cubics(
        M,
        f,
        grad_f,
        Hess_f,
        p;
        evaluation = evaluation,
        retraction_method = retraction_method,
        kwargs...,
    )
end
function adaptive_regularization_with_cubics(
        M::AbstractManifold, mho::O, p = rand(M); kwargs...
    ) where {O <: Union{ManifoldHessianObjective, AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return adaptive_regularization_with_cubics!(M, mho, q; kwargs...)
end

@doc "$_doc_ARC"
adaptive_regularization_with_cubics!(M::AbstractManifold, args...; kwargs...)
function adaptive_regularization_with_cubics!(
        M::AbstractManifold,
        f,
        grad_f,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        kwargs...,
    )
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation = evaluation, retraction_method = retraction_method
    )
    return adaptive_regularization_with_cubics!(
        M,
        f,
        grad_f,
        hess_f,
        p;
        evaluation = evaluation,
        retraction_method = retraction_method,
        kwargs...,
    )
end
function adaptive_regularization_with_cubics!(
        M::AbstractManifold,
        f,
        grad_f,
        Hess_f::TH,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    ) where {TH <: Function}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f; evaluation = evaluation)
    return adaptive_regularization_with_cubics!(M, mho, p; evaluation = evaluation, kwargs...)
end
function adaptive_regularization_with_cubics!(
        M::AbstractManifold,
        mho::O,
        p = rand(M);
        debug = if is_tutorial_mode()
            DebugIfEntry(
                :ρ_denominator, >(-1.0e-8); message = "denominator nonpositive", type = :error
            )
        else
            []
        end,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        initial_tangent_vector::T = zero_vector(M, p),
        maxIterLanczos = min(300, manifold_dimension(M)),
        objective_type = :Riemannian,
        ρ_regularization::R = 1.0e3,
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        σmin::R = 1.0e-10,
        σ::R = 100.0 / sqrt(manifold_dimension(M)),
        η1::R = 0.1,
        η2::R = 0.9,
        γ1::R = 0.1,
        γ2::R = 2.0,
        θ::R = 0.5,
        sub_kwargs = (;),
        sub_stopping_criterion::StoppingCriterion = StopAfterIteration(maxIterLanczos) |
            StopWhenFirstOrderProgress(θ),
        sub_state::Union{<:AbstractManoptSolverState, <:AbstractEvaluationType} = decorate_state!(
            LanczosState(
                TangentSpace(M, copy(M, p));
                maxIterLanczos = maxIterLanczos,
                σ = σ,
                θ = θ,
                stopping_criterion = sub_stopping_criterion,
                sub_kwargs...,
            );
            sub_kwargs,
        ),
        sub_objective = nothing,
        sub_problem = nothing,
        stopping_criterion::StoppingCriterion = if sub_state isa LanczosState
            StopAfterIteration(40) |
                StopWhenGradientNormLess(1.0e-9) |
                StopWhenAllLanczosVectorsUsed(maxIterLanczos - 1)
        else
            StopAfterIteration(40) | StopWhenGradientNormLess(1.0e-9)
        end,
        kwargs...,
    ) where {T, R, O <: Union{ManifoldHessianObjective, AbstractDecoratedManifoldObjective}}
    dmho = decorate_objective!(M, mho; objective_type = objective_type, kwargs...)
    if isnothing(sub_objective)
        sub_objective = decorate_objective!(
            M, AdaptiveRagularizationWithCubicsModelObjective(dmho, σ); sub_kwargs...
        )
    end
    if isnothing(sub_problem)
        sub_problem = DefaultManoptProblem(TangentSpace(M, copy(M, p)), sub_objective)
    end
    sub_state_storage = maybe_wrap_evaluation_type(sub_state)
    X = copy(M, p, initial_tangent_vector)
    dmp = DefaultManoptProblem(M, dmho)
    arcs = AdaptiveRegularizationState(
        M,
        sub_problem,
        sub_state_storage;
        p = p,
        X = X,
        σ = σ,
        ρ_regularization = ρ_regularization,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        σmin = σmin,
        η1 = η1,
        η2 = η2,
        γ1 = γ1,
        γ2 = γ2,
    )
    darcs = decorate_state!(arcs; debug, kwargs...)
    solve!(dmp, darcs)
    return get_solver_return(get_objective(dmp), darcs)
end

function initialize_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState)
    get_gradient!(dmp, arcs.X, arcs.p)
    return arcs
end
function step_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState, k)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    # Update sub state
    # Set point also in the sub problem (eventually the tangent space)
    get_gradient!(M, arcs.X, mho, arcs.p)
    # Update base point in manifold
    set_parameter!(arcs.sub_problem, :Manifold, :p, copy(M, arcs.p))
    set_parameter!(arcs.sub_problem, :Objective, :σ, arcs.σ)
    set_iterate!(arcs.sub_state, M, copy(M, arcs.p, arcs.X))
    set_parameter!(arcs.sub_state, :σ, arcs.σ)
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
    ) where {P <: AbstractManoptProblem, S <: AbstractManoptSolverState}
    solve!(problem, state)
    copyto!(M, s, p, get_solver_result(state))
    return s
end
function solve_arc_subproblem!(
        M, s, problem::P, ::ClosedFormSubSolverState{AllocatingEvaluation}, p
    ) where {P <: Function}
    copyto!(M, s, p, problem(M, p))
    return s
end
function solve_arc_subproblem!(
        M, s, problem!::P, ::ClosedFormSubSolverState{InplaceEvaluation}, p
    ) where {P <: Function}
    problem!(M, s, p)
    return s
end
