@doc raw"""
    AdaptiveRegularizationState{P,T} <: AbstractHessianSolverState

A state for the [`adaptive_regularization_with_cubics`](@ref) solver.

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `η1`, `η2`           – (`0.1`, `0.9`) bounds for evaluating the regularization parameter
* `γ1`, `γ2`           – (`0.1`, `2.0`) shrinking and exansion factors for regularization parameter `σ`
* `p`                  – (`rand(M)` the current iterate
* `X`                  – (`zero_vector(M,p)`) the current gradient ``\operatorname{grad}f(p)``
* `s`                  - (`zero_vector(M,p)`) the tangent vector step resulting from minimizing the model
  problem in the tangent space ``\mathcal T_{p} \mathcal M``
* `σ`                 – the current cubic regularization parameter
* `σmin`               – (`1e-7`) lower bound for the cubic regularization parameter
* `ρ_regularization`   – (1e3) regularization paramter for computing ρ. As we approach convergence the ρ may be difficult to compute with numerator and denominator approachign zero. Regularizing the the ratio lets ρ go to 1 near convergence.
* `evaluation`         - (`AllocatingEvaluation()`) if you provide a
* `retraction_method`  – (`default_retraction_method(M)`) the retraction to use
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `sub_problem`        - sub problem solved in each iteration
* `sub_state`          - sub state for solving the sub problem – either a solver state if
                         the problem is an [`AbstractManoptProblem`](@ref) or an [`AbstractEvaluationType`](@ref) if it is a function,
                         where it defaults to [`AllocatingEvaluation`](@ref).

Furthermore the following interal fields are defined

* `q`                  - (`copy(M,p)`) a point for the candidates to evaluate model and ρ
* `H`                  – (`copy(M, p, X)`) the current hessian, ``\operatorname{Hess}F(p)[⋅]``
* `S`                  – (`copy(M, p, X)`) the current solution from the subsolver
* `ρ`                  – the current regularized ratio of actual improvement and model improvement.
* `ρ_denominator`      – (`one(ρ)`) a value to store the denominator from the computation of ρ
                         to allow for a warning or error when this value is non-positive.

# Constructor

    AdaptiveRegularizationState(M, p=rand(M); X=zero_vector(M, p); kwargs...)

Construct the solver state with all fields stated above as keyword arguments.
"""
mutable struct AdaptiveRegularizationState{
    P,
    T,
    Pr<:AbstractManoptProblem,
    St<:AbstractManoptSolverState,
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
    ρ_denonimator::R
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
        DefaultManoptProblem(M, sub_objective)
    end,
    sub_state::St=sub_problem isa Function ? AllocatingEvaluation() : LanczosState(M, p),
    σ::R=100.0 / sqrt(manifold_dimension(M)),# Had this to inital value of 0.01. However try same as in MATLAB: 100/sqrt(dim(M))
    ρ_regularization::R=1e3,
    stopping_criterion::SC=StopAfterIteration(100),
    retraction_method::RTM=default_retraction_method(M),
    σmin::R=1e-10, #Set the below to appropriate default vals.
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

    ## Stopping Criterion
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

on the tangent space at the current iterate ``p_k``, i.e. ``X ∈ T_{p_k}\mathcal M`` and
where ``σ_k > 0`` is a regularization parameter.

Let ``X_k`` denote the minimizer of the model ``m_k``, then we use the model improvement

```math
ρ_k = \frac{f(p_k) - f(\operatorname{retr}_{p_k}(X_k))}{m_k(0) - m_k(s) + \frac{σ_k}{3}\lVert X_k\rVert^3}.
```

We use two thresholds ``η_2 ≥ η_1 > 0`` and set
``p_{k+1} = \operatorname{retr}_{p_k}(X_k)`` if ``ρ ≥ η_1`` and reject the candidate otherwise, i.e. set ``p_{k+1} = p_k``.

We further update the regularozation parameter using factors ``0 < γ_1 < 1 < γ_2``

```math
σ_{k+1} =
\begin{cases}
    \max\{σ_{\min}, γ_1σ_k\} & \text{ if } ρ \geq η_2 &\text{   (the model was very successful)},\\
    σ_k & \text{ if } ρ \in [η_1, η_2)&\text{   (the model was succesful)},\\
    γ_2σ_k & \text{ if } ρ < η_1&\text{   (the model was unsuccesful)}.
\end{cases}
```

For more details see [Agarwal, Boumal, Bullins, Cartis, Math. Prog., 2020](@cite AgarwalBoumalBullinsCartis:2020).

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f`- the gradient ``\operatorname{grad}F: \mathcal M → T \mathcal M`` of ``F``
* `Hess_f` – (optional) the hessian ``H( \mathcal M, x, ξ)`` of ``F``
* `p` – an initial value ``p  ∈  \mathcal M``

For the case that no hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

the cost `f` and its gradient and hessian might also be provided as a [`ManifoldHessianObjective`](@ref)

# Keyword arguments
the default values are given in brackets

* `σ`                      - (`100.0 / sqrt(manifold_dimension(M)`) initial regularization parameter
* `σmin`                   - (`1e-10`) minimal regularization value ``σ_{\min}``
* `η1`                     - (`0.1`) lower model success threshold
* `η2`                     - (`0.9`) upper model success threshold
* `γ1`                     - (`0.1`) regularization reduction factor (for the success case)
* `γ2`                     - (`2.0`) regularization increment factor (for the non-success case)
* `evaluation`             – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `grad_f(M, p)`
                             or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `grad_f!(M, X, p)` and analogously for the hessian.
* `retraction_method`      – (`default_retraction_method(M, typeof(p))`) a retraction to use
* `initial_tangent_vector` - (`zero_vector(M, p)`) initialize any tangent vector data,
* `maxIterLanczos`         - (`200`) a shortcut to set the stopping criterion in the sub_solver,
* `ρ_regularization`       - (`1e3`) a regularization to avoid dividing by zero for small values of cost and model
* `stopping_criterion`     - ([`StopAfterIteration`](@ref)`(40) | `[`StopWhenGradientNormLess`](@ref)`(1e-9) | `[`StopWhenAllLanczosVectorsUsed`](@ref)`(maxIterLanczos)`)
* `sub_state`              - [`LanczosState`](@ref)`(M, copy(M, p); maxIterLanczos=maxIterLanczos, σ=σ)
                             a state for the subproblem or an [`AbstractEvaluationType`](@ref) if the problem is a funtion.
* `sub_objective`               - a shortcut to modify the objective of the subproblem used within in the
* `sub_problem`            - [`DefaultManoptProblem`](@ref)`(M, sub_objective)` the problem (or a function) for the sub problem

All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.
If you provide the [`ManifoldGradientObjective`](@ref) directly, these decorations can still be specified

By default the `debug=` keyword is set to [`DebugIfEntry`](@ref)`(:ρ_denonimator, >(0); message="Denominator nonpositive", type=:error)``
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
    mho = ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation)
    return adaptive_regularization_with_cubics(M, mho, p; evaluation=evaluation, kwargs...)
end
function adaptive_regularization_with_cubics(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    Hess_f::THF,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF,THF}
    q = [p]
    f_(M, p) = f(M, p[])
    Hess_f_ = Hess_f
    # For now we can not update the gradient within the ApproxHessian so the filled default
    # Hessian fails here
    if evaluation isa AllocatingEvaluation
        grad_f_ = (M, p) -> [grad_f(M, p[])]
        Hess_f_ = (M, p, X) -> [Hess_f(M, p[], X[])]
    else
        grad_f_ = (M, X, p) -> (X .= [grad_f(M, p[])])
        Hess_f_ = (M, Y, p, X) -> (Y .= [Hess_f(M, p[], X[])])
    end
    rs = adaptive_regularization_with_cubics(
        M, f_, grad_f_, Hess_f_, q; evaluation=evaluation, kwargs...
    )
    return (typeof(q) == typeof(rs)) ? rs[] : rs
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
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f`- the gradient ``\operatorname{grad}F: \mathcal M → T \mathcal M`` of ``F``
* `Hess_f` – (optional) the hessian ``H( \mathcal M, x, ξ)`` of ``F``
* `p` – an initial value ``p  ∈  \mathcal M``

For the case that no hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

the cost `f` and its gradient and hessian might also be provided as a [`ManifoldHessianObjective`](@ref)

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
        :ρ_denonimator, >(0); message="Denominator nonpositive", type=:error
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
    sub_kwargs=[],
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(maxIterLanczos) |
                                              StopWhenFirstOrderProgress(θ),
    sub_state::Union{<:AbstractManoptSolverState,<:AbstractEvaluationType}=LanczosState(
        M,
        copy(M, p);
        maxIterLanczos=maxIterLanczos,
        σ=σ,
        θ=θ,
        stopping_criterion=sub_stopping_criterion,
    ),
    sub_objective=decorate_objective!(M, mho; objective_type=objective_type, sub_kwargs...),
    sub_problem=DefaultManoptProblem(M, sub_objective),
    stopping_criterion::StoppingCriterion=if sub_state isa LanczosState
        StopAfterIteration(40) |
        StopWhenGradientNormLess(1e-9) |
        StopWhenAllLanczosVectorsUsed(maxIterLanczos - 1)
    else
        StopAfterIteration(40) | StopWhenGradientNormLess(1e-9)
    end,
    kwargs...,
) where {T,R,O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective}}
    X = copy(M, p, initial_tangent_vector)
    dmho = decorate_objective!(M, mho; objective_type=objective_type, kwargs...)
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
    set_manopt_parameter!(arcs.sub_problem, :Objective, :Cost, :X, copy(M, arcs.p, arcs.X))
    set_manopt_parameter!(arcs.sub_problem, :Objective, :Cost, :σ, arcs.σ)
    set_manopt_parameter!(
        arcs.sub_problem, :Objective, :Gradient, :X, copy(M, arcs.p, arcs.X)
    )
    set_manopt_parameter!(arcs.sub_problem, :Objective, :Gradient, :σ, arcs.σ)
    set_iterate!(arcs.sub_state, M, copy(M, arcs.p, arcs.X))
    set_manopt_parameter!(arcs.sub_state, :σ, arcs.σ)
    set_manopt_parameter!(arcs.sub_state, :p, copy(M, arcs.p))
    #Solve the sub_problem – via dispatch depending on type
    solve_arc_subproblem!(M, arcs.S, arcs.sub_problem, arcs.sub_state, arcs.p)
    # Compute ρ
    retract!(M, arcs.q, arcs.p, arcs.S, arcs.retraction_method)
    cost = get_cost(M, mho, arcs.p)
    ρ_num = cost - get_cost(M, mho, arcs.q)
    ρ_vec = arcs.X + 0.5 * get_hessian(M, mho, arcs.p, arcs.S)
    ρ_den = -inner(M, arcs.p, arcs.S, ρ_vec)
    ρ_reg = arcs.ρ_regularization * eps(Float64) * max(abs(cost), 1)
    arcs.ρ_denonimator = ρ_den + ρ_reg # <= 0 -> the default debug kicks in
    arcs.ρ = (ρ_num + ρ_reg) / (ρ_den + ρ_reg)

    #Update iterate
    if arcs.ρ >= arcs.η1
        copyto!(M, arcs.p, arcs.q)
        get_gradient!(dmp, arcs.X, arcs.p) #only compute gradient when we update the point
    end
    #Update regularization parameter - in the mid interval between η1 and η2 we leave it as is
    if arcs.ρ >= arcs.η2 #very successful, reduce
        arcs.σ = max(arcs.σmin, arcs.γ1 * arcs.σ)
    elseif arcs.ρ < arcs.η1 # unsuccessful
        arcs.σ = arcs.γ2 * arcs.σ
    end
    return arcs
end

# Dispatch on different forms of sub_solvers
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

#
# Lanczos sub solver
#

@doc raw"""
    LanczosState{P,T,SC,B,I,R,TM,V,Y} <: AbstractManoptSolverState

Solve the adaptive regularized subproblem with a Lanczos iteration

# Fields

* `p` the current iterate
* `stop` – the stopping criterion
* `σ` – the current regularization parameter
* `X` the current gradient
* `Lanczos_vectors` – the obtained Lanczos vectors
* `tridig_matrix` the tridigonal coefficient matrix T
* `coefficients` the coefficients `y_1,...y_k`` that deteermine the solution
* `Hp` – a temporary vector containing the evaluation of the Hessian
* `Hp_residual` – a temporary vector containing the residual to the Hessian
* `S` – the current obtained / approximated solution
"""
mutable struct LanczosState{P,T,R,SC,SCN,B,TM,C} <: AbstractManoptSolverState
    p::P
    X::T
    σ::R
    stop::SC           # Notation in ABBC
    stop_newton::SCN
    Lanczos_vectors::B # qi
    tridig_matrix::TM  # T
    coefficients::C     # y
    Hp::T              # Hess_f A temporary vector for evaluations of the hessian
    Hp_residual::T     # A residual vector
    # Maybe not necessary?
    S::T               # store the tangent vector that solves the minimization problem
end
function LanczosState(
    M::AbstractManifold,
    p::P=rand(M);
    X::T=zero_vector(M, p),
    maxIterLanczos=200,
    θ=0.5,
    stopping_criterion::SC=StopAfterIteration(maxIterLanczos) |
                           StopWhenFirstOrderProgress(θ),
    stopping_criterion_newtown::SCN=StopAfterIteration(200),
    σ::R=10.0,
) where {P,T,SC<:StoppingCriterion,SCN<:StoppingCriterion,R}
    tridig = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    coeffs = zeros(maxIterLanczos)
    Lanczos_vectors = typeof(X)[]
    return LanczosState{P,T,R,SC,SCN,typeof(Lanczos_vectors),typeof(tridig),typeof(coeffs)}(
        p,
        X,
        σ,
        stopping_criterion,
        stopping_criterion_newtown,
        Lanczos_vectors,
        tridig,
        coeffs,
        copy(M, p, X),
        copy(M, p, X),
        copy(M, p, X),
    )
end
function get_solver_result(ls::LanczosState)
    return ls.S
end
function set_manopt_parameter!(ls::LanczosState, ::Val{:p}, p)
    ls.p = p
    return ls
end
function set_iterate!(ls::LanczosState, M, X)
    ls.X = X
    return ls
end
function set_manopt_parameter!(ls::LanczosState, ::Val{:σ}, σ)
    ls.σ = σ
    return ls
end

function show(io::IO, ls::LanczosState)
    i = get_count(ls, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ls.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Lanczos Iteration
    $Iter
    ## Parameters
    * σ                         : $(ls.σ)
    * # of Lanczos vectors used : $(length(ls.Lanczos_vectors))

    ## Stopping Criteria
    (a) For the Lanczos Iteration
    $(status_summary(ls.stop))
    (b) For the Newton sub solver
    $(status_summary(ls.stop_newton))
    This indicates convergence: $Conv"""
    return print(io, s)
end

#
# The Lanczos Subsolver implementation
#
function initialize_solver!(dmp::AbstractManoptProblem, ls::LanczosState)
    M = get_manifold(dmp)
    # Maybe better to allocate once and just reset the number of vectors k?
    maxIterLanczos = size(ls.tridig_matrix, 1)
    ls.tridig_matrix = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    ls.coefficients = zeros(maxIterLanczos)
    for X in ls.Lanczos_vectors
        zero_vector!(M, X, ls.p)
    end
    zero_vector!(M, ls.Hp, ls.p)
    zero_vector!(M, ls.Hp_residual, ls.p)
    return ls
end

function step_solver!(dmp::AbstractManoptProblem, ls::LanczosState, i)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    if i == 1 #we can easily compute the first Lanczos vector
        nX = norm(M, ls.p, ls.X)
        if length(ls.Lanczos_vectors) == 0
            push!(ls.Lanczos_vectors, ls.X ./ nX)
        else
            copyto!(M, ls.Lanczos_vectors[1], ls.p, ls.X ./ nX)
        end
        get_hessian!(M, ls.Hp, mho, ls.p, ls.Lanczos_vectors[1])
        α = inner(M, ls.p, ls.Lanczos_vectors[1], ls.Hp)
        # This is also the first coefficient in the tridigianoal matrix
        ls.tridig_matrix[1, 1] = α
        ls.Hp_residual .= ls.Hp - α * ls.Lanczos_vectors[1]
        #argmin of one dimensional model
        ls.coefficients[1] = (α - sqrt(α^2 + 4 * ls.σ * nX)) / (2 * ls.σ)
    else # i > 1
        β = norm(M, ls.p, ls.Hp_residual)
        if β > 1e-12 # Obtained new orth Lanczos long enough cf. to num stability
            if length(ls.Lanczos_vectors) < i
                push!(ls.Lanczos_vectors, ls.Hp_residual ./ β)
            else
                copyto!(M, ls.Lanczos_vectors[i], ls.p, ls.Hp_residual ./ β)
            end
        else # Generate new random vec and MGS of new vec wrt. Q
            rand!(M, ls.Hp_residual; vector_at=ls.p)
            for k in 1:(i - 1)
                ls.Hp_residual .=
                    ls.Hp_residual -
                    inner(M, ls.p, ls.Lanczos_vectors[k], ls.Hp_residual) *
                    ls.Lanczos_vectors[k]
            end
            if length(ls.Lanczos_vectors) < i
                push!(ls.Lanczos_vectors, ls.Hp_residual ./ norm(M, ls.p, ls.Hp_residual))
            else
                copyto!(
                    M,
                    ls.Lanczos_vectors[i],
                    ls.p,
                    ls.Hp_residual ./ norm(M, ls.p, ls.Hp_residual),
                )
            end
        end
        # Update Hessian and residual
        get_hessian!(M, ls.Hp, mho, ls.p, ls.Lanczos_vectors[i])
        ls.Hp_residual .= ls.Hp - β * ls.Lanczos_vectors[i - 1]
        α = inner(M, ls.p, ls.Hp_residual, ls.Lanczos_vectors[i])
        ls.Hp_residual .= ls.Hp_residual - α * ls.Lanczos_vectors[i]
        # Update tridiagonal matric
        ls.tridig_matrix[i, i] = α
        ls.tridig_matrix[i - 1, i] = β
        ls.tridig_matrix[i, i - 1] = β
        min_cubic_Newton!(dmp, ls, i)
    end
    copyto!(M, ls.S, ls.p, sum(ls.Lanczos_vectors[k] * ls.coefficients[k] for k in 1:i))
    return ls
end
#
# Solve Lanczos sub problem
#
function min_cubic_Newton!(mp::AbstractManoptProblem, ls::LanczosState, i)
    M = get_manifold(mp)
    tol = 1e-16 # TODO: Put into a stopping criterion

    gvec = zeros(i)
    gvec[1] = norm(M, ls.p, ls.X)
    λ = opnorm(Array(@view ls.tridig_matrix[1:i, 1:i])) + 2
    T_λ = @view(ls.tridig_matrix[1:i, 1:i]) + λ * I

    λ_min = eigmin(Array(@view ls.tridig_matrix[1:i, 1:i]))
    lower_barrier = max(0, -λ_min)
    k = 0
    y = zeros(i)
    while !ls.stop_newton(mp, ls, k)
        k += 1
        y = -(T_λ \ gvec)
        ynorm = norm(y, 2)
        ϕ = 1 / ynorm - ls.σ / λ #when ϕ is "zero", y is the solution.
        (abs(ϕ) < tol * ynorm) && break
        #compute the newton step
        ψ = ynorm^2
        Δy = -(T_λ) \ y
        ψ_prime = 2 * dot(y, Δy)
        # Quadratic polynomial coefficients
        p0 = 2 * ls.σ * ψ^(1.5)
        p1 = -2 * ψ - λ * ψ_prime
        p2 = ψ_prime
        #Polynomial roots
        r1 = (-p1 + sqrt(p1^2 - 4 * p2 * p0)) / (2 * p2)
        r2 = (-p1 - sqrt(p1^2 - 4 * p2 * p0)) / (2 * p2)

        Δλ = max(r1, r2) - λ

        #if we jumped past the lower barrier for λ, jump to midpoint between current and lower λ.
        (λ + Δλ <= lower_barrier) && (Δλ = -0.5 * (λ - lower_barrier))
        #if the steps we make are to small, terminate
        (abs(Δλ) <= eps(λ)) && break
        T_λ = T_λ + Δλ * I
        λ = λ + Δλ
    end
    ls.coefficients[1:i] .= y
    return ls.coefficients
end

#
# Stopping Criteria
#
@doc raw"""
    StopWhenFirstOrderProgress <: StoppingCriterion

A stopping criterion related to the Riemannian adaptive regularization with cubics (ARC)
solver indicating that the model function at the current (outer) iterate, i.e.

```math
    m(X) = f(p) + <X, \operatorname{grad}f(p)>
      + \frac{1}{2} <X, \operatorname{Hess} f(p)[X]> +  \frac{σ}{3} \lVert X \rVert^3,
```

defined on the tangent space ``T_{p}\mathcal M``
fulfills at the current iterate ``X_k`` that

```math
m(X_k) \leq m(0)
\quad\text{ and }\quad
\lVert \operatorname{grad} m(X_k) \rVert ≤ θ \lVert X_k \rVert^2
```

# Fields

* `θ` – the factor ``θ`` in the second condition above
* `reason` – a String indicating the reason if the criterion indicated to stop
"""
mutable struct StopWhenFirstOrderProgress <: StoppingCriterion
    θ::Float64 #θ
    reason::String
    StopWhenFirstOrderProgress(θ::Float64) = new(θ, "")
end
function (c::StopWhenFirstOrderProgress)(
    dmp::AbstractManoptProblem, ls::LanczosState, i::Int
)
    if (i == 0)
        c.reason = ""
        return false
    end
    #Update Gradient
    M = get_manifold(dmp)
    get_gradient!(dmp, ls.X, ls.p)
    nX = norm(M, ls.p, ls.X)
    y = @view(ls.coefficients[1:(i - 1)])
    Ty = @view(ls.tridig_matrix[1:i, 1:(i - 1)]) * y
    ny = norm(y)
    model_grad_norm = norm(nX .* [1, zeros(i - 1)...] + Ty + ls.σ * ny * [y..., 0])
    if (i > 0) && (model_grad_norm <= c.θ * ny^2)
        c.reason = "The subproblem has reached a point with ||grad m(X)|| ≤ θ ||X||^2, θ = $(c.θ)."
        return true
    end
    return false
end
function status_summary(c::StopWhenFirstOrderProgress)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "First order progress with θ=$(c.θ):\t$s"
end
indicates_convergence(c::StopWhenFirstOrderProgress) = true
function show(io::IO, c::StopWhenFirstOrderProgress)
    return print(io, "StopWhenFirstOrderProgress($(repr(c.θ)))\n    $(status_summary(c))")
end

#A new stopping criterion that deals with the scenario when a step needs more Lanczos vectors than preallocated.
#Previously this would just cause an error due to out of bounds error. So this stopping criterion deals both with the scenario
#of too few allocated vectors and stagnation in the solver.
@doc raw"""
    StopWhenAllLanczosVectorsUsed <: StoppingCriterion

When an inner iteration has used up all Lanczos vectors, then this stoping crtierion is
a fallback / security stopping criterion in order to not access a non-existing field
in the array allocated for vectors.

Note that this stopping criterion (for now) is only implemented for the case that an
[`AdaptiveRegularizationState`](@ref) when using a [`LanczosState`](@ref) subsolver

# Fields

* `maxLanczosVectors` – maximal number of Lanczos vectors
* `reason` – a String indicating the reason if the criterion indicated to stop

# Constructor

    StopWhenAllLanczosVectorsUsed(maxLancosVectors::Int)

"""
mutable struct StopWhenAllLanczosVectorsUsed <: StoppingCriterion
    maxLanczosVectors::Int
    reason::String
    StopWhenAllLanczosVectorsUsed(maxLanczosVectors::Int) = new(maxLanczosVectors, "")
end
function (c::StopWhenAllLanczosVectorsUsed)(
    ::AbstractManoptProblem,
    arcs::AdaptiveRegularizationState{P,T,Pr,<:LanczosState},
    i::Int,
) where {P,T,Pr}
    (i == 0) && (c.reason = "") # reset on init
    if (i > 0) && length(arcs.sub_state.Lanczos_vectors) == c.maxLanczosVectors
        c.reason = "The algorithm used all ($(c.maxLanczosVectors)) preallocated Lanczos vectors and may have stagnated.\n Consider increasing this value.\n"
        return true
    end
    return false
end
function status_summary(c::StopWhenAllLanczosVectorsUsed)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "All Lanczos vectors ($(c.maxLanczosVectors)) used:\t$s"
end
indicates_convergence(c::StopWhenAllLanczosVectorsUsed) = false
function show(io::IO, c::StopWhenAllLanczosVectorsUsed)
    return print(
        io,
        "StopWhenAllLanczosVectorsUsed($(repr(c.maxLanczosVectors)))\n    $(status_summary(c))",
    )
end
