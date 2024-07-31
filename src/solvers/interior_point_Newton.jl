@doc raw"""
    interior_point_Newton(M, f,. grad_f, Hess_f, p=rand(M); kwargs...)

perform the interior point Newton method following [LaiYoshise:2024](@cite).

In order to solve the constrained problem

```math
\begin{aligned}
\min_{p ∈\mathcal{M}} &f(p)\\
\text{subject to } &g_i(p)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(p)=0 \quad \text{ for } j=1,…,n,
\end{aligned}
```

This algorithms iteratively solves the linear system based on extending the KKT system
by a slack variable `s`.

```math
\operatorname{J} F(p, μ, λ, s)[X, Y, Z, W] = -F(p, μ, λ, s),
\text{ where }
X ∈ T_p\mathcal M, Y,W ∈ ℝ^m, Z ∈ ℝ^n,
```
see [`CondensedKKTVectorFieldJacobian`](@ref) and [`CondensedKKTVectorField`](@ref), respectively,
for the reduced form, this is usually solved in.
From the resulting `X` and `Z` in the reeuced form, the other two can be computed.

Note that since the vector field ``F`` includes the gradients of the constraint
functions ``g,h`, its gradient or Jacobian requires the Hessians of the constraints.

For that seach direction a line search is performed, that additionally ensures that
the constraints are further fulfilled.

# Input

* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``f : \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad} f : \mathcal M → T \mathcal M`` of ``f``
* `Hess_f`: the Hessian ``\operatorname{Hess}f(p): T_p\mathcal M → T_p\mathcal M``, ``X ↦ \operatorname{Hess}F(p)[X] = ∇_X\operatorname{grad}f(p)``
* `p=`[`rand`](@extref Base.rand-Tuple{AbstractManifold})`(M)`: an initial value ``p  ∈  \mathcal M``

# Keyword arguments

* `equality_constraints=nothing`: the number ``n`` of equality constraints.
* `g=nothing`: the inequality constraints
* `grad_g=nothing`: the gradient of the inequality constraints
* `grad_h=nothing`: the gradient of the equality constraints
* `gradient_range=nothing`: specify how gradients are represented, where `nothing` is equivalent to [`NestedPowerRepresentation`](@extref)
* `gradient_equality_range=gradient_range`: specify how the gradients of the equality constraints are represented
* `gradient_inequality_range=gradient_range`: specify how the gradients of the inequality constraints are represented
* `h=nothing`: the equality constraints
* `Hess_g=nothing`: the Hessian of the inequality constraints
* `Hess_h=nothing`: the Hessian of the equality constraints
* `inequality_constraints=nothing`: the number ``m`` of inequality constraints.
* `λ=ones(size(h(M,x),1))`: the Lagrange multiplier with respect to the equality constraints ``h``
* `μ=ones(length(g(M,x)))`: the Lagrange multiplier with respect to the inequality constraints ``g``
* `s=μ`: initial value for the slack variables
* `σ=μ's/length(μ)`: ? (TODO find details about barrier parameter)
* `stopping_criterion::StoppingCriterion=`[`StopAfterIteration`](@ref)`(200)`[` | `](@ref StopWhenAny)[`StopWhenChangeLess`](@ref)`(1e-5)`: a stopping criterion
* `retraction_method=`[`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`: the retraction to use, defaults to the default set `M` with respect to the representation for `p` chosen.
* `stepsize=` TODO
* `sub_kwargs=(;)`: keyword arguments to decorate the sub options, for example debug, that automatically respects the main solvers debug options (like sub-sampling) as well
* `sub_stopping_criterion=TODO`: specify a stopping criterion for the subsolver.
* `sub_problem=TODO`: provide a problem for the subsolver, which is assumed to work on the tangent space of ``\mathcal M \times ℝ^n``
* `sub_state=TODO`: a state specifying the subsolver

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details

"""
function interior_point_Newton(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p=rand(M);
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    inequality_constrains::Union{Integer,Nothing}=nothing,
    equality_constrains::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    q = copy(M, p)
    cmo = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        hess_f=Hess_f,
        hess_g=Hess_g,
        hess_h=Hess_h,
        evaluation=evaluation,
        inequality_constrains=inequality_constrains,
        equality_constrains=equality_constrains,
        M=M,
        p=p,
    )
    return interior_point_Newton!(M, cmo, q; evaluation=evaluation, kwargs...)
end
function interior_point_Newton(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    kwargs...,
)
    p_ = [p]
    f_(M, p) = f(M, p[])

    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    Hess_f_ = _to_mutating_gradient(Hess_f, evaluation)

    g_ = isnothing(g) ? nothing : (M, p) -> g(M, p[])
    grad_g_ = isnothing(grad_g) ? nothing : _to_mutating_gradient(grad_g, evaluation)
    Hess_g_ = isnothing(Hess_g) ? nothing : _to_mutating_gradient(Hess_g, evaluation)
    h_ = isnothing(h) ? nothing : (M, p) -> h(M, p[])
    grad_h_ = isnothing(grad_h) ? nothing : _to_mutating_gradient(grad_h, evaluation)
    Hess_h_ = isnothing(Hess_h) ? nothing : _to_mutating_gradient(Hess_h, evaluation)

    rs = interior_point_Newton(
        M,
        f_,
        grad_f_,
        Hess_f_,
        p_;
        evaluation=evaluation,
        g=g_,
        h=h_,
        grad_g=grad_g_,
        grad_h=grad_h_,
        Hess_g=Hess_g_,
        Hess_h=Hess_h_,
        kwargs...,
    )
    return (typeof(p_) == typeof(rs)) ? rs[] : rs
end
function interior_point_Newton(
    M::AbstractManifold, cmo::O, p; kwargs...
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return interior_point_Newton!(M, cmo, q; kwargs...)
end
function interior_point_Newton!(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    inequality_constrains=nothing,
    equality_constrains=nothing,
    kwargs...,
)
    cmo = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        hess_f=Hess_f,
        hess_g=Hess_g,
        hess_h=Hess_h,
        evaluation=evaluation,
        equality_constrains=equality_constrains,
        inequality_constrains=inequality_constrains,
        M=M,
        p=p,
    )
    dcmo = decorate_objective!(M, cmo; kwargs...)
    return interior_point_Newton!(
        M,
        dcmo,
        p;
        evaluation=evaluation,
        equality_constrains=equality_constrains,
        inequality_constrains=inequality_constrains,
        kwargs...,
    )
end
function interior_point_Newton!(
    M::AbstractManifold,
    cmo::O,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    X=get_gradient(M, cmo, p),
    μ::Vector=ones(length(get_inequality_constraint(M, cmo, p, :))),
    Y=zero(μ),
    λ::Vector=zeros(length(get_equality_constraint(M, cmo, p, :))),
    Z=zero(λ),
    s=μ,
    W=zero(s),
    ρ=μ's / length(μ),
    σ=calculate_σ(M, cmo, p, μ, λ, s),
    γ=0.9,
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    sub_kwargs=(;),
    vector_space=Rn,
    centrality_condition=InteriorPointCentralityCondition(cmo, γ),
    step_objective=ManifoldGradientObjective(
        KKTVectorFieldNormSq(cmo), KKTVectorFieldNormSqGradient(cmo); evaluation=evaluation
    ),
    _step_M=M × vector_space(length(μ)) × vector_space(length(λ)) × vector_space(length(s)),
    step_problem=DefaultManoptProblem(_step_M, step_objective),
    _step_p=rand(_step_M),
    step_state=StepsizeState(_step_p, zero_vector(_step_M, _step_p)),
    stepsize::Stepsize=ArmijoLinesearch(
        _step_M;
        retraction_method=default_retraction_method(_step_M),
        initial_stepsize=1.0,
        initial_guess=interior_point_initial_guess,
        additional_decrease_condition=centrality_condition,
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenKKTResidualLess(1e-8),
    _sub_M=M × vector_space(length(λ)),
    _sub_p=rand(_sub_M),
    _sub_X=rand(_sub_M; vector_at=_sub_p),
    sub_objective=decorate_objective!(
        TangentSpace(_sub_M, _sub_p),
        SymmetricLinearSystemObjective(
            CondensedKKTVectorFieldJacobian(cmo, μ, s, σ * ρ),
            CondensedKKTVectorField(cmo, μ, s, σ * ρ),
        ),
        sub_kwargs...,
    ),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(manifold_dimension(M)) |
                                              StopWhenRelativeResidualLess(
        norm(_sub_M, _sub_p, get_b(TangentSpace(_sub_M, _sub_p), sub_objective, _sub_X)),
        1e-8,
    ),
    sub_state::St=decorate_state!(
        ConjugateResidualState(
            TangentSpace(_sub_M, _sub_p),
            sub_objective;
            X=_sub_X,
            stop=sub_stopping_criterion,
            sub_kwargs...,
        );
        sub_kwargs...,
    ),
    sub_problem::Pr=DefaultManoptProblem(TangentSpace(_sub_M, _sub_p), sub_objective),
    kwargs...,
) where {
    O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective},
    St<:Union{AbstractEvaluationType,AbstractManoptSolverState},
    Pr<:Union{F,AbstractManoptProblem} where {F},
}
    !is_feasible(M, cmo, p; error=:error)
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = InteriorPointNewtonState(
        M,
        cmo,
        p,
        sub_problem,
        sub_state;
        X=X,
        μ=μ,
        λ=λ,
        s=s,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        step_problem=step_problem,
        step_state=step_state,
        stepsize=stepsize,
        kwargs...,
    )
    ips = decorate_state!(ips; kwargs...)
    solve!(dmp, ips)
    return get_solver_return(get_objective(dmp), ips)
end
function initialize_solver!(::AbstractManoptProblem, ips::InteriorPointNewtonState)
    return ips
end

function step_solver!(amp::AbstractManoptProblem, ips::InteriorPointNewtonState, i)
    M = get_manifold(amp)
    cmo = get_objective(amp)
    N = base_manifold(get_manifold(ips.sub_problem))
    q = base_point(get_manifold(ips.sub_problem))
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.λ)
    set_iterate!(ips.sub_state, get_manifold(ips.sub_problem), zero_vector(N, q))

    set_manopt_parameter!(ips.sub_problem, :Manifold, :Basepoint, q)
    set_manopt_parameter!(ips.sub_problem, :Objective, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :s, ips.s)
    set_manopt_parameter!(ips.sub_problem, :Objective, :β, ips.ρ * ips.σ)
    # product manifold on which to perform linesearch

    X2 = get_solver_result(solve!(ips.sub_problem, ips.sub_state))
    ips.X, ips.Z = X2[N, 1], X2[N, 2] #for p and λ

    # Compute the remaining part of the solution
    m, n = length(ips.μ), length(ips.λ)
    if m > 0
        g = get_inequality_constraint(amp, ips.p, :)
        grad_g = get_grad_inequality_constraint(amp, ips.p, :)
        β = ips.ρ * ips.σ
        # for s and μ
        ips.W = -[inner(M, ips.p, grad_g[i], ips.X) for i in 1:m] - g - ips.s
        ips.Y = (β .- ips.μ .* (ips.s + ips.W)) ./ ips.s
    end

    N = get_manifold(ips.step_problem)
    # generate current full iterate in step state
    q = get_iterate(ips.step_state)
    copyto!(N[1], q[N, 1], get_iterate(ips))
    copyto!(N[2], q[N, 2], ips.μ)
    copyto!(N[3], q[N, 3], ips.λ)
    copyto!(N[4], q[N, 4], ips.s)
    set_iterate!(ips.step_state, M, q)
    # generate current full gradient
    X = get_gradient(ips.step_state)
    copyto!(N[1], X[N, 1], ips.X)
    (m > 0) && (copyto!(N[2], X[N, 2], ips.Z))
    (n > 0) && (copyto!(N[3], X[N, 3], ips.Y))
    (m > 0) && (copyto!(N[4], X[N, 4], ips.W))
    set_gradient!(ips.step_state, M, q, X)
    # Update centrality factor – Maybe do this as an update function?
    γ = get_manopt_parameter(ips.stepsize, :DecreaseCondition, :γ)
    set_manopt_parameter!(ips.stepsize, :DecreaseCondition, :γ, (γ + 0.5) / 2)
    # determine stepsize
    α = ips.stepsize(ips.step_problem, ips.step_state, i)
    # Update Parameters and slack
    retract!(M, ips.p, ips.p, α * ips.X, ips.retraction_method)
    if m > 0
        ips.μ += α * ips.Y
        ips.s += α * ips.W
        ips.ρ = ips.μ'ips.s / m
        # we can use the memory from above still
        ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s; N=N, q=q)
    end
    (n > 0) && (ips.λ += α * ips.Z)
    return ips
end

get_solver_result(ips::InteriorPointNewtonState) = ips.p
