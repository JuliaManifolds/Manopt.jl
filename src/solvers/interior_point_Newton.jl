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

```math
\operatorname{Jacobian} F(p, μ, λ, s)[X, Y, Z, W] = -F(p, μ, λ, s),
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

(TODO: Link to sub cost/grad/Hessian, and Linesearch once documented)

# Input

* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``f : \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad} f : \mathcal M → T \mathcal M`` of ``f``
* `Hess_f`: the Hessian ``\operatorname{Hess}f(p): T_p\mathcal M → T_p\mathcal M``, ``X ↦ \operatorname{Hess}F(p)[X] = ∇_X\operatorname{grad}f(p)``
* `p=[`rand`](@extref ManifoldsBase.rand)`(M)`: an initial value ``x  ∈  \mathcal M``

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
* `μ=ones(size(h(M,x),1))`: the Lagrange multiplier with respect to the inequality constraints ``g``
* `s=μ`: initial value for the slack variables
* `σ= μ's/length(μ)`: ? (TODO find details about barrier parameter)
* `stopping_criterion::StoppingCriterion=`[`StopAfterIteration`](@ref)`(200)`[` | `](@ref StopWhenAny)[`StopWhenChangeLess`](@ref)`(1e-5)`: a stopping criterion
* `retraction_method=[`default_retraction_method`](@extref)`(M, typeof(p))`: the retraction to use, defaults to the default set `M` with respect to the representation for `p` chosen.
* `stepsize=`[`InteriorPointLinesearch`](@ref)`()`:
* `sub_kwargs=(;)`: keyword arguments to decorate the sub options, for example debug, that automatically respects the main solvers debug options (like sub-sampling) as well
* `sub_stopping_criterion=TODO`: specify a stopping criterion for the subsolver.
* `sub_problem=TODO`: provide a problem for the subsolver, which is assumed to work on the tangent space of `\mathcal M \times ℝ^n`
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
    num_eq = if isnothing(equality_constrains)
        _number_of_constraints(h, grad_h; M=M, p=p)
    else
        inequality_constrains
    end
    num_ineq = if isnothing(inequality_constrains)
        _number_of_constraints(g, grad_g; M=M, p=p)
    else
        inequality_constrains
    end
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
        inequality_constrains=num_ineq,
        equality_constrains=num_eq,
        M=M,
        p=p,
    )
    return interior_point_Newton!(
        M,
        cmo,
        q;
        evaluation=evaluation,
        inequality_constrains=num_ineq,
        equality_constrains=num_eq,
        kwargs...,
    )
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
    if isnothing(inequality_constrains)
        inequality_constrains = _number_of_constraints(g, grad_g; M=M, p=p)
    end
    if isnothing(equality_constrains)
        equality_constrains = _number_of_constraints(h, grad_h; M=M, p=p)
    end
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
    λ::Vector=zeros(length(get_equality_constraint(M, cmo, p, :))),
    s=μ,
    ρ=μ's / length(μ),
    σ=calculate_σ(M, cmo, p, μ, λ, s),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenChangeLess(1e-5),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    sub_kwargs=(;),
    _sub_M=M × Rn(length(λ)),
    _sub_p=rand(_sub_M),
    centrality_condition=ConstraintLineSearchCheckFunction(
        cmo, length(μ) * minimum(μ .* s) / sum(μ .* s), sum(μ .* s), 0.1
    ), # TODO
    step_objective=ManifoldGradientObjective(
        KKTVectorFieldNormSq(cmo, μ, λ, s),
        KKTVectorFieldNormSqGradient(cmo, μ, λ, s);
        evaluation=evaluation,
    ),
    step_problem=DefaultManoptProblem(
        M × Rn(length(μ)) × Rn(length(λ)) × Rn(length(s)), step_objective
    ),
    stepsize::Stepsize=ArmijoLinesearch(
        get_manifold(step_problem);
        retraction_method=default_retraction_method(get_manifold(step_problem)),
        initial_stepsize=1.0,
        additional_decrease_condition=centrality_condition,
    ),
    sub_objective=decorate_objective!(
        TangentSpace(_sub_M, _sub_p),
        SymmetricLinearSystemObjective(
            CondensedKKTVectorFieldJacobian(cmo, μ, λ, s, σ * ρ),
            CondensedKKTVectorField(cmo, μ, λ, s, σ * ρ),
        ),
        sub_kwargs...,
    ),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(manifold_dimension(M)) |
                                              StopWhenGradientNormLess(1e-5),
    sub_state::St=decorate_state!(
        ConjugateResidualState(
            TangentSpace(_sub_M, rand(_sub_M)),
            sub_objective;
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
        stop=stopping_criterion,
        retraction_method=retraction_method,
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
    m, n = length(ips.μ), length(ips.λ)
    g = get_inequality_constraint(amp, ips.p, :)
    grad_g = get_grad_inequality_constraint(amp, ips.p, :)

    N = base_manifold(get_manifold(ips.sub_problem))
    q = base_point(get_manifold(ips.sub_problem))
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.λ)
    set_iterate!(ips.sub_state, get_manifold(ips.sub_problem), zero_vector(N, q))

    set_manopt_parameter!(ips.sub_problem, :Manifold, :Basepoint, q)

    set_manopt_parameter!(ips.sub_problem, :Objective, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :s, ips.s)
    set_manopt_parameter!(ips.sub_problem, :Objective, :barrier_param, ips.ρ * ips.σ)
    # product manifold on which to perform linesearch

    Y = get_solver_result(solve!(ips.sub_problem, ips.sub_state))
    Xp, Xλ = Y[N, 1], Y[N, 2]

    # Compute the remaining part of the solution
    if m > 0
        b = ips.ρ * ips.σ
        Xs = -[inner(M, ips.p, grad_g[i], Xp) for i in 1:m] - g - ips.s
        Xμ = (b .- ips.μ .* (ips.s + Xs)) ./ ips.s
    end

    # How to find K in a good way? -> move that to setting something in the stepsize?
    K = M × Rn(m) × Rn(n) × Rn(m)
    X = allocate_result(K, rand)
    copyto!(K[1], X[N, 1], Xp)
    (m > 0) && (copyto!(K[2], X[K, 2], Xμ))
    (n > 0) && (copyto!(K[3], X[K, 3], Xλ))
    (m > 0) && (copyto!(K[4], X[K, 4], Xs))
    # determine stepsize
    α = ips.stepsize(amp, ips, i, X)
    # Update Parameters and slack
    retract!(M, ips.p, ips.p, α * Xp, ips.retraction_method)
    if m > 0
        ips.μ += α * Xμ
        ips.s += α * Xs
        ips.ρ = ips.μ'ips.s / m
        ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s)
    end
    (n > 0) && (ips.λ += α * Xλ)
    get_gradient!(M, ips.X, cmo, ips.p)
    return ips
end

get_solver_result(ips::InteriorPointNewtonState) = ips.p
