struct StepsizeState{P,T} <: AbstractManoptSolverState
    q::P
    X::T
end
get_iterate(s::StepsizeState) = s.q
get_gradient(s::StepsizeState) = s.X
set_iterate!(s::StepsizeState, M, q) = copyto!(M, s.q, q)
set_gradient!(s::StepsizeState, M, q, X) = copyto!(M, s.X, q, X)

# struct for state of interior point algorithm
"""
    InteriorPointNewtonState <: AbstractHessianSolverState

# Fields

* `p` the current iterate
* `sub_problem`:        an [`AbstractManoptProblem`](@ref) problem for the subsolver
* `sub_state`:          an [`AbstractManoptSolverState`](@ref) for the subsolver
* `X`: TODO
* `λ`:                  the Lagrange multiplier with respect to the equality constraints
* `μ`:                  the Lagrange multiplier with respect to the inequality constraints
* `s`: the current slack variable
* `ρ`: TODO
* `σ`: TODO
* `stop`: a [`StoppingCriterion`](@ref) indicating when to stop.
* `retraction_method`: the retraction method to use on `M`.
* `stepsize::`[`Stepsize`](@ref): the stepsize to use
* `step_problem::AbstractManoptObjective`: the problem used in the step size
* `centrality_condition`: add a further check to accept steps in the `stepsize`

# Constructor

    InteriorPointNewtonState(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        p,
        sub_problem::Pr,
        sub_state::St;
        kwargs...
    )

Initialize the state, where both the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) and the [`ConstrainedManifoldObjective`](@ref)
are used to fill in reasonable defaults for the keywords.

# Input

TODO

# Keyword arguments

TODO
"""
mutable struct InteriorPointNewtonState{
    P,
    T,
    Pr<:Union{AbstractManoptProblem,F} where {F},
    St<:Union{AbstractManoptSolverState,AbstractEvaluationType},
    R,
    SC<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    TStepsize<:Stepsize,
    TStepPr<:AbstractManoptProblem,
    TStepSt<:AbstractManoptSolverState,
} <: AbstractHessianSolverState
    p::P
    sub_problem::Pr
    sub_state::St
    X::T # not sure if needed?
    μ::T
    λ::T
    s::T
    ρ::R
    σ::R
    stop::SC
    retraction_method::TRTM
    stepsize::TStepsize
    step_problem::TStepPr
    step_state::TStepSt
    function InteriorPointNewtonState(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        p::P,
        sub_problem::Pr,
        sub_state::St;
        X::T=get_gradient(M, cmo, p), # not sure if needed?
        μ::T=ones(length(get_inequality_constraint(M, cmo, p, :))),
        λ::T=zeros(length(get_equality_constraint(M, cmo, p, :))),
        s::T=ones(length(get_inequality_constraint(M, cmo, p, :))),
        ρ::R=μ's / length(get_inequality_constraint(M, cmo, p, :)),
        σ::R=calculate_σ(M, cmo, p, μ, λ, s),
        stopping_criterion::SC=StopAfterIteration(200) | StopWhenChangeLess(1e-8),
        retraction_method::RTM=default_retraction_method(M),
        step_objective=ManifoldGradientObjective(
            KKTVectorFieldNormSq(cmo),
            KKTVectorFieldNormSqGradient(cmo);
            evaluation=InplaceEvaluation(),
        ),
        step_problem::StepPr=DefaultManoptProblem(
            M × Rn(length(μ)) × Rn(length(λ)) × Rn(length(s)), step_objective
        ),
        _q=rand(get_manifold(step_problem)),
        step_state::StepSt=StepsizeState(_q, zero_vector(get_manifold(step_problem), _q)),
        centrality_condition::F=(N, p) -> true, # Todo
        stepsize::S=ArmijoLinesearch(
            get_manifold(step_problem);
            retraction_method=default_retraction_method(get_manifold(step_problem)),
            initial_stepsize=1.0,
            additional_decrease_condition=centrality_condition,
        ),
        kwargs...,
    ) where {
        P,
        T,
        Pr,
        St,
        R,
        F,
        SC<:StoppingCriterion,
        StepPr<:AbstractManoptProblem,
        StepSt<:AbstractManoptSolverState,
        RTM<:AbstractRetractionMethod,
        S<:Stepsize,
    }
        ips = new{P,T,Pr,St,R,SC,RTM,S,StepPr,StepSt}()
        ips.p = p
        ips.sub_problem = sub_problem
        ips.sub_state = sub_state
        ips.X = X
        ips.μ = μ
        ips.λ = λ
        ips.s = s
        ips.ρ = ρ
        ips.σ = σ
        ips.stop = stopping_criterion
        ips.retraction_method = retraction_method
        ips.stepsize = stepsize
        ips.step_problem = step_problem
        ips.step_state = step_state
        return ips
    end
end

# get & set iterate
get_iterate(ips::InteriorPointNewtonState) = ips.p
function set_iterate!(ips::InteriorPointNewtonState, ::AbstractManifold, p)
    ips.p = p
    return ips
end
# get & set gradient (not sure if needed?)
get_gradient(ips::InteriorPointNewtonState) = ips.X
function set_gradient!(ips::InteriorPointNewtonState, ::AbstractManifold, X)
    ips.X = X
    return ips
end
# only message on stepsize for now
function get_message(ips::InteriorPointNewtonState)
    return get_message(ips.stepsize)
end
# pretty print state info
function show(io::IO, ips::InteriorPointNewtonState)
    i = get_count(ips, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ips.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Interior Point Newton Method
    $Iter
    ## Parameters
    * ρ: $(ips.ρ)
    * σ: $(ips.σ)
    * retraction method: $(ips.retraction_method)

    ## Stopping criterion
    $(status_summary(ips.stop))
    ## Stepsize
    $(ips.stepsize)
    This indicates convergence: $Conv
    """
    return print(io, s)
end

#
#
# A special linesearch for IP Newton
function interior_point_initial_guess(
    mp::AbstractManoptProblem, ips::InteriorPointNewtonState, ::Int, l::Real
)
    N = get_manifold(mp) × Rn(length(ips.μ)) × Rn(length(ips.λ)) × Rn(length(ips.s))
    q = rand(N)
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.μ)
    copyto!(N[3], q[N, 3], ips.λ)
    copyto!(N[4], q[N, 4], ips.s)
    Y = GradMeritFunction(N, get_objective(mp), q)
    grad_norm = norm(N, q, Y)
    max_step = max_stepsize(N, q)
    return ifelse(isfinite(max_step), min(l, max_step / grad_norm), l)
end

#
# Constraint functors
#

@doc raw"""
    CondensedKKTVectorField{O<:ConstrainedManifoldObjective,V,R} <: AbstractConstrainedSlackFunctor

Fiven the constrained optimixzation problem

```math
\begin{aligned}
\min_{p ∈\mathcal{M}} &f(p)\\
\text{subject to } &g_i(p)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(p)=0 \quad \text{ for } j=1,…,n,
\end{aligned}
```

Then reformulating the KKT conditions of the Lagrangian
from the optimality conditions of the Lagrangian

```math
\mathcal L(p, μ, λ) = f(p) + \sum_{j=1}^n λ_jh_j(p) + \sum_{i=1}^m μ_ig_i(p)
```

in a perturbed / barrier method in a condensed form
using a slack variable ``s ∈ ℝ^m`` and a barrier parameter ``β``
and the Riemannian gradient of the Lagrangian with respect to the first parameter
``\operatorname{grad}_p L(p, μ, λ)``.

Let ``\mathcal N = \mathcal M × ℝ^n``. We obtain the linear system

```math
\mathcal A(p,λ)[X,Y] = -b(p,λ),\qquad \text{where } (X,Y) ∈ T_{(p,λ)}\mathcal N
```

where ``\mathcal A: T_{(p,λ)}\mathcal N → T_{(p,λ)}\mathcal N`` is a linear operator and
this struct models the right hand side ``b(p,λ) ∈ T_{(p,λ)}\mathcal M`` given by

```math
b(p,λ) = \begin{pmatrix}
\operatorname{grad} f(p)
+ \displaystyle\sum_{j=1}^n λ_j \operatorname{grad} h_j(p)
+ \displaystyle\sum_{i=1}^m μ_i \operatorname{grad} g_i(p)
+ \displaystyle\sum_{i=1}^m \frac{μ_i}{s_i}\bigl(
  μ_i(g_i(p)+s_i) + β - μ_is_i
\bigr)\operatorname{grad} g_i(p)\\
h(p)
\end{pmatrix}
```

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)
* `μ::V` the vector in ``ℝ^m`` of coefficients for the inequality constraints
* `λ::V` the vector in ``ℝ^n`` of coefficients for the equality constraints
* `s::V` the vector in ``ℝ^m`` of sclack variables
* `β::R` the barrier parameter ``β∈ℝ``
"""
mutable struct CondensedKKTVectorField{O<:ConstrainedManifoldObjective,V,R} <:
               AbstractConstrainedSlackFunctor
    cmo::O
    μ::V
    s::V
    β::R
end
function (cKKTvf::CondensedKKTVectorField)(N, q)
    Y = zero_vector(N, q)
    cKKTvf(N, Y, q)
    return Y
end
function (cKKTvf::CondensedKKTVectorField)(N, Y, q)
    M = N[1]
    cmo = cKKTvf.cmo
    p, μ, λ, s = q[N, 1], cKKTvf.μ, q[N, 2], cKKTvf.s
    β = cKKTvf.β
    m, n = length(μ), length(λ)
    # First term of the lagrangian
    get_gradient!(M, Y[N, 1], cmo, p) #grad f
    X = zero_vector(M, p)
    for i in 1:m
        get_grad_inequality_constraint!(M, X, cKKTvf.cmo, p, i)
        gi = get_inequality_constraint(M, cKKTvf.cmo, p, i)
        # Lagrangian term
        Y[N, 1] += μ[i] * X
        #
        Y[N, 1] += (μ[i] / s[i]) * (μ[i] * (gi + s[i]) + β - μ[i] * s[i]) * X
    end
    for j in 1:n
        get_grad_equality_constraint!(M, X, cKKTvf.cmo, p, j)
        hj = get_equality_constraint(M, cKKTvf.cmo, p, j)
        Y[N, 1] += λ[j] * X
        Y[N, 2][j] = hj
    end
    return Y
end

function set_manopt_parameter!(cKKTvf::CondensedKKTVectorField, ::Val{:β}, β)
    cKKTvf.β = β
    return cKKTvf
end
function set_manopt_parameter!(cKKTvf::CondensedKKTVectorField, ::Val{:μ}, μ)
    cKKTvf.μ = μ
    return cKKTvf
end
function set_manopt_parameter!(cKKTvf::CondensedKKTVectorField, ::Val{:s}, s)
    cKKTvf.s = s
    return cKKTvf
end

function show(io::IO, CKKTvf::CondensedKKTVectorField)
    return print(
        io, "CondensedKKTVectorField\n\twith μ=$(CKKTvf.μ), s=$(CKKTvf.s), β=$(CKKTvf.β)"
    )
end

@doc raw"""
    CondensedKKTVectorFieldJacobian{O<:ConstrainedManifoldObjective,V,T}  <: AbstractConstrainedSlackFunctor

Fiven the constrained optimixzation problem

```math
\begin{aligned}
\min_{p ∈\mathcal{M}} &f(p)\\
\text{subject to } &g_i(p)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(p)=0 \quad \text{ for } j=1,…,n,
\end{aligned}
```

we reformulate the KKT conditions of the Lagrangian
from the optimality conditions of the Lagrangian

```math
\mathcal L(p, μ, λ) = f(p) + \sum_{j=1}^n λ_jh_j(p) + \sum_{i=1}^m μ_ig_i(p)
```

in a perturbed / barrier method enhanced as well as condensed form as using ``\operatorname{grad}_o L(p, μ, λ)``
the Riemannian gradient of the Lagrangian with respect to the first parameter.

Let ``\mathcal N = \mathcal M × ℝ^n``. We obtain the linear system

```math
\mathcal A(p,λ)[X,Y] = -b(p,λ),\qquad \text{where } X ∈ T_p\mathcal M, Y ∈ ℝ^n
```
where ``\mathcal A: T_{(p,λ)}\mathcal N → T_{(p,λ)}\mathcal N`` is a linear operator
on ``T_{(p,λ)}\mathcal N = T_p\mathcal M × ℝ^n`` given by

```math
\mathcal A(p,λ)[X,Y] = \begin{pmatrix}
\operatorname{Hess}_p\mathcal L(p, μ, λ)[X]
+ \displaystyle\sum_{i=1}^m \frac{μ_i}{s_i}⟨\operatorname{grad} g_i(p), X⟩\operatorname{grad} g_i(p)
+ \displaystyle\sum_{j=1}^n Y_j \operatorname{grad} h_j(p)
\\
\Bigl( ⟨\operatorname{grad} h_j(p), X⟩ \Bigr)_{j=1}^n
\end{pmatrix}
```
"""
mutable struct CondensedKKTVectorFieldJacobian{O<:ConstrainedManifoldObjective,V,R} <:
               AbstractConstrainedSlackFunctor
    cmo::O
    μ::V
    s::V
    β::R
end
function (cKKTvfJ::CondensedKKTVectorFieldJacobian)(N, q, X)
    Y = zero_vector(N, q)
    cKKTvfJ(N, Y, q, X)
    return Y
end
function (cKKTvfJ::CondensedKKTVectorFieldJacobian)(N, Y, q, X)
    M = N[1]
    p, μ, λ, s = q[N, 1], cKKTvfJ.μ, q[N, 2], cKKTvfJ.s
    m, n = length(μ), length(λ)
    Xp, Xλ = X[N, 1], X[N, 2]
    zero_vector!(N, Y, q)
    Xt = zero_vector(M, p)
    # First Summand of Hess L
    copyto!(M, Y[N, 1], get_hessian(M, cKKTvfJ.cmo, p, Xp))
    # Build the rest iteratively
    for i in 1:m #ineq
        get_hess_inequality_constraint!(M, Xt, cKKTvfJ.cmo, p, Xp, i)
        # Summand of Hess L
        Y[N, 1] += μ[i] * Xt
        get_grad_inequality_constraint!(M, Xt, cKKTvfJ.cmo, p, i)
        # condensed term
        Y[N, 1] += (μ[i] / s[i]) * inner(M, p, Xt, Xp) * Xt
    end
    for j in 1:n #eq
        get_hess_equality_constraint!(M, Xt, cKKTvfJ.cmo, p, Xp, j)
        # Summand of Hess L
        Y[N, 1] += λ[j] * Xt
        get_grad_equality_constraint!(M, Xt, cKKTvfJ.cmo, p, j)
        # condensed term
        Y[N, 1] += Xλ[j] * Xt
        # condensed term in second part
        Y[N, 2][j] = inner(M, p, Xt, Xp)
    end
    return Y
end

function set_manopt_parameter!(cKKTvfJ::CondensedKKTVectorFieldJacobian, ::Val{:β}, β)
    cKKTvfJ.β = β
    return cKKTvfJ
end
function set_manopt_parameter!(cKKTvfJ::CondensedKKTVectorFieldJacobian, ::Val{:μ}, μ)
    cKKTvfJ.μ = μ
    return cKKTvfJ
end
function set_manopt_parameter!(cKKTvfJ::CondensedKKTVectorFieldJacobian, ::Val{:s}, s)
    cKKTvfJ.s = s
    return cKKTvfJ
end
function show(io::IO, CKKTvfJ::CondensedKKTVectorFieldJacobian)
    return print(
        io,
        "CondensedKKTVectorFieldJacobian\n\twith μ=$(CKKTvfJ.μ), s=$(CKKTvfJ.s), β=$(CKKTvfJ.β)",
    )
end

@doc raw"""
    KKTVectorField <: AbstractConstrainedSlackFunctor

Implement the vectorfield ``F`` KKT-conditions, inlcuding a slack variable
for the inequality constraints.

Given the [`LagrangianCost`](@ref)

```math
\mathcal L(p, μ, λ) = f(p) + \sum_{j=1}^n λ_jh_j(p) + \sum_{i=1}^m μ_ig_i(p),
```

the [`LagrangianGradient`](@ref)

```math
\operatorname{grad}\mathcal L(p, μ, λ) = \operatorname{grad}f(p) + \sum_{j=1}^n λ_j \operatorname{grad} h_j(p) + \sum_{i=1}^m μ_i \operatorname{grad} g_i(p),
```

and introducing the slack variables ``s=-g(p) ∈ ℝ^m``
the vector field is given by

```math
F(p, μ, λ, s) = \begin{pmatrix}
\operatorname{grad}\mathcal L(p, μ, λ)\\
g(p) + s\\
h(p)\\
μ ⊙ s
\end{pmatrix}, \text{ where } p \in \mathcal M, μ, s \in ℝ^m\text{ and } λ \in ℝ^n,
```
where ``⊙`` denotes the Hadamard (or elementwise) product

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

While the point `p` is arbitrary and usually not needed, it serves as internal memory
in the computations. Furthermore Both fields together also calrify the product manifold structure to use.

# Constructor

    KKTVectorField(cmo::ConstrainedManifoldObjective)

# Example

Define `F = KKTVectorField(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``\mathcal M×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `F(N, q)` or as the in-place variant `F(N, Y, q)`,
where `q` is a point on `N` and `Y` is a tangent vector at `q` for the result.
"""
struct KKTVectorField{O<:ConstrainedManifoldObjective} <: AbstractConstrainedSlackFunctor
    cmo::O
end
function (KKTvf::KKTVectorField)(N, q)
    Y = zero_vector(N, q)
    return KKTvf(N, Y, q)
end
function (KKTvf::KKTVectorField)(N, Y, q)
    M = N[1]
    p = q[N, 1]
    # To improve readability
    μ, λ, s = q[N, 2], q[N, 3], q[N, 4]
    LagrangianGradient(KKTvf.cmo, μ, λ)(M, Y[N, 1], p)
    m, n = length(μ), length(λ)
    (m > 0) && (Y[N, 2] = get_inequality_constraint(M, KKTvf.cmo, p, :) + s)
    (n > 0) && (Y[N, 3] = get_equality_constraint(M, KKTvf.cmo, p, :))
    (m > 0) && (Y[N, 4] = μ .* s)
    return Y
end
function show(io::IO, KKTvf::KKTVectorField)
    return print(io, "KKTVectorField\nwith the objective\n\t$(KKTvf.cmo)")
end

@doc raw"""
    KKTVectorFieldJacobian <: AbstractConstrainedSlackFunctor

Implement the Jacobian of the vector field ``F`` of the KKT-conditions, inlcuding a slack variable
for the inequality constraints, see [`KKTVectorField`](@ref) and [`KKTVectorFieldAdjointJacobian`](@ref)..

```math
\operatorname{J} F(p, μ, λ, s)[X, Y, Z, W] = \begin{pmatrix}
    \operatorname{Hess} \mathcal L(p, μ, λ)[X] + \displaystyle\sum_{i=1}^m Y_i \operatorname{grad} g_i(p) + \displaystyle\sum_{j=1}^n Z_j \operatorname{grad} h_j(p)\\
    \Bigl( ⟨\operatorname{grad} g_i(p), X⟩ + W_i\Bigr)_{i=1}^m\\
    \Bigl( ⟨\operatorname{grad} h_j(p), X⟩ \Bigr)_{j=1}^n\\
    μ ⊙ W + s ⊙ Y
\end{pmatrix},
```
where ``⊙`` denotes the Hadamard (or elementwise) product

See also the [`LagrangianHessian`](@ref) ``\operatorname{Hess} \mathcal L(p, μ, λ)[X]``.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldJacobian(cmo::ConstrainedManifoldObjective)

Generate the Jacobian of the KKT vector field related to some [`ConstrainedManifoldObjective`](@ref) `cmo`.

# Example

Define `JF = KKTVectorFieldJacobian(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``\mathcal M×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `JF(N, q, Y)` or as the in-place variant `JF(N, Z, q, Y)`,
where `q` is a point on `N` and `Y` and `Z` are a tangent vector at `q`.
"""
mutable struct KKTVectorFieldJacobian{O<:ConstrainedManifoldObjective} <:
               AbstractConstrainedSlackFunctor
    cmo::O
end
function (KKTvfJ::KKTVectorFieldJacobian)(N, q, Y)
    Z = zero_vector(N, q)
    return KKTvfJ(N, Z, q, Y)
end
function (KKTvfJ::KKTVectorFieldJacobian)(N, Z, q, Y)
    M = N[1]
    p = q[N, 1]
    μ, λ, s = q[N, 2], q[N, 3], q[N, 4]
    X = Y[N, 1]
    # First component
    LagrangianHessian(KKTvfJ.cmo, μ, λ)(M, Z[N, 1], p, Y[N, 1])
    Xt = copy(M, p, X)
    m, n = length(μ), length(λ)
    for i in 1:m
        get_grad_inequality_constraint!(M, Xt, KKTvfJ.cmo, p, i)
        copyto!(M, Z[N, 1], p, Z[N, 1] + Y[N, 2][i] * Xt)
        # set second components as well
        Z[N, 2][i] = inner(M, p, Xt, X) + Y[N, 4][i]
    end
    for j in 1:n
        get_grad_equality_constraint!(M, Xt, KKTvfJ.cmo, p, j)
        copyto!(M, Z[N, 1], p, Z[N, 1] + Y[N, 3][j] * Xt)
        # set third components as well
        Z[N, 3][j] = inner(M, p, Xt, X)
    end
    # Fourth component
    Z[N, 4] = μ .* Y[N, 4] + s .* Y[N, 2]
    return Z
end
function show(io::IO, KKTvfJ::KKTVectorFieldJacobian)
    return print(io, "KKTVectorFieldJacobian\nwith the objective\n\t$(KKTvfJ.cmo)")
end

@doc raw"""
    KKTVectorFieldAdjointJacobian <: AbstractConstrainedSlackFunctor

Implement the Adjoint of the Jacobian of the vector field ``F`` of the KKT-conditions, inlcuding a slack variable
for the inequality constraints, see [`KKTVectorField`](@ref) and [`KKTVectorFieldJacobian`](@ref).

```math
\operatorname{J}^* F(p, μ, λ, s)[X, Y, Z, W] = \begin{pmatrix}
    \operatorname{Hess} \mathcal L(p, μ, λ)[X] + \displaystyle\sum_{i=1}^m Y_i \operatorname{grad} g_i(p) + \displaystyle\sum_{j=1}^n Z_j \operatorname{grad} h_j(p)\\
    \Bigl( ⟨\operatorname{grad} g_i(p), X⟩ + s_iW_i\Bigr)_{i=1}^m\\
    \Bigl( ⟨\operatorname{grad} h_j(p), X⟩ \Bigr)_{j=1}^n\\
    μ ⊙ W + Y
\end{pmatrix},
```
where ``⊙`` denotes the Hadamard (or elementwise) product

See also the [`LagrangianHessian`](@ref) ``\operatorname{Hess} \mathcal L(p, μ, λ)[X]``.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldAdjointJacobian(cmo::ConstrainedManifoldObjective)

Generate the Adjoint Jacobian of the KKT vector field related to some [`ConstrainedManifoldObjective`](@ref) `cmo`.

# Example

Define `AdJF = KKTVectorFieldAdjointJacobian(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``\mathcal M×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `AdJF(N, q, Y)` or as the in-place variant `AdJF(N, Z, q, Y)`,
where `q` is a point on `N` and `Y` and `Z` are a tangent vector at `q`.
"""
mutable struct KKTVectorFieldAdjointJacobian{O<:ConstrainedManifoldObjective} <:
               AbstractConstrainedSlackFunctor
    cmo::O
end
function (KKTvfJa::KKTVectorFieldAdjointJacobian)(N, q, Y)
    Z = zero_vector(N, q)
    return KKTvfJa(N, Z, q, Y)
end
function (KKTvfAdJ::KKTVectorFieldAdjointJacobian)(N, Z, q, Y)
    M = N[1]
    p = q[N, 1]
    μ, λ, s = q[N, 2], q[N, 3], q[N, 4]
    X = Y[N, 1]
    # First component
    LagrangianHessian(KKTvfAdJ.cmo, μ, λ)(M, Z[N, 1], p, X)
    Xt = copy(M, p, X)
    m, n = length(μ), length(λ)
    for i in 1:m
        get_grad_inequality_constraint!(M, Xt, KKTvfAdJ.cmo, p, i)
        copyto!(M, Z[N, 1], p, Z[N, 1] + Y[N, 2][i] * Xt)
        # set second components as well
        Z[N, 2][i] = inner(M, p, Xt, X) + s[i] * Y[N, 4][i]
    end
    for j in 1:n
        get_grad_equality_constraint!(M, Xt, KKTvfAdJ.cmo, p, j)
        copyto!(M, Z[N, 1], p, Z[N, 1] + Y[N, 3][j] * Xt)
        # set third components as well
        Z[N, 3][j] = inner(M, p, Xt, X)
    end
    # Fourth component
    Z[N, 4] = μ .* Y[N, 4] + Y[N, 2]
    return Z
end
function show(io::IO, KKTvfAdJ::KKTVectorFieldAdjointJacobian)
    return print(io, "KKTVectorFieldAdjointJacobian\nwith the objective\n\t$(KKTvfAdJ.cmo)")
end

@doc raw"""
    KKTVectorFieldNormSq <: AbstractConstrainedSlackFunctor

Implement the square of the norm of the vectorfield ``F`` of the KKT-conditions, inlcuding a slack variable
for the inequality constraints, see [`KKTVectorField`](@ref), where this functor applies the norm to.
In [LaiYoshise:2024](@cite) this is called the merit function.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldNormSq(cmo::ConstrainedManifoldObjective)

# Example

Define `f = KKTVectorFieldNormSq(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``\mathcal M×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `f(N, q)`, where `q` is a point on `N`.
"""
mutable struct KKTVectorFieldNormSq{O<:ConstrainedManifoldObjective} <:
               AbstractConstrainedSlackFunctor
    cmo::O
end
function (KKTvc::KKTVectorFieldNormSq)(N, q)
    Y = zero_vector(N, q)
    KKTVectorField(KKTvc.cmo)(N, Y, q)
    return inner(N, q, Y, Y)
end
function show(io::IO, KKTvfNSq::KKTVectorFieldNormSq)
    return print(io, "KKTVectorFieldNormSq\nwith the objective\n\t$(KKTvfNSq.cmo)")
end

@doc raw"""
    KKTVectorFieldNormSqGradient <: AbstractConstrainedSlackFunctor

Compute the gradient of the [`KKTVectorFieldNormSq`](@ref) ``φ(p,μ,λ,s) = \lVert F(p,μ,λ,s)\rvert^2``,
that is of the norm squared of the [`KKTVectorField`](@ref) ``F``.

This is given in [LaiYoshise:2024](@cite) as the gradient of their merit function,
which we can write with the adjoint ``J^*`` of the Jacobian

```math
\operatorname{grad} φ = 2\operatorname{J}^* F(p, μ, λ, s)[F(p, μ, λ, s)],
```

and hence is computed with [`KKTVectorFieldAdjointJacobian`](@ref) and [`KKTVectorField`](@ref).

For completeness, the gradient reads, using the [`LagrangianGradient`](@ref) ``L = \operatorname{grad} \mathcal L(p,μ,λ) ∈ T_p\mathcal M``,
for a shorthand of the first component of ``F``, as

```math
\operatorname{grad} φ
=
2 \begin{pmatrix}
\operatorname{grad} \mathcal L(p,μ,λ)[L] + (g_i(p) + s_i)\operatorname{grad} g_i(p) + h_j(p)\operatorname{grad} h_j(p)\\
  \Bigl( ⟨\operatorname{grad} g_i(p), L⟩ + s_i\Bigr)_{i=1}^m + μ ⊙ s ⊙ s\\
  \Bigl( ⟨\operatorname{grad} h_j(p), L⟩ \Bigr)_{j=1}^n\\
  g + s + μ ⊙ μ ⊙ s
\end{pmatrix},
```
where ``⊙`` denotes the Hadamard (or elementwise) product.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldNormSqGradient(cmo::ConstrainedManifoldObjective)

# Example

Define `grad_f = KKTVectorFieldNormSqGradient(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``\mathcal M×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `grad_f(N, q)` or as the in-place variant `grad_f(N, Y, q)`,
where `q` is a point on `N` and `Y` is a tangent vector at `q` returning the resulting gradient at.
"""
mutable struct KKTVectorFieldNormSqGradient{O<:ConstrainedManifoldObjective} <:
               AbstractConstrainedSlackFunctor
    cmo::O
end
function (KKTcfNG::KKTVectorFieldNormSqGradient)(N, q)
    Y = zero_vector(N, q)
    KKTcfNG(N, Y, q)
    return Y
end
function (KKTcfNG::KKTVectorFieldNormSqGradient)(N, Y, q)
    Z = copy(N, q, Y)
    KKTVectorField(KKTcfNG.cmo)(N, Z, q)
    KKTVectorFieldAdjointJacobian(KKTcfNG.cmo)(N, Y, q, Z)
    return Y
end
function show(io::IO, KKTvfNSqGrad::KKTVectorFieldNormSqGradient)
    return print(
        io, "KKTVectorFieldNormSqGradient\nwith the objective\n\t$(KKTvfNSqGrad.cmo)"
    )
end

# -----------------------------------------------------------------------------
# old code, old names - TODO check / rename / document
function calculate_σ(
    N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, p, μ, λ, s
)
    return calculate_σ(N, get_objective(cmo, true), p, μ, λ, s)
end
function calculate_σ(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s)
    N = M × Rn(length(μ)) × Rn(length(λ)) × Rn(length(s))
    q = allocate_result(N, rand)
    copyto!(N[1], q[N, 1], p)
    copyto!(N[2], q[N, 2], μ)
    copyto!(N[3], q[N, 3], λ)
    copyto!(N[4], q[N, 4], s)
    return min(0.5, (KKTVectorFieldNormSq(cmo)(N, q))^(1 / 4))
end
mutable struct ConstraintLineSearchCheckFunction{CO}
    cmo::CO
    τ1::Float64
    τ2::Float64
    γ::Float64
end
function (clcf::ConstraintLineSearchCheckFunction)(N, q)
    μ = q[N, 2]
    s = q[N, 4]
    KKTvf = KKTVectorFieldNormSq(clcf.cmo)
    (minimum(μ .* s) - clcf.γ * clcf.τ1 / length(μ) < 0) && return false
    (sum(μ .* s) - clcf.γ * clcf.τ2 * sqrt(KKTvf(N, q)) < 0) && return false
    return true
end
