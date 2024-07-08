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
* `stepsize` the stepsize to use
"""
mutable struct InteriorPointNewtonState{
    P,
    Pr<:AbstractManoptProblem,
    St<:AbstractManoptSolverState,
    T,
    R,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    TStepsize<:Stepsize,
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
    stop::TStop
    retraction_method::TRTM
    stepsize::TStepsize
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
        stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-8),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        stepsize::Stepsize=InteriorPointLinesearch(
            M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s);
            retraction_method=default_retraction_method(
                M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s)
            ),
            initial_stepsize=1.0,
        ),
        kwargs...,
    ) where {P,Pr,St,T,R}
        ips = new{
            P,
            typeof(sub_problem),
            typeof(sub_state),
            T,
            R,
            typeof(stop),
            typeof(retraction_method),
            typeof(stepsize),
        }()
        ips.p = p
        ips.sub_problem = sub_problem
        ips.sub_state = sub_state
        ips.X = X
        ips.μ = μ
        ips.λ = λ
        ips.s = s
        ips.ρ = ρ
        ips.σ = σ
        ips.stop = stop
        ips.retraction_method = retraction_method
        ips.stepsize = stepsize
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
    N = get_manifold(mp) × ℝ^length(ips.μ) × ℝ^length(ips.λ) × ℝ^length(ips.s)
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

@doc """
    InteriorPointLinesearch <: Linesearch

An (maybe interims) line search for IPNewton.

TODO: Check whether we can hopefully optimize / adapt this function to be just Armijo again,
by checking/externalizing the current `MeritFunction` and `GradMeritFunction` dependency,
probably by storing these in the state or even the objective?
"""
mutable struct InteriorPointLinesearch{TRM<:AbstractRetractionMethod,Q,F,DF,IF} <:
               Linesearch
    candidate_point::Q
    contraction_factor::Float64
    initial_guess::F
    initial_stepsize::Float64
    last_stepsize::Float64
    message::String
    retraction_method::TRM
    sufficient_decrease::Float64
    stop_when_stepsize_less::Float64
    stop_when_stepsize_exceeds::Float64
    stop_increasing_at_step::Int
    stop_decreasing_at_step::Int
    additional_decrease_condition::DF
    additional_increase_condition::IF
    function InteriorPointLinesearch(
        N::AbstractManifold=DefaultManifold();
        additional_decrease_condition::DF=(N, q) -> true,
        additional_increase_condition::IF=(N, q) -> true,
        candidate_point::Q=allocate_result(N, rand),
        contraction_factor::Real=0.95,
        initial_stepsize::Real=1.0,
        initial_guess=interior_point_initial_guess,
        retraction_method::TRM=default_retraction_method(N),
        stop_when_stepsize_less::Real=0.0,
        stop_when_stepsize_exceeds::Real=max_stepsize(N),
        stop_increasing_at_step::Int=0,
        stop_decreasing_at_step::Int=1000,
        sufficient_decrease=0.1,
    ) where {TRM,Q,DF,IF}
        return new{TRM,Q,typeof(initial_guess),DF,IF}(
            candidate_point,
            contraction_factor,
            initial_guess,
            initial_stepsize,
            initial_stepsize,
            "", # initialize an empty message
            retraction_method,
            sufficient_decrease,
            stop_when_stepsize_less,
            stop_when_stepsize_exceeds,
            stop_increasing_at_step,
            stop_decreasing_at_step,
            additional_decrease_condition,
            additional_increase_condition,
        )
    end
end
function (ipls::InteriorPointLinesearch)(
    mp::AbstractManoptProblem, ips::InteriorPointNewtonState, i::Int, η; kwargs...
)
    N = get_manifold(mp) × ℝ^length(ips.μ) × ℝ^length(ips.λ) × ℝ^length(ips.s)
    q = allocate_result(N, rand)
    copyto!(N[1], q[N, 1], get_iterate(ips))
    copyto!(N[2], q[N, 2], ips.μ)
    copyto!(N[3], q[N, 3], ips.λ)
    copyto!(N[4], q[N, 4], ips.s)
    X = GradMeritFunction(N, get_objective(mp), q)
    (ipls.last_stepsize, ipls.message) = linesearch_backtrack!(
        N,
        ipls.candidate_point,
        (N, q) -> MeritFunction(N, get_objective(mp), q),
        q,
        X,
        ipls.initial_guess(mp, ips, i, ipls.last_stepsize),
        ipls.sufficient_decrease,
        ipls.contraction_factor,
        η;
        retraction_method=ipls.retraction_method,
        stop_when_stepsize_less=ipls.stop_when_stepsize_less / norm(N, q, η),
        stop_when_stepsize_exceeds=ipls.stop_when_stepsize_exceeds / norm(N, q, η),
        stop_increasing_at_step=ipls.stop_increasing_at_step,
        stop_decreasing_at_step=ipls.stop_decreasing_at_step,
        additional_decrease_condition=ipls.additional_decrease_condition,
        additional_increase_condition=ipls.additional_increase_condition,
    )
    return ipls.last_stepsize
end
get_initial_stepsize(ipls::InteriorPointLinesearch) = ipls.initial_stepsize
function show(io::IO, ipls::InteriorPointLinesearch)
    return print(
        io,
        """
        InteriorPointLinesearch() with keyword parameters
          * initial_stepsize    = $(ipls.initial_stepsize)
          * retraction_method   = $(ipls.retraction_method)
          * contraction_factor  = $(ipls.contraction_factor)
          * sufficient_decrease = $(ipls.sufficient_decrease)""",
    )
end
function status_summary(ipls::InteriorPointLinesearch)
    return "$(ipls)\nand a computed last stepsize of $(ipls.last_stepsize)"
end
get_message(ipls::InteriorPointLinesearch) = ipls.message

function get_last_stepsize(step::InteriorPointLinesearch, ::Any...)
    return step.last_stepsize
end

@doc raw"""
    CondensedKKTVectorField{}

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
\mathcal L(p, μ, λ,) = f(p) + \sum_{j=1}^n λ_jh_j(p) + \sum_{i=1}^m μ_ig_i(p)
```

in a perturbed / barrier method in a condensed form
using a slack variable ``s ∈ ℝ^m`` and a barrier parameter ``β``
and the Riemannian gradient of the Lagrangian with respect to the first parameter
``\operatorname{grad}_p L(p, μ, λ)``.

Let ``\mathcal N = \mathcal M × ℝ^n``. We obtain the linear system

```math
\mathcal A[X,Y] = -b,\qquad \text{where } X ∈ T_p\mathcal M, Y ∈ ℝ^n
```
where ``\mathcal A: T_q\mathcal N → T_q\mathcal N`` is a linear operator and
this struct models the right hand side

```math
b = \begin{pmatrix}
\operatorname{grad} f(p)
+ \displaystyle\sum_{j=1}^n λ_j \operatorname{grad} h_j(p)
+ \displaystyle\sum_{i=1}^m μ_i \operatorname{grad} g_i(p)
+ \displaystyle\sum_{i=1}^m \frac{μ_i}{s_i}\bigl(
  μ_i(g_i(p)+s_i) + β - μ_is_i
\bigr)\operatorname{grad} g_i(p)\\
h(p)
\end{pmatrix} ∈ T_p\mathcal M × ℝ^n
```

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)
* `μ::V` the vector in ``ℝ^m`` of coefficients for the inequality constraints
* `λ::V` the vector in ``ℝ^n`` of coefficients for the equality constraints
* `s::Vthe vector in ``ℝ^m`` of sclack variables
* `β::R` the barrier parameter ``β∈ℝ``
"""
mutable struct CondensedKKTVectorField{O<:ConstrainedManifoldObjective,V,R}
    cmo::O
    μ::V
    λ::V
    s::V
    β::R
end
function (cKKTvf::CondensedKKTVectorField)(N::AbstractManifold, q)
    Y = zero_vector(N, q)
    cKKTvf(N, Y, q)
    return Y
end
function (cKKTvf::CondensedKKTVectorField)(N::AbstractManifold, Y, q)
    M = N[1]
    cmo = cKKTvf.cmo
    p, μ, λ, s = q[N, 1], cKKTvf.μ, q[N, 2], cKKTvf.s
    β = cKKTvf.β
    m, n = length(μ), length(λ)
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)
    grad_g = get_grad_inequality_constraint(M, cmo, p, :)
    # grad_h = get_grad_equality_constraint(M, cmo, p, :)
    copyto!(M, Y[N, 1], get_gradient(M, cmo, p))

    ν = μ + (μ .* g .+ β) ./ s
    (m > 0) && (
        Y[N, 1] += sum(
            ((μ[i] * g[i] + β - μ[i] * s[i]) / s[i]) * grad_g[i] * ν[i] for i in 1:m
        )
    )
    # I do not understand where this comes from
    # (n > 0) && (X[N, 1] -= sum(grad_h[j] * λ[j] for j in 1:n))
    (n > 0) && (copyto!(N[2], Y[N, 2], h))
    return Y
end

function set_manopt_parameter!(cKKTvf::CondensedKKTVectorField, ::Val{:μ}, μ)
    cKKTvf.μ = μ
    return cKKTvf
end

function set_manopt_parameter!(cKKTvf::CondensedKKTVectorField, ::Val{:λ}, λ)
    cKKTvf.λ = λ
    return cKKTvf
end

function set_manopt_parameter!(cKKTvf::CondensedKKTVectorField, ::Val{:s}, s)
    cKKTvf.s = s
    return cKKTvf
end

function set_manopt_parameter!(cKKTvf::CondensedKKTVectorField, ::Val{:β}, β)
    cKKTvf.β = β
    return cKKTvf
end

@doc raw"""
    CondensedKKTVectorFieldJacobian{O<:ConstrainedManifoldObjective,V,T}

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
\mathcal L(p, μ, λ, s) = f(p) + \sum_{j=1}^n λ_jh_j(p) + \sum_{i=1}^m μ_ig_i(p)
```

in a perturbed / barrier method enhanced as well as condensed form as using ``\operatorname{grad}_o L(p, μ, λ)``
the Riemannian gradient of the Lagrangian with respect to the first parameter.

Let ``\mathcal N = \mathcal M × ℝ^n``. We obtain the linear system

```math
\mathcal A[X,Y] = -b,\qquad \text{where } X ∈ T_p\mathcal M, Y ∈ ℝ^n
```
where ``\mathcal A: T_q\mathcal N → T_q\mathcal N`` is a linear operator

```math
\mathcal A[X,Y] = \begin{pmatrix}
\operatorname{Hess}_p\mathcal L(p, μ, λ)
+ \displaystyle\sum_{i=1}^m \frac{μ_i}{s_i}⟨\operatorname{grad} g_i(p), X⟩\operatorname{grad} g_i(p)
+ \displaystyle\sum_{j=1}^n λ_j \operatorname{grad} h_j(p)
\\
\Bigl( ⟨\operatorname{grad} h_j(p), X⟩ \Bigr)_{j=1}^n
\end{pmatrix} ∈ T_p\mathcal M × ℝ^n
```
"""
mutable struct CondensedKKTVectorFieldJacobian{O<:ConstrainedManifoldObjective,V,R}
    cmo::O
    μ::V
    λ::V
    s::V
    b::R
end
function (cKKTvfJ::CondensedKKTVectorFieldJacobian)(N, p, X)
    Y = zero_vector(N, p)
    cKKTvfJ(N, Y, p, X)
    return Y
end
function (cKKTvfJ::CondensedKKTVectorFieldJacobian)(N, Y, q, X)
    M = N[1]
    cmo = cKKTvfJ.cmo
    p, μ, λ, s = q[N, 1], cKKTvfJ.μ, q[N, 2], cKKTvfJ.s
    m, n = length(μ), length(λ)
    Xp, Xλ = X[N, 1], X[N, 2]
    # First Summand of Hess L
    copyto!(M, X[N, 1], get_hessian(M, cmo, p, Xp))
    if m > 0
        grad_g = get_grad_inequality_constraint(M, cmo, p, :)
        H_g = get_hess_inequality_constraint(M, cmo, p, Xp, :)
        # Summand of Hess L
        Y[N, 1] += sum([μ[i] * H_g[i] for i in 1:m])
        Y[N, 1] += sum([μ[i] / s[i] * inner(M, p, grad_g[i], Xp) * grad_g[i] for i in 1:m])
    end
    if n > 0
        grad_h = get_grad_equality_constraint(M, cmo, p, :)
        H_h = get_hess_equality_constraint(M, cmo, p, Xp, :)
        # Summand of Hess L
        Y[N, 1] += sum([λ[j] * H_h[j] for j in 1:n])
        Y[N, 1] += sum([Xλ[j] * grad_h[j] for j in 1:n])
        copyto!(N[2], Y[N, 2], [inner(M, p, grad_h[j], Xp) for j in 1:n])
    end
    return Y
end

function set_manopt_parameter!(cKKTvfJ::CondensedKKTVectorFieldJacobian, ::Val{:μ}, μ)
    cKKTvfJ.μ = μ
    return cKKTvfJ
end

function set_manopt_parameter!(cKKTvfJ::CondensedKKTVectorFieldJacobian, ::Val{:λ}, λ)
    cKKTvfJ.λ = λ
    return cKKTvfJ
end

function set_manopt_parameter!(cKKTvfJ::CondensedKKTVectorFieldJacobian, ::Val{:s}, s)
    cKKTvfJ.s = s
    return cKKTvfJ
end

function set_manopt_parameter!(cKKTvfJ::CondensedKKTVectorFieldJacobian, ::Val{:β}, β)
    cKKTvfJ.β = β
    return cKKTvfJ
end
#
#
# Subproblem gradient and hessian
mutable struct NegativeReducedLagrangianGrad{T,R}
    cmo::ConstrainedManifoldObjective
    μ::T
    λ::T
    s::T
    barrier_param::R
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:μ}, μ)
    nrlg.μ = μ
    return nrlg
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:λ}, λ)
    nrlg.λ = λ
    return nrlg
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:s}, s)
    nrlg.s = s
    return nrlg
end

function set_manopt_parameter!(
    nrlg::NegativeReducedLagrangianGrad, ::Val{:barrier_param}, barrier_param
)
    nrlg.barrier_param = barrier_param
    return nrlg
end

function (nrlg::NegativeReducedLagrangianGrad)(N::AbstractManifold, q)
    M = N[1]
    cmo = nrlg.cmo
    p, μ, λ, s = q[N, 1], nrlg.μ, q[N, 2], nrlg.s
    b = nrlg.barrier_param
    m, n = length(μ), length(λ)
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)
    grad_g = get_grad_inequality_constraints(M, cmo, p)
    grad_h = get_grad_equality_constraints(M, cmo, p)
    X = allocate_result(N, rand)
    copyto!(M, X[N, 1], get_gradient(M, cmo, p))
    ν = μ + (μ .* g .+ b) ./ s
    (m > 0) && (X[N, 1] += sum(grad_g[i] * ν[i] for i in 1:m))
    (n > 0) && (X[N, 1] += sum(grad_h[j] * λ[j] for j in 1:n))
    (n > 0) && (copyto!(ℝ^n, X[N, 2], h))
    return -X
end

mutable struct ReducedLagrangianHess{T}
    cmo::ConstrainedManifoldObjective
    μ::T
    λ::T
    s::T
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:μ}, μ)
    rlh.μ = μ
    return rlh
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:λ}, λ)
    rlh.λ = λ
    return rlh
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:s}, s)
    rlh.s = s
    return rlh
end

function (rlh::ReducedLagrangianHess)(N::AbstractManifold, q, Y)
    M = N[1]
    cmo = rlh.cmo
    p, μ, λ, s = q[N, 1], rlh.μ, q[N, 2], rlh.s
    m, n = length(μ), length(λ)
    Yp, Yλ = Y[N, 1], Y[N, 2]
    grad_g = get_grad_inequality_constraint(M, cmo, p, :)
    grad_h = get_grad_equality_constraint(M, cmo, p, :)
    X = allocate_result(N, rand)
    copyto!(M, X[N, 1], get_hessian(M, cmo, p, Yp))
    if m > 0
        H_g = get_hess_inequality_constraint(M, cmo, p, Yp, :)
        X[N, 1] += sum([μ[i] * H_g[i] for i in 1:m])
        X[N, 1] += sum([μ[i] / s[i] * inner(M, p, grad_g[i], Yp) * grad_g[i] for i in 1:m])
    end
    if n > 0
        H_h = get_hess_equality_constraint(M, cmo, p, Yp, :)
        X[N, 1] += sum([λ[j] * H_h[j] for j in 1:n])
        X[N, 1] += sum([Yλ[j] * grad_h[j] for j in 1:n])
        copyto!(ℝ^n, X[N, 2], [inner(M, p, grad_h[j], Yp) for j in 1:n])
    end
    return X
end

function MeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, q)
    return MeritFunction(N, cmo, q[N, 1], q[N, 2], q[N, 3], q[N, 4])
end
function MeritFunction(N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, q)
    return MeritFunction(N, get_objective(cmo, true), q[N, 1], q[N, 2], q[N, 3], q[N, 4])
end
function MeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s)
    M = N[1]
    m, n = length(μ), length(λ)
    g = get_inequality_constraint(M, cmo, p, :)
    h = get_equality_constraint(M, cmo, p, :)
    grad_g = get_grad_inequality_constraint(M, cmo, p, :)
    grad_h = get_grad_equality_constraint(M, cmo, p, :)
    grad_L = get_gradient(M, cmo, p)
    (m > 0) && (grad_L += sum([μ[i] * grad_g[i] for i in 1:m]))
    (n > 0) && (grad_L += sum([λ[j] * grad_h[j] for j in 1:n]))
    ϕ = inner(M, p, grad_L, grad_L)
    (m > 0) && (ϕ += norm(g + s)^2 + norm(μ .* s)^2)
    (n > 0) && (ϕ += norm(h)^2)
    return ϕ
end

function calculate_σ(
    N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, p, μ, λ, s
)
    return calculate_σ(N, get_objective(cmo, true), p, μ, λ, s)
end
function calculate_σ(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s)
    N = M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s)
    q = allocate_result(N, rand)
    copyto!(N[1], q[N, 1], p)
    copyto!(N[2], q[N, 2], μ)
    copyto!(N[3], q[N, 3], λ)
    copyto!(N[4], q[N, 4], s)
    return min(0.5, MeritFunction(N, cmo, q)^(1 / 4))
end

function GradMeritFunction(N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, q)
    return GradMeritFunction(N, get_objective(cmo, true), q)
end
function GradMeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, q)
    M = N[1]
    p, μ, λ, s = q[N, 1], q[N, 2], q[N, 3], q[N, 4]
    m, n = length(μ), length(λ)
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)
    grad_g = get_grad_inequality_constraints(M, cmo, p)
    grad_h = get_grad_equality_constraints(M, cmo, p)
    grad_L = get_gradient(M, cmo, p)
    (m > 0) && (grad_L += sum([μ[i] * grad_g[i] for i in 1:m]))
    (n > 0) && (grad_L += sum([λ[j] * grad_h[j] for j in 1:n]))
    X = allocate_result(N, rand)
    copyto!(M, X[N, 1], get_hessian(M, cmo, p, grad_L))
    if m > 0
        H_g = get_hess_inequality_constraint(M, cmo, p, grad_L, :)
        X[N, 1] += sum([μ[i] * H_g[i] for i in 1:m])
        X[N, 1] += sum([(g + s)[i] * grad_g[i] for i in 1:m])
        copyto!(ℝ^m, X[N, 2], [inner(M, p, grad_g[i], grad_L) for i in 1:m] + μ .* s .* s)
        copyto!(ℝ^m, X[N, 4], g + s + μ .* μ .* s)
    end
    if n > 0
        H_h = get_hess_equality_constraint(M, cmo, p, grad_L, :)
        X[N, 1] += sum([λ[j] * H_h[j] for i in 1:n])
        X[N, 1] += sum([h[j] * grad_h[j] for j in 1:n])
        copyto!(ℝ^n, X[N, 3], [inner(M, p, grad_h[j], grad_L) for j in 1:n])
    end
    return 2 * X
end

mutable struct ConstraintLineSearchCheckFunction{CO}
    cmo::CO
    τ1::Float64
    τ2::Float64
    γ::Float64
end
function (clcf::ConstraintLineSearchCheckFunction)(N, q)
    #p = q[N,1]
    μ = q[N, 2]
    #λ = q[N,3]
    s = q[N, 4]
    (minimum(μ .* s) - clcf.γ * clcf.τ1 / length(μ) < 0) && return false
    (sum(μ .* s) - clcf.γ * clcf.τ2 * sqrt(MeritFunction(N, clcf.cmo, q)) < 0) &&
        return false
    return true
end
