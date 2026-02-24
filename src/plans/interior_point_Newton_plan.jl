"""
    StepsizeState{P,T} <: AbstractManoptSolverState

A state to store a point and a descent direction used within a linesearch,
if these are different from the iterate and search direction of the main solver.

# Fields

* `p::P`: a point on a manifold
* `X::T`: a tangent vector at `p`.

# Constructor

    StepsizeState(p,X)
    StepsizeState(M::AbstractManifold; p=rand(M), x=zero_vector(M,p)

# See also

[`interior_point_Newton`](@ref)
"""
struct StepsizeState{P, T} <: AbstractManoptSolverState
    p::P
    X::T
end
StepsizeState(M::AbstractManifold; p = rand(M), X = zero_vector(M, p)) = StepsizeState(p, X)
get_iterate(s::StepsizeState) = s.p
get_gradient(s::StepsizeState) = s.X
set_iterate!(s::StepsizeState, M, p) = copyto!(M, s.p, p)
set_gradient!(s::StepsizeState, M, p, X) = copyto!(M, s.X, p, X)

@doc """
    InteriorPointNewtonState{P,T} <: AbstractHessianSolverState

# Fields

* `λ`:           the Lagrange multiplier with respect to the equality constraints
* `μ`:           the Lagrange multiplier with respect to the inequality constraints
$(_fields(:p; add_properties = [:as_Iterate]))
* `s`:           the current slack variable
$(_fields(:sub_problem))
$(_fields(:sub_state))
* `X`:           the current gradient with respect to `p`
* `Y`:           the current gradient with respect to `μ`
* `Z`:           the current gradient with respect to `λ`
* `W`:           the current gradient with respect to `s`
* `ρ`:           store the orthogonality `μ's/m` to compute the barrier parameter `β` in the sub problem
* `σ`:           scaling factor for the barrier parameter `β` in the sub problem
$(_fields(:stopping_criterion; name = "stop"))
$(_fields([:retraction_method, :stepsize]))
* `step_problem`: an [`AbstractManoptProblem`](@ref) storing the manifold and objective for the line search
* `step_state`: storing iterate and search direction in a state for the line search, see [`StepsizeState`](@ref)

# Constructor

    InteriorPointNewtonState(
        M::AbstractManifold, cmo::ConstrainedManifoldObjective, sub_problem::Pr, sub_state::St;
        kwargs...
    )
    InteriorPointNewtonState(
        M::AbstractManifold, cmo::ConstrainedManifoldObjective, sub_problem::Pr;
        evaluation = AllocatingEvaluation(), kwargs...
    )
    InteriorPointNewtonState(
        sub_problem::Pr, sub_state::St;
        kwargs...
    )

Initialize the state, where both the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) and the [`ConstrainedManifoldObjective`](@ref)
are used to fill in reasonable defaults for the keywords.
For a closed form solution of the sub solver, you can provide the evaluation either as `St` in the first
constructor or as a keyword like in the second.
The third constructor is considered an internal constructor accepting the same keywords,
but those that are filled by defaults based on `M` or `cmo` become mandatory

# Input

$(_args(:M))
* `cmo`:         a [`ConstrainedManifoldObjective`](@ref)
$(_args([:sub_problem, :sub_state]))

# Keyword arguments

Let `m` and `n` denote the number of inequality and equality constraints, respectively

$(_kwargs(:p; add_properties = [:as_Initial]))
* `μ=ones(m)` Lagrange multipliers for the inequality constraints
* `λ=zeros(n)` Lagrange multipliers for the equality constraints
* `X=`[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`
* `Y=zero(μ)` tangent vector (gradient) for the inequality constraints
* `Z=zero(λ)` tangent vector (gradient) for the equality constraints
* `s=ones(m)` slack variables for the inequality constraints
* `W=zero(s)` tangent vector (gradient) for the slack variables
* `ρ=μ's/m`  storage for the orthogonality check
* `σ=`[`calculate_σ`](@ref)`(M, cmo, p, μ, λ, s)`
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`[` | `](@ref StopWhenAny)[`StopWhenChangeLess`](@ref)`(1e-8)"))
$(_kwargs(:retraction_method))
* `step_objective=`[`ManifoldGradientObjective`](@ref)`(`[`KKTVectorFieldNormSq`](@ref)`(cmo)`, [`KKTVectorFieldNormSqGradient`](@ref)`(cmo)`; evaluation=[`InplaceEvaluation`](@ref)`())`
* `vector_space=`[`Rn`](@ref Manopt.Rn): a function that, given an integer, returns the manifold to be used for the vector space components ``ℝ^m,ℝ^n``
* `step_problem`: wrap the manifold ``$(_math(:Manifold)) × ℝ^m × ℝ^n × ℝ^m``
* `step_state`: the [`StepsizeState`](@ref) with point and search direction
$(_kwargs(:stepsize; default = " `[`ArmijoLinesearch`](@ref)`()"))
  with the [`InteriorPointCentralityCondition`](@ref) as additional condition to accept a step"))
* `is_feasible_error=:error`: specify how to handle infeasible starting points, see [`is_feasible`](@ref) for options.

and internally `_step_M` and `_step_p` for the manifold and point in the stepsize.
"""
mutable struct InteriorPointNewtonState{
        P, T, Pr <: Union{AbstractManoptProblem, F} where {F}, St <: AbstractManoptSolverState,
        V, R <: Real,
        SC <: StoppingCriterion, TRTM <: AbstractRetractionMethod, TStepsize <: Stepsize,
        TStepPr <: AbstractManoptProblem, TStepSt <: AbstractManoptSolverState,
    } <: AbstractHessianSolverState
    p::P
    X::T
    sub_problem::Pr
    sub_state::St
    μ::V
    λ::V
    s::V
    Y::V
    Z::V
    W::V
    ρ::R
    σ::R
    stop::SC
    retraction_method::TRTM
    stepsize::TStepsize
    step_problem::TStepPr
    step_state::TStepSt
    is_feasible_error::Symbol
    function InteriorPointNewtonState(
            sub_problem::Pr, sub_state::St;
            p::P, X::T, μ::V, λ::V, s::V,
            ρ::R, σ::R, is_feasible_error::Symbol = :error,
            Y::V = zero(μ), Z::V = zero(λ), W::V = zero(s),
            stopping_criterion::SC = StopAfterIteration(200) | StopWhenChangeLess(1.0e-8),
            retraction_method::RTM,
            step_problem::StepPr, step_state::StepSt, stepsize::S, kwargs...
        ) where {
            P, T, V, R,
            Pr <: Union{AbstractManoptProblem, F} where {F}, St <: AbstractManoptSolverState,
            StepPr <: AbstractManoptProblem, StepSt <: AbstractManoptSolverState,
            SC <: StoppingCriterion, RTM <: AbstractRetractionMethod, S <: Stepsize,
        }
        ips = new{P, T, Pr, St, V, R, SC, RTM, S, StepPr, StepSt}()
        ips.p = p; ips.sub_problem = sub_problem; ips.sub_state = sub_state
        ips.μ = μ; ips.λ = λ; ips.s = s; ips.ρ = ρ; ips.σ = σ
        ips.X = X; ips.Y = Y; ips.Z = Z; ips.W = W
        ips.stop = stopping_criterion
        ips.retraction_method = retraction_method
        ips.stepsize = stepsize
        ips.step_problem = step_problem; ips.step_state = step_state
        ips.is_feasible_error = is_feasible_error
        return ips
    end
    function InteriorPointNewtonState(
            M::AbstractManifold, cmo::ConstrainedManifoldObjective, sub_problem::Pr, sub_state::St;
            p = rand(M), X = zero_vector(M, p),
            μ = ones(length(get_inequality_constraint(M, cmo, p, :))),
            λ = zeros(length(get_equality_constraint(M, cmo, p, :))),
            s = ones(length(get_inequality_constraint(M, cmo, p, :))),
            ρ = μ's / length(get_inequality_constraint(M, cmo, p, :)),
            σ = calculate_σ(M, cmo, p, μ, λ, s),
            retraction_method::RTM = default_retraction_method(M),
            step_objective = ManifoldGradientObjective(
                KKTVectorFieldNormSq(cmo), KKTVectorFieldNormSqGradient(cmo);
                evaluation = InplaceEvaluation(),
            ),
            vector_space = Rn,
            _step_M = M × vector_space(length(μ)) × vector_space(length(λ)) × vector_space(length(s)),
            step_problem::StepPr = DefaultManoptProblem(_step_M, step_objective),
            _step_p = rand(_step_M),
            step_state::StepSt = StepsizeState(_step_p, zero_vector(_step_M, _step_p)),
            centrality_condition = (N, p) -> true,
            stepsize::S = ArmijoLinesearchStepsize(
                get_manifold(step_problem);
                retraction_method = default_retraction_method(get_manifold(step_problem)),
                initial_stepsize = 1.0, additional_decrease_condition = centrality_condition,
            ),
            kwargs...,
        ) where {
            Pr <: Union{AbstractManoptProblem, F} where {F}, St <: AbstractManoptSolverState,
            RTM <: AbstractRetractionMethod, S <: Stepsize,
            StepPr <: AbstractManoptProblem, StepSt <: AbstractManoptSolverState,
        }
        return InteriorPointNewtonState(
            sub_problem, sub_state;
            p = p, X = X, μ = μ, λ = λ, s = s, ρ = ρ, σ = σ, retraction_method = retraction_method,
            step_problem = step_problem, step_state, stepsize = stepsize,
            kwargs...
        )
    end
end
function InteriorPointNewtonState(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        sub_problem;
        evaluation::E = AllocatingEvaluation(),
        kwargs...,
    ) where {E <: AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation = evaluation)
    return InteriorPointNewtonState(M, cmo, sub_problem, cfs; kwargs...)
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
function status_summary(ips::InteriorPointNewtonState; context = :default)
    i = get_count(ips, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ips.stop) ? "Yes" : "No"
    _is_inline(context) && (return "$(repr(ips)) – $(Iter) $(has_converged(ips) ? "(converged)" : "")")
    s = """
    # Solver state for `Manopt.jl`s Interior Point Newton Method
    $Iter
    ## Parameters
    * ρ: $(ips.ρ)
    * σ: $(ips.σ)
    * retraction method: $(ips.retraction_method)

    ## Stepsize
    $(status_summary(ips.stepsize; context = context))

    ## Stopping criterion
    $(status_summary(ips.stop; context = context))
    This indicates convergence: $Conv"""
    return s
end
function Base.show(io::IO, ipns::InteriorPointNewtonState)
    print(io, "InteriorPointNewtonState(")
    print(io, repr(ipns.sub_problem)); print(io, ", "); print(io, repr(ipns.sub_state)); print(io, ";\n")
    print(io, "is_feasibility_error="); print(io, ipns.is_feasible_error)
    print(io, ",retraction_method = $(ipns.retraction_method)")
    print(io, ",\np = "); print(io, ipns.p); print(io, ", X = "); print(io, ipns.X)
    print(io, ", μ = "); print(io, ipns.μ); print(io, ", Y = "); print(io, ipns.Y)
    print(io, ", λ = "); print(io, ipns.λ); print(io, ", Z = "); print(io, ipns.Z)
    print(io, ", s = "); print(io, ipns.s); print(io, ", W = "); print(io, ipns.W)
    print(io, ", ρ = "); print(io, ipns.ρ); print(io, ", σ = "); print(io, ipns.σ)
    print(io, ",\nstep_problem = "); print(io, ipns.step_problem)
    print(io, ", step_state = ")
    return print(io, ipns.step_state)
end
#
# Constraint functors
#

@doc """
    CondensedKKTVectorField{O<:ConstrainedManifoldObjective,T,R} <: AbstractConstrainedSlackFunctor{T,R}

Given the constrained optimization problem

```math
$(
    _tex(
        :aligned,
        "$(_tex(:min))_{p ∈ $(_math(:Manifold))nifold)))} & f(p)",
        "$(_tex(:text, " subject to ")) &g_i(p) ≤ 0 $(_tex(:quad)) $(_tex(:text, " for ")) i= 1, …, m,",
        "$(_tex(:quad)) & h_j(p) = 0 $(_tex(:quad))$(_tex(:text, " for ")) j=1,…,n,",
    )
)
```

Then reformulating the KKT conditions of the Lagrangian
from the optimality conditions of the Lagrangian

```math
$(_tex(:Cal, "L"))(p, μ, λ) = f(p) + $(_tex(:sum, "j=1", "n")) λ_jh_j(p) + $(_tex(:sum, "i=1", "m")) μ_ig_i(p)
```

in a perturbed / barrier method in a condensed form
using a slack variable ``s ∈ ℝ^m`` and a barrier parameter ``β``
and the Riemannian gradient of the Lagrangian with respect to the first parameter
``$(_tex(:grad))_p L(p, μ, λ)``.

Let ``$(_tex(:Cal, "N")) = $(_math(:Manifold))) × ℝ^n``. We obtain the linear system

```math
$(_tex(:Cal, "A"))(p,λ)[X,Y] = -b(p,λ),$(_tex(:qquad)) $(_tex(:text, "where ")) (X,Y) ∈ T_{(p,λ)}$(_tex(:Cal, "N"))
```

where ``$(_tex(:Cal, "A")): T_{(p,λ)}$(_tex(:Cal, "N")) → T_{(p,λ)}$(_tex(:Cal, "N"))`` is a linear operator and
this struct models the right hand side ``b(p,λ) ∈ T_{(p,λ)}$(_math(:Manifold)))`` given by

```math
b(p,λ) = $(
    _tex(
        :pmatrix,
        "$(_tex(:grad)) f(p) + $(_tex(:displaystyle))$(_tex(:sum, "j=1", "n")) λ_j $(_tex(:grad)) h_j(p) + $(_tex(:displaystyle))$(_tex(:sum, "i=1", "m")) μ_i $(_tex(:grad)) g_i(p) + $(_tex(:displaystyle))$(_tex(:sum, "i=1", "m")) $(_tex(:frac, "μ_i", "s_i"))$(_tex(:bigl))( μ_i(g_i(p)+s_i) + β - μ_is_i $(_tex(:bigr)))$(_tex(:grad)) g_i(p)",
        "h(p)",
    )
)
```

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)
* `μ::T` the vector in ``ℝ^m`` of coefficients for the inequality constraints
* `s::T` the vector in ``ℝ^m`` of sclack variables
* `β::R` the barrier parameter ``β∈ℝ``

# Constructor

    CondensedKKTVectorField(cmo, μ, s, β)
"""
mutable struct CondensedKKTVectorField{O <: ConstrainedManifoldObjective, T, R} <: AbstractConstrainedSlackFunctor{T, R}
    cmo::O
    μ::T
    s::T
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
    Y1, Y2 = submanifold_components(N, Y)
    # First term of the lagrangian
    get_gradient!(M, Y1, cmo, p) #grad f
    X = zero_vector(M, p)
    for i in 1:m
        get_grad_inequality_constraint!(M, X, cKKTvf.cmo, p, i)
        gi = get_inequality_constraint(M, cKKTvf.cmo, p, i)
        # Lagrangian term
        Y1 .+= μ[i] .* X
        #
        Y1 .+= (μ[i] / s[i]) * (μ[i] * (gi + s[i]) + β - μ[i] * s[i]) .* X
    end
    for j in 1:n
        get_grad_equality_constraint!(M, X, cKKTvf.cmo, p, j)
        hj = get_equality_constraint(M, cKKTvf.cmo, p, j)
        Y1 .+= λ[j] .* X
        Y2[j] = hj
    end
    return Y
end
function status_summary(CKKTvf::CondensedKKTVectorField; context = :default)
    _is_inline(context) && (return repr(CKKTvf))
    return """
    The condensed KKT vector field for the constrained objective
    $(_MANOPT_INDENT)$(replace(status_summary(CKKTvf.cmo; context = context), "\n" => "\n$(_MANOPT_INDENT)"))
    with μ=$(CKKTvf.μ) s=$(CKKTvf.s) β=$(CKKTvf.β)"""
end
function show(io::IO, CKKTvf::CondensedKKTVectorField)
    print(io, "CondensedKKTVectorField(")
    print(io, CKKTvf.cmo)
    return print(io, ", $(CKKTvf.μ), $(CKKTvf.s), $(CKKTvf.β))")
end

@doc """
    CondensedKKTVectorFieldJacobian{O<:ConstrainedManifoldObjective,T,R}  <: AbstractConstrainedSlackFunctor{T,R}

Given the constrained optimization problem

```math
$(
    _tex(
        :aligned,
        "$(_tex(:min))_{p ∈ $(_math(:Manifold)))} & f(p)",
        "$(_tex(:text, "subject to")) & g_i(p) ≤ 0 $(_tex(:quad))$(_tex(:text, " for ")) i= 1, …, m,",
        "$(_tex(:quad)) & h_j(p)=0 $(_tex(:quad)) $(_tex(:text, " for ")) j=1,…,n,",
    )
)
```

we reformulate the KKT conditions of the Lagrangian
from the optimality conditions of the Lagrangian

```math
$(_tex(:Cal, "L"))(p, μ, λ) = f(p) + $(_tex(:sum, "j=1", "n")) λ_jh_j(p) +$(_tex(:sum, "i=1", "m")) μ_ig_i(p)
```

in a perturbed / barrier method enhanced as well as condensed form as using ``$(_tex(:grad))_o L(p, μ, λ)``
the Riemannian gradient of the Lagrangian with respect to the first parameter.

Let ``$(_tex(:Cal, "N")) = $(_math(:Manifold))) × ℝ^n``. We obtain the linear system

```math
$(_tex(:Cal, "A"))(p,λ)[X,Y] = -b(p,λ),$(_tex(:qquad)) $(_tex(:text, "where ")) X ∈ T_p$(_math(:Manifold))), Y ∈ ℝ^n
```
where ``$(_tex(:Cal, "A")): T_{(p,λ)}$(_tex(:Cal, "N")) → T_{(p,λ)}$(_tex(:Cal, "N"))`` is a linear operator
on ``T_{(p,λ)}$(_tex(:Cal, "N")) = T_p$(_math(:Manifold))) × ℝ^n`` given by

```math
$(_tex(:Cal, "A"))(p,λ)[X,Y] =
$(
    _tex(
        :pmatrix,
        "$(_tex(:Hess))_p$(_tex(:Cal, "L"))(p, μ, λ)[X] + $(_tex(:displaystyle))$(_tex(:sum, "i=1", "m")) $(_tex(:frac, "μ_i", "s_i")) ⟨$(_tex(:grad)) g_i(p), X⟩$(_tex(:grad)) g_i(p) + $(_tex(:displaystyle))$(_tex(:sum, "j=1", "n")) Y_j $(_tex(:grad)) h_j(p)",
        "$(_tex(:Bigl))( ⟨$(_tex(:grad)) h_j(p), X⟩ $(_tex(:Bigr)))_{j=1}^n",
    )
)
```

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)
* `μ::V` the vector in ``ℝ^m`` of coefficients for the inequality constraints
* `s::V` the vector in ``ℝ^m`` of slack variables
* `β::R` the barrier parameter ``β∈ℝ``

# Constructor

    CondensedKKTVectorFieldJacobian(cmo, μ, s, β)
"""
mutable struct CondensedKKTVectorFieldJacobian{O <: ConstrainedManifoldObjective, T, R} <: AbstractConstrainedSlackFunctor{T, R}
    cmo::O
    μ::T
    s::T
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
    Y1, Y2 = submanifold_components(N, Y)
    copyto!(M, Y1, get_hessian(M, cKKTvfJ.cmo, p, Xp)) # Hess f
    # Build the rest iteratively
    for i in 1:m #ineq
        get_hess_inequality_constraint!(M, Xt, cKKTvfJ.cmo, p, Xp, i)
        # Summand of Hess L
        Y1 .+= μ[i] .* Xt
        get_grad_inequality_constraint!(M, Xt, cKKTvfJ.cmo, p, i)
        # condensed term
        Y1 .+= ((μ[i] / s[i]) * inner(M, p, Xt, Xp)) .* Xt
    end
    for j in 1:n #eq
        get_hess_equality_constraint!(M, Xt, cKKTvfJ.cmo, p, Xp, j)
        # Summand of Hess L
        Y1 .+= λ[j] .* Xt
        get_grad_equality_constraint!(M, Xt, cKKTvfJ.cmo, p, j)
        # condensed term
        Y1 .+= Xλ[j] .* Xt
        # condensed term in second part
        Y2[j] = inner(M, p, Xt, Xp)
    end
    return Y
end
function status_summary(CKKTvfJ::CondensedKKTVectorFieldJacobian; context = :default)
    _is_inline(context) && (return repr(CKKTvfJ))
    return """
    The Jacobian of the condensed KKT vector field for the constrained objective
    $(_MANOPT_INDENT)$(replace(status_summary(CKKTvfJ.cmo; context = context), "\n" => "\n$(_MANOPT_INDENT)"))
    with μ=$(CKKTvfJ.μ) s=$(CKKTvfJ.s) β=$(CKKTvfJ.β)"""
end
function show(io::IO, CKKTvfJ::CondensedKKTVectorFieldJacobian)
    print(io, "CondensedKKTVectorFieldJacobian(")
    print(io, CKKTvfJ.cmo)
    return print(io, ", $(CKKTvfJ.μ), $(CKKTvfJ.s), $(CKKTvfJ.β))")
end

@doc """
    KKTVectorField{O<:ConstrainedManifoldObjective}

Implement the vector field ``F`` KKT-conditions, including a slack variable
for the inequality constraints.

Given the [`LagrangianCost`](@ref)

```math
$(_tex(:Cal, "L"))(p; μ, λ) = f(p) + $(_tex(:sum, "i=1", "m")) μ_ig_i(p) + $(_tex(:sum, "j=1", "n")) λ_jh_j(p)
```

the [`LagrangianGradient`](@ref)

```math
$(_tex(:grad))$(_tex(:Cal, "L"))(p, μ, λ) = $(_tex(:grad))f(p) + $(_tex(:sum, "j=1", "n")) λ_j $(_tex(:grad)) h_j(p) + $(_tex(:sum, "i=1", "m")) μ_i $(_tex(:grad)) g_i(p),
```

and introducing the slack variables ``s=-g(p) ∈ ℝ^m``
the vector field is given by

```math
F(p, μ, λ, s) = $(
    _tex(
        :pmatrix,
        "$(_tex(:grad))_p $(_tex(:Cal, "L"))(p, μ, λ)",
        "g(p) + s",
        "h(p)",
        "μ ⊙ s",
    )
),
```
where ``p ∈ $(_math(:Manifold))nifold))nifold)))``, ``μ, s ∈ ℝ^m`` and ``λ ∈ ℝ^n``,
and ``⊙`` denotes the Hadamard (or elementwise) product

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

While the point `p` is arbitrary and usually not needed, it serves as internal memory
in the computations. Furthermore Both fields together also clarify the product manifold structure to use.

# Constructor

    KKTVectorField(cmo::ConstrainedManifoldObjective)

# Example

Define `F = KKTVectorField(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``$(_math(:Manifold))nifold)))×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `F(N, q)` or as the in-place variant `F(N, Y, q)`,
where `q` is a point on `N` and `Y` is a tangent vector at `q` for the result.
"""
struct KKTVectorField{O <: ConstrainedManifoldObjective}
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
    _q1, μ, λ, s = submanifold_components(N, q)
    Y1, Y2, Y3, Y4 = submanifold_components(N, Y)
    LagrangianGradient(KKTvf.cmo, μ, λ)(M, Y[N, 1], p)
    m, n = length(μ), length(λ)
    (m > 0) && (Y2 .= get_inequality_constraint(M, KKTvf.cmo, p, :) .+ s)
    (n > 0) && (Y3 .= get_equality_constraint(M, KKTvf.cmo, p, :))
    (m > 0) && (Y4 .= μ .* s)
    return Y
end
function show(io::IO, KKTvf::KKTVectorField)
    print(io, "KKTVectorField(")
    print(io, KKTvf.cmo)
    return print(io, ")")
end
function status_summary(KKTvf::KKTVectorField; context = :default)
    _is_inline(context) && (return repr(KKTvf))
    return "The KKT vector field for the constrained objective\n$(_MANOPT_INDENT)$(status_summary(KKTvf.cmo; context = context))"
end

@doc """
    KKTVectorFieldJacobian{O<:ConstrainedManifoldObjective}

Implement the Jacobian of the vector field ``F`` of the KKT-conditions, including a slack variable
for the inequality constraints, see [`KKTVectorField`](@ref) and [`KKTVectorFieldAdjointJacobian`](@ref)..

```math
$(_tex(:operatorname, "J")) F(p, μ, λ, s)[X, Y, Z, W] =
$(
    _tex(
        :pmatrix,
        "$(_tex(:Hess))_p $(_tex(:Cal, "L"))(p, μ, λ)[X] + $(_tex(:displaystyle))$(_tex(:sum, "i=1", "m")) Y_i $(_tex(:grad)) g_i(p) + $(_tex(:displaystyle))$(_tex(:sum, "j=1", "n")) Z_j $(_tex(:grad)) h_j(p)",
        "$(_tex(:Bigl))( ⟨$(_tex(:grad)) g_i(p), X⟩ + W_i$(_tex(:Bigr)))_{i=1}^m",
        "$(_tex(:Bigl))( ⟨$(_tex(:grad)) h_j(p), X⟩ $(_tex(:Bigr)))_{j=1}^n",
        "μ ⊙ W + s ⊙ Y",
    )
)
```
where ``⊙`` denotes the Hadamard (or elementwise) product

See also the [`LagrangianHessian`](@ref) ``$(_tex(:Hess))_p $(_tex(:Cal, "L"))(p, μ, λ)[X]``.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldJacobian(cmo::ConstrainedManifoldObjective)

Generate the Jacobian of the KKT vector field related to some [`ConstrainedManifoldObjective`](@ref) `cmo`.

# Example

Define `JF = KKTVectorFieldJacobian(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``$(_math(:Manifold))nifold)))×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `JF(N, q, Y)` or as the in-place variant `JF(N, Z, q, Y)`,
where `q` is a point on `N` and `Y` and `Z` are a tangent vector at `q`.
"""
mutable struct KKTVectorFieldJacobian{O <: ConstrainedManifoldObjective}
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

    Y1, Y2, Y3, Y4 = submanifold_components(N, Y)
    Z1, Z2, Z3, Z4 = submanifold_components(N, Z)
    # First component
    LagrangianHessian(KKTvfJ.cmo, μ, λ)(M, Z1, p, Y1)
    Xt = copy(M, p, X)
    m, n = length(μ), length(λ)
    for i in 1:m
        get_grad_inequality_constraint!(M, Xt, KKTvfJ.cmo, p, i)
        Z1 .+= Y2[i] .* Xt
        # set second components as well
        Z2[i] = inner(M, p, Xt, X) + Y4[i]
    end
    for j in 1:n
        get_grad_equality_constraint!(M, Xt, KKTvfJ.cmo, p, j)
        Z1 .+= Y3[j] .* Xt
        # set third components as well
        Z3[j] = inner(M, p, Xt, X)
    end
    # Fourth component
    Z4 .= μ .* Y4 .+ s .* Y2
    return Z
end
function show(io::IO, KKTvfJ::KKTVectorFieldJacobian)
    print(io, "KKTVectorFieldJacobian(")
    print(io, KKTvfJ.cmo)
    return print(io, ")")
end
function status_summary(KKTvfJ::KKTVectorFieldJacobian; context = :default)
    _is_inline(context) && (return repr(KKTvfJ))
    return "The Jacobian of the KKT vector field for the constrained objective\n$(_MANOPT_INDENT)$(status_summary(KKTvfJ.cmo; context = context))"
end

@doc """
    KKTVectorFieldAdjointJacobian{O<:ConstrainedManifoldObjective}

Implement the Adjoint of the Jacobian of the vector field ``F`` of the KKT-conditions, including a slack variable
for the inequality constraints, see [`KKTVectorField`](@ref) and [`KKTVectorFieldJacobian`](@ref).

```math
$(_tex(:operatorname, "J"))^*
F(p, μ, λ, s)[X, Y, Z, W] = $(
    _tex(
        :pmatrix,
        "$(_tex(:Hess))_p $(_tex(:Cal, "L"))(p, μ, λ)[X] + $(_tex(:displaystyle))$(_tex(:sum, "i=1", "m")) Y_i $(_tex(:grad)) g_i(p) + $(_tex(:displaystyle))$(_tex(:sum, "j=1", "n")) Z_j $(_tex(:grad)) h_j(p)",
        "$(_tex(:Bigl))( ⟨$(_tex(:grad)) g_i(p), X⟩ + s_iW_i$(_tex(:Bigr)))_{i=1}^m",
        "$(_tex(:Bigl))( ⟨$(_tex(:grad)) h_j(p), X⟩ $(_tex(:Bigr)))_{j=1}^n",
        "μ ⊙ W + Y"
    )
),
```
where ``⊙`` denotes the Hadamard (or elementwise) product

See also the [`LagrangianHessian`](@ref) ``$(_tex(:Hess))_p $(_tex(:Cal, "L"))(p, μ, λ)[X]``.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldAdjointJacobian(cmo::ConstrainedManifoldObjective)

Generate the Adjoint Jacobian of the KKT vector field related to some [`ConstrainedManifoldObjective`](@ref) `cmo`.

# Example

Define `AdJF = KKTVectorFieldAdjointJacobian(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``$(_math(:Manifold)))×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `AdJF(N, q, Y)` or as the in-place variant `AdJF(N, Z, q, Y)`,
where `q` is a point on `N` and `Y` and `Z` are a tangent vector at `q`.
"""
mutable struct KKTVectorFieldAdjointJacobian{O <: ConstrainedManifoldObjective}
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
    Z1, Z2, Z3, Z4 = submanifold_components(N, Z)
    Y1, Y2, Y3, Y4 = submanifold_components(N, Y)
    for i in 1:m
        get_grad_inequality_constraint!(M, Xt, KKTvfAdJ.cmo, p, i)
        Z1 .+= Y2[i] .* Xt
        # set second components as well
        Z2[i] = inner(M, p, Xt, X) + s[i] * Y4[i]
    end
    for j in 1:n
        get_grad_equality_constraint!(M, Xt, KKTvfAdJ.cmo, p, j)
        Z1 .+= Y3[j] .* Xt
        # set third components as well
        Z3[j] = inner(M, p, Xt, X)
    end
    # Fourth component
    Z4 .= μ .* Y4 .+ Y2
    return Z
end
function show(io::IO, KKTvfAdJ::KKTVectorFieldAdjointJacobian)
    print(io, "KKTVectorFieldAdjointJacobian(")
    print(io, KKTvfAdJ.cmo)
    return print(io, ")")
end
function status_summary(KKTvfAdJ::KKTVectorFieldAdjointJacobian; context = :default)
    _is_inline(context) && (return repr(KKTvfAdJ))
    return "The adjoint Jacobian of the KKT vector field for the constrained objective\n$(_MANOPT_INDENT)$(status_summary(KKTvfAdJ.cmo; context = context))"
end

@doc """
    KKTVectorFieldNormSq{O<:ConstrainedManifoldObjective}

Implement the square of the norm of the vector field ``F`` of the KKT-conditions, including a slack variable
for the inequality constraints, see [`KKTVectorField`](@ref), where this functor applies the norm to.
In [LaiYoshise:2024](@cite) this is called the merit function.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldNormSq(cmo::ConstrainedManifoldObjective)

# Example

Define `f = KKTVectorFieldNormSq(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``$(_math(:Manifold))nifold)))×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `f(N, q)`, where `q` is a point on `N`.
"""
mutable struct KKTVectorFieldNormSq{O <: ConstrainedManifoldObjective}
    cmo::O
end
function (KKTvc::KKTVectorFieldNormSq)(N, q)
    Y = zero_vector(N, q)
    KKTVectorField(KKTvc.cmo)(N, Y, q)
    return inner(N, q, Y, Y)
end
function show(io::IO, KKTvfNSq::KKTVectorFieldNormSq)
    print(io, "KKTVectorFieldNormSq(")
    print(io, KKTvfNSq.cmo)
    return print(io, ")")
end
function status_summary(KKTvfNSq::KKTVectorFieldNormSq; context = :default)
    _is_inline(context) && (return repr(KKTvfNSq))
    return "The KKT vector field in normed squared for the constrained objective\n$(_MANOPT_INDENT)$(status_summary(KKTvfNSq.cmo; context = context))"
end

@doc """
    KKTVectorFieldNormSqGradient{O<:ConstrainedManifoldObjective}

Compute the gradient of the [`KKTVectorFieldNormSq`](@ref) ``φ(p,μ,λ,s) = $(_tex(:norm, "F(p,μ,λ,s)"))^2``,
that is of the norm squared of the [`KKTVectorField`](@ref) ``F``.

This is given in [LaiYoshise:2024](@cite) as the gradient of their merit function,
which we can write with the adjoint ``J^*`` of the Jacobian

```math
$(_tex(:grad)) φ = 2$(_tex(:operatorname, "J"))^* F(p, μ, λ, s)[F(p, μ, λ, s)],
```

and hence is computed with [`KKTVectorFieldAdjointJacobian`](@ref) and [`KKTVectorField`](@ref).

For completeness, the gradient reads, using the [`LagrangianGradient`](@ref) ``L = $(_tex(:grad))_p $(_tex(:Cal, "L"))(p,μ,λ) ∈ T_p$(_math(:Manifold)))``,
for a shorthand of the first component of ``F``, as

```math
$(_tex(:grad)) φ
=
2
$(
    _tex(
        :pmatrix,
        "$(_tex(:grad))_p $(_tex(:Cal, "L"))(p,μ,λ)[L] + (g_i(p) + s_i)$(_tex(:grad)) g_i(p) + h_j(p)$(_tex(:grad)) h_j(p)",
        "$(_tex(:Bigl))( ⟨$(_tex(:grad)) g_i(p), L⟩ + s_i$(_tex(:Bigr)))_{i=1}^m + μ ⊙ s ⊙ s",
        "$(_tex(:Bigl))( ⟨$(_tex(:grad)) h_j(p), L⟩ $(_tex(:Bigr)))_{j=1}^n",
        "g + s + μ ⊙ μ ⊙ s",
    )
),
```
where ``⊙`` denotes the Hadamard (or elementwise) product.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldNormSqGradient(cmo::ConstrainedManifoldObjective)

# Example

Define `grad_f = KKTVectorFieldNormSqGradient(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``$(_math(:Manifold)))×ℝ^m×ℝ^n×ℝ^m``.
Then, you can call this cost as `grad_f(N, q)` or as the in-place variant `grad_f(N, Y, q)`,
where `q` is a point on `N` and `Y` is a tangent vector at `q` returning the resulting gradient at.
"""
mutable struct KKTVectorFieldNormSqGradient{O <: ConstrainedManifoldObjective}
    cmo::O
end
function (KKTcfNG::KKTVectorFieldNormSqGradient)(N, q)
    Y = zero_vector(N, q)
    KKTcfNG(N, Y, q)
    return Y
end
function (KKTcfNG::KKTVectorFieldNormSqGradient)(N, Y, q)
    Z = allocate(N, Y)
    KKTVectorField(KKTcfNG.cmo)(N, Z, q)
    KKTVectorFieldAdjointJacobian(KKTcfNG.cmo)(N, Y, q, Z)
    Y .*= 2
    return Y
end
function show(io::IO, KKTvfNSqGrad::KKTVectorFieldNormSqGradient)
    print(io, "KKTVectorFieldNormSqGradient(")
    print(io, KKTvfNSqGrad.cmo)
    return print(io, ")")
end
function status_summary(KKTvfNSqGrad::KKTVectorFieldNormSqGradient; context = :default)
    _is_inline(context) && (return repr(KKTvfNSqGrad))
    return "The gradient of the KKT vector field in normed squared for the constrained objective\n$(_MANOPT_INDENT)$(status_summary(KKTvfNSqGrad.cmo; context = context))"
end
#
#
# A special linesearch for IP Newton
function interior_point_initial_guess(
        mp::AbstractManoptProblem, ips::StepsizeState, ::Int, l::R, η; kwargs...
    ) where {R <: Real}
    N = get_manifold(mp)
    Y = get_gradient(N, get_objective(mp), ips.p)
    grad_norm = norm(N, ips.p, Y)
    max_step = max_stepsize(N, ips.p)
    return ifelse(isfinite(max_step), min(l, max_step / grad_norm), l)
end

@doc """
    InteriorPointCentralityCondition{CO,R}

A functor to check the centrality condition.

In order to obtain a step in the linesearch performed within the [`interior_point_Newton`](@ref),
Section 6 of [LaiYoshise:2024](@cite) propose the following additional conditions to hold
inspired by the Euclidean case described in Section 6 [El-BakryTapiaTsuchiyaZhang:1996](@cite):

For a given [`ConstrainedManifoldObjective`](@ref) assume consider the [`KKTVectorField`](@ref) ``F``,
that is we are at a point ``q = (p, λ, μ, s)``  on ``$(_math(:Manifold))) × ℝ^m × ℝ^n × ℝ^m``and a search direction ``V = (X, Y, Z, W)``.

Then, let

```math
τ_1 = $(_tex(:frac, "m$(_tex(:min))$(_tex(:set, "μ ⊙ s"))", "μ^{$(_tex(:rm, "T"))}s"))
$(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad))
τ_2 = $(_tex(:frac, "μ^{$(_tex(:rm, "T"))}s", "$(_tex(:norm, "F(q)"))")),
```
where ``⊙`` denotes the Hadamard (or elementwise) product.

For a new candidate ``q(α) = $(_tex(:bigl))(p(α), λ(α), μ(α), s(α)$(_tex(:bigr)) := ($(_tex(:retr))_p(αX), λ+αY, μ+αZ, s+αW)``,
we then define two functions

```math
c_1(α) = $(_tex(:min))$(_tex(:set, "μ(α) ⊙ s(α)")) - $(_tex(:frac, "γτ_1 μ(α)^{$(_tex(:rm, "T"))}s(α)", "m"))
$(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad))
c_2(α) = μ(α)^{$(_tex(:rm, "T"))}s(α) – γτ_2 $(_tex(:norm, "F(q(α))")).
```

While the paper now states that the (Armijo) line search starts at a point
``$(_tex(:tilde)) α``, it is easier to include the condition that ``c_1(α) ≥ 0`` and ``c_2(α) ≥ 0``
into the line search as well.

The functor `InteriorPointCentralityCondition(cmo, γ, μ, s, normKKT)(N,qα)`
defined here evaluates this condition and returns true if both ``c_1`` and ``c_2`` are non-negative.

# Fields

* `cmo`: a [`ConstrainedManifoldObjective`](@ref)
* `γ`: a constant
* `τ1`, `τ2`: the constants given in the formula.

# Constructor

    InteriorPointCentralityCondition(cmo, γ)
    InteriorPointCentralityCondition(cmo, γ, τ1, τ2)

Initialise the centrality conditions.
The parameters `τ1`, `τ2` are initialise to zero if not provided.

!!! note

    Besides [`get_parameter`](@ref) for all three constants,
    and [`set_parameter!`](@ref) for ``γ``,
    to update ``τ_1`` and ``τ_2``, call `set_parameter(ipcc, :τ, N, q)` to update
    both ``τ_1`` and ``τ_2`` according to the formulae above.
"""
mutable struct InteriorPointCentralityCondition{CO, R}
    cmo::CO
    γ::R
    τ1::R
    τ2::R
end
function InteriorPointCentralityCondition(cmo::CO, γ::R) where {CO, R}
    return InteriorPointCentralityCondition{CO, R}(cmo, γ, zero(γ), zero(γ))
end
function (ipcc::InteriorPointCentralityCondition)(N, qα)
    μα = qα[N, 2]
    sα = qα[N, 4]
    m = length(μα)
    # f1 false
    (minimum(μα .* sα) - ipcc.γ * ipcc.τ1 * sum(μα .* sα) / m < 0) && return false
    normKKTqα = sqrt(KKTVectorFieldNormSq(ipcc.cmo)(N, qα))
    # f2 false
    (sum(μα .* sα) - ipcc.γ * ipcc.τ2 * normKKTqα < 0) && return false
    return true
end
function get_parameter(ipcc::InteriorPointCentralityCondition, ::Val{:γ})
    return ipcc.γ
end
function set_parameter!(ipcc::InteriorPointCentralityCondition, ::Val{:γ}, γ)
    ipcc.γ = γ
    return ipcc
end
function get_parameter(ipcc::InteriorPointCentralityCondition, ::Val{:τ1})
    return ipcc.τ1
end
function get_parameter(ipcc::InteriorPointCentralityCondition, ::Val{:τ2})
    return ipcc.τ2
end
function set_parameter!(ipcc::InteriorPointCentralityCondition, ::Val{:τ}, N, q)
    μ = q[N, 2]
    s = q[N, 4]
    m = length(μ)
    normKKTq = sqrt(KKTVectorFieldNormSq(ipcc.cmo)(N, q))
    ipcc.τ1 = m * minimum(μ .* s) / sum(μ .* s)
    ipcc.τ2 = sum(μ .* s) / normKKTq
    return ipcc
end

@doc """
    StopWhenKKTResidualLess <: StoppingCriterion

Stop when the KKT residual

```
r^2
= $(_tex(:norm, "$(_tex(:grad))_p $(_tex(:Cal, "L"))(p, μ, λ) "))^2
+ $(_tex(:sum, "i=1", "m")) [μ_i]_{-}^2 + [g_i(p)]_+^2 + $(_tex(:abs, "μ_i g_i(p)"))^2
+ $(_tex(:sum, "j=1", "n")) $(_tex(:abs, "h_i(p)"))^2.
```

is less than a given threshold ``r < ε``.
We use ``[v]_+ = $(_tex(:max))$(_tex(:set, "0,v"))`` and ``[v]_- = $(_tex(:min))$(_tex(:set, "0,t"))``
for the positive and negative part of ``v``, respectively

# Fields

* `ε`: a threshold
* `residual`: store the last residual if the stopping criterion is hit.
* `at_iteration`:
"""
mutable struct StopWhenKKTResidualLess{R} <: StoppingCriterion
    ε::R
    residual::R
    at_iteration::Int
    function StopWhenKKTResidualLess(ε::R) where {R}
        return new{R}(ε, zero(ε), -1)
    end
end
function (c::StopWhenKKTResidualLess)(
        amp::AbstractManoptProblem, ipns::InteriorPointNewtonState, k::Int
    )
    M = get_manifold(amp)
    (k <= 0) && return false
    # now k > 0
    # Check residual
    μ, λ, s, p = ipns.μ, ipns.λ, ipns.s, ipns.p
    c.residual = 0.0
    m, n = length(ipns.μ), length(ipns.λ)
    # First component
    c.residual += norm(M, p, LagrangianGradient(get_objective(amp), μ, λ)(M, p))
    # ineq constr part
    for i in 1:m
        gi = get_inequality_constraint(amp, ipns.p, i)
        c.residual += min(0.0, μ[i])^2 + max(gi, 0)^2 + abs(μ[i] * gi)^2
    end
    # eq constr part
    for j in 1:n
        hj = get_equality_constraint(amp, ipns.p, j)
        c.residual += abs(hj)^2
    end
    c.residual = sqrt(c.residual)
    if c.residual < c.ε
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenKKTResidualLess)
    if (c.at_iteration >= 0)
        return "After iteration #$(c.at_iteration) the algorithm stopped with a KKT residual $(c.residual) < $(c.ε).\n"
    end
    return ""
end
function status_summary(swrr::StopWhenKKTResidualLess; context = :default)
    has_stopped = (swrr.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "‖F(p, λ, μ)‖ < ε = $(swrr.ε):$(_MANOPT_INDENT)" : "Stop when the KKT resudual is less than ε = $(c.ε)\n$(_MANOPT_INDENT)") * s
end
indicates_convergence(::StopWhenKKTResidualLess) = true
function show(io::IO, c::StopWhenKKTResidualLess)
    return print(io, "StopWhenKKTResidualLess($(c.ε))")
end

# An internal function to compute the new σ
@doc """
    calculate_σ(M, cmo, p, μ, λ, s; kwargs...)

Compute the new ``σ`` factor for the barrier parameter in [`interior_point_Newton`](@ref) as

```math
$(_tex(:min))$(_tex(:set, "$(_tex(:frac, "1", "2")), $(_tex(:norm, "F(p; μ, λ, s)"))^{$(_tex(:frac, "1", "2"))}")),
```
where ``F`` is the KKT vector field, hence the [`KKTVectorFieldNormSq`](@ref) is used.

# Keyword arguments

* `vector_space=`[`Rn`](@ref Manopt.Rn) a function that, given an integer, returns the manifold to be used for the vector space components ``ℝ^m,ℝ^n``
* `N` the manifold ``$(_math(:Manifold))) × ℝ^m × ℝ^n × ℝ^m`` the vector field lives on (generated using `vector_space`)
* `q` provide memory on `N` for interims evaluation of the vector field
"""
function calculate_σ(
        N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, p, μ, λ, s; kwargs...
    )
    return calculate_σ(N, get_objective(cmo, true), p, μ, λ, s; kwargs...)
end
function calculate_σ(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        p,
        μ,
        λ,
        s;
        vector_space = Rn,
        N = ProductManifold(
            M,
            vector_space(length(μ)),
            vector_space(length(λ)),
            vector_space(length(s)),
        ),
        q = allocate_result(N, rand),
    )
    q1, q2, q3, q4 = submanifold_components(N, q)
    copyto!(N[1], q1, p)
    q2 .= μ
    q3 .= λ
    q4 .= s
    return min(0.5, (KKTVectorFieldNormSq(cmo)(N, q))^(1 / 4))
end
