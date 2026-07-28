@doc """
    CondensedKKTVectorField{O<:ConstrainedManifoldObjective,T,R} <: AbstractConstrainedSlackFunction{T,R}

Given the constrained optimization problem

```math
$(
    _tex(
        :aligned,
        "$(_tex(:min))_{p ∈ $(_math(:Manifold))} & f(p)",
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

Let ``$(_math(:Manifold, M = "N")) = $(_math(:Manifold)) × ℝ^n``. We obtain the linear system

```math
$(_tex(:Cal, "A"))(p,λ)[X,Y] = -b(p,λ),$(_tex(:qquad)) $(_tex(:text, "where ")) (X,Y) ∈ $(_math(:TangentSpace; p = "(p, λ)", M = "N"))
```

where ``$(_tex(:Cal, "A")): $(_math(:TangentSpace; p = "(p, λ)", M = "N")) → $(_math(:TangentSpace; p = "(p, λ)", M = "N"))`` is a linear operator and
this struct models the right hand side ``b(p,λ) ∈ T_{(p,λ)}$(_math(:Manifold))`` given by

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
mutable struct CondensedKKTVectorField{O <: ConstrainedManifoldObjective, T, R} <: AbstractConstrainedSlackFunction{T, R}
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
function status_summary(CKKTvf::CondensedKKTVectorField; context::Symbol = :default)
    _is_inline(context) && (return repr(CKKTvf))
    return """
    The condensed KKT vector field for the constrained objective
    $(_in_str(status_summary(CKKTvf.cmo; context = context); indent = 1))
    with μ=$(CKKTvf.μ) s=$(CKKTvf.s) β=$(CKKTvf.β)"""
end
function Base.show(io::IO, CKKTvf::CondensedKKTVectorField)
    print(io, "CondensedKKTVectorField(")
    print(io, CKKTvf.cmo)
    return print(io, ", $(CKKTvf.μ), $(CKKTvf.s), $(CKKTvf.β))")
end

@doc """
    CondensedKKTVectorFieldJacobian{O<:ConstrainedManifoldObjective,T,R}  <: AbstractConstrainedSlackFunction{T,R}

Given the constrained optimization problem

```math
$(
    _tex(
        :aligned,
        "$(_tex(:min))_{p ∈ $(_math(:Manifold))} & f(p)",
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

Let ``$(_math(:Manifold; M = "N")) = $(_math(:Manifold)) × ℝ^n``. We obtain the linear system

```math
$(_tex(:Cal, "A"))(p,λ)[X,Y] = -b(p,λ),$(_tex(:qquad)) $(_tex(:text, "where ")) X ∈ T_p$(_math(:Manifold)), Y ∈ ℝ^n
```
where ``$(_tex(:Cal, "A")): T_{(p,λ)}$(_math(:Manifold; M = "N")) → T_{(p,λ)}$(_math(:Manifold; M = "N"))`` is a linear operator
on ``T_{(p,λ)}$(_math(:Manifold; M = "N")) = T_p$(_math(:Manifold)) × ℝ^n`` given by

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
mutable struct CondensedKKTVectorFieldJacobian{O <: ConstrainedManifoldObjective, T, R} <: AbstractConstrainedSlackFunction{T, R}
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
function status_summary(CKKTvfJ::CondensedKKTVectorFieldJacobian; context::Symbol = :default)
    _is_inline(context) && (return repr(CKKTvfJ))
    return """
    The Jacobian of the condensed KKT vector field for the constrained objective
    $(_in_str(status_summary(CKKTvfJ.cmo; context = context); indent = 1))
    with μ=$(CKKTvfJ.μ) s=$(CKKTvfJ.s) β=$(CKKTvfJ.β)"""
end
function Base.show(io::IO, CKKTvfJ::CondensedKKTVectorFieldJacobian)
    print(io, "CondensedKKTVectorFieldJacobian(")
    print(io, CKKTvfJ.cmo)
    return print(io, ", $(CKKTvfJ.μ), $(CKKTvfJ.s), $(CKKTvfJ.β))")
end

"""
    abstract type SmoothingTechnique

Specify a smoothing technique, see for example [`ExactPenaltyCost`](@ref) and [`ExactPenaltyGrad`](@ref).
"""
abstract type SmoothingTechnique end

@doc """
    LogarithmicSumOfExponentials <: SmoothingTechnique

Specify a smoothing based on ``$(_tex(:max))$(_tex(:max, "a,b")) ≈ u $(_tex(:log))($(_tex(:rm, "e"))^{$(_tex(:frac, "a", "u"))}+$(_tex(:rm, "e"))^{$(_tex(:frac, "b", "u"))})``
for some ``u``.
"""
struct LogarithmicSumOfExponentials <: SmoothingTechnique end

@doc """
    LinearQuadraticHuber <: SmoothingTechnique

Specify a smoothing based on ``$(_tex(:max))$(_tex(:set, "0,x")) ≈ $(_tex(:Cal, "P"))(x,u)`` for some ``u``, where

```math
$(_tex(:Cal, "P")) = $(
    _tex(
        :cases,
        "0 & $(_tex(:text, " if ")) x ≤ 0,",
        "$(_tex(:frac, "x^2", "2u")) & $(_tex(:text, " if ")) 0 ≤ x ≤ u",
        "x-$(_tex(:frac, "u", "2")) & $(_tex(:text, " if ")) x ≥ u"
    )
)
```
"""
struct LinearQuadraticHuber <: SmoothingTechnique end

@doc """
    ExactPenaltyCost{S, Pr, R}

Represent the cost of the exact penalty method based on a [`ConstrainedManifoldObjective`](@ref) `P`
and a parameter ``ρ`` given by

```math
f(p) + ρ$(_tex(:Bigl))(
    $(_tex(:sum, "i=0", "m")) $(_tex(:max))$(_tex(:set, "0,g_i(p)")) + $(_tex(:sum, "j=0", "n")) $(_tex(:abs, "h_j(p)"))
$(_tex(:Bigr))),
```
where an additional parameter ``u`` is used as well as a smoothing technique,
for example [`LogarithmicSumOfExponentials`](@ref) or [`LinearQuadraticHuber`](@ref)
to obtain a smooth cost function. This struct is also a functor `(M,p) -> v` of the cost ``v``.

## Fields

* `ρ`, `u`: as described in the mathematical formula, .
* `co`:     the original cost

## Constructor

    ExactPenaltyCost(co::ConstrainedManifoldObjective, ρ, u; smoothing=LinearQuadraticHuber())
"""
mutable struct ExactPenaltyCost{S, CO, R}
    co::CO
    ρ::R
    u::R
end
function ExactPenaltyCost(
        co::ConstrainedManifoldObjective, ρ::R, u::R; smoothing = LinearQuadraticHuber()
    ) where {R}
    return ExactPenaltyCost{typeof(smoothing), typeof(co), R}(co, ρ, u)
end
function set_parameter!(epc::ExactPenaltyCost, ::Val{:ρ}, ρ)
    epc.ρ = ρ
    return epc
end
function set_parameter!(epc::ExactPenaltyCost, ::Val{:u}, u)
    epc.u = u
    return epc
end
function (L::ExactPenaltyCost{<:LogarithmicSumOfExponentials})(M::AbstractManifold, p)
    gp = get_inequality_constraint(M, L.co, p, :)
    hp = get_equality_constraint(M, L.co, p, :)
    m = length(gp)
    n = length(hp)
    cost_ineq = (m > 0) ? sum(L.u .* log.(1 .+ exp.(gp ./ L.u))) : 0.0
    cost_eq = (n > 0) ? sum(L.u .* log.(exp.(hp ./ L.u) .+ exp.(-hp ./ L.u))) : 0.0
    return get_cost(M, L.co, p) + (L.ρ) * (cost_ineq + cost_eq)
end
function (L::ExactPenaltyCost{<:LinearQuadraticHuber})(M::AbstractManifold, p)
    gp = get_inequality_constraint(M, L.co, p, :)
    hp = get_equality_constraint(M, L.co, p, :)
    m = length(gp)
    n = length(hp)
    cost_eq_greater_u = (m > 0) ? sum((gp .- L.u / 2) .* (gp .> L.u)) : 0.0
    cost_eq_pos_smaller_u = (m > 0) ? sum((gp .^ 2 ./ (2 * L.u)) .* (0 .< gp .<= L.u)) : 0.0
    cost_ineq = cost_eq_greater_u + cost_eq_pos_smaller_u
    cost_eq = (n > 0) ? sum(sqrt.(hp .^ 2 .+ L.u^2)) : 0.0
    return get_cost(M, L.co, p) + (L.ρ) * (cost_ineq + cost_eq)
end

@doc """
    ExactPenaltyGrad{S, CO, R}

Represent the gradient of the [`ExactPenaltyCost`](@ref) based on a [`ConstrainedManifoldObjective`](@ref) `co`
and a parameter ``ρ`` and a smoothing technique, which uses an additional parameter ``u``.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

## Fields

* `ρ`, `u` as stated before
* `co` the nonsmooth objective

## Constructor

    ExactPenaltyGradient(co::ConstrainedManifoldObjective, ρ, u; smoothing=LinearQuadraticHuber())
"""
mutable struct ExactPenaltyGrad{S, CO, R}
    co::CO
    ρ::R
    u::R
end
function set_parameter!(epg::ExactPenaltyGrad, ::Val{:ρ}, ρ)
    epg.ρ = ρ
    return epg
end
function set_parameter!(epg::ExactPenaltyGrad, ::Val{:u}, u)
    epg.u = u
    return epg
end
function ExactPenaltyGrad(
        co::ConstrainedManifoldObjective, ρ::R, u::R; smoothing = LinearQuadraticHuber()
    ) where {R}
    return ExactPenaltyGrad{typeof(smoothing), typeof(co), R}(co, ρ, u)
end
# Default (functions constraints): evaluate all gradients
# Since for LogExp the pre-factor c seems to not be zero, this might be the best way to go here
function (EG::ExactPenaltyGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return EG(M, X, p)
end
function (EG::ExactPenaltyGrad{<:LogarithmicSumOfExponentials})(M::AbstractManifold, X, p)
    gp = get_inequality_constraint(M, EG.co, p, :)
    hp = get_equality_constraint(M, EG.co, p, :)
    m = length(gp)
    n = length(hp)
    # start with `gradf`
    get_gradient!(M, X, EG.co, p)
    c = 0
    # add gradient of the components of g
    (m > 0) && (c = EG.ρ .* exp.(gp ./ EG.u) ./ (1 .+ exp.(gp ./ EG.u)))
    (m > 0) && (X .+= sum(get_grad_inequality_constraint(M, EG.co, p, :) .* c))
    # add gradient of the components of h
    (n > 0) && (
        c =
            EG.ρ .* (exp.(hp ./ EG.u) .- exp.(-hp ./ EG.u)) ./
            (exp.(hp ./ EG.u) .+ exp.(-hp ./ EG.u))
    )
    (n > 0) && (X .+= sum(get_grad_equality_constraint(M, EG.co, p, :) .* c))
    return X
end

# Default (functions constraints): evaluate all gradients
function (EG::ExactPenaltyGrad{<:LinearQuadraticHuber})(
        M::AbstractManifold, X, p::P
    ) where {P}
    gp = get_inequality_constraint(M, EG.co, p, :)
    hp = get_equality_constraint(M, EG.co, p, :)
    m = length(gp)
    n = length(hp)
    get_gradient!(M, X, EG.co, p)
    if m > 0
        gradgp = get_grad_inequality_constraint(M, EG.co, p, :)
        X .+= sum(gradgp .* (gp .>= EG.u) .* EG.ρ) # add the ones >= u
        X .+= sum(gradgp .* (gp ./ EG.u .* (0 .<= gp .< EG.u)) .* EG.ρ) # add < u
    end
    if n > 0
        c = (hp ./ sqrt.(hp .^ 2 .+ EG.u^2)) .* EG.ρ
        X .+= sum(get_grad_equality_constraint(M, EG.co, p, :) .* c)
    end
    return X
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
where ``p ∈ $(_math(:Manifold))``, ``μ, s ∈ ℝ^m`` and ``λ ∈ ℝ^n``,
and ``⊙`` denotes the Hadamard (or elementwise) product

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

While the point `p` is arbitrary and usually not needed, it serves as internal memory
in the computations. Furthermore Both fields together also clarify the product manifold structure to use.

# Constructor

    KKTVectorField(cmo::ConstrainedManifoldObjective)

# Example

Define `F = KKTVectorField(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``$(_math(:Manifold))×ℝ^m×ℝ^n×ℝ^m``.
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
function Base.show(io::IO, KKTvf::KKTVectorField)
    print(io, "KKTVectorField(")
    print(io, KKTvf.cmo)
    return print(io, ")")
end
function status_summary(KKTvf::KKTVectorField; context::Symbol = :default)
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
and let `N` be the product manifold of ``$(_math(:Manifold))×ℝ^m×ℝ^n×ℝ^m``.
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
function Base.show(io::IO, KKTvfJ::KKTVectorFieldJacobian)
    print(io, "KKTVectorFieldJacobian(")
    print(io, KKTvfJ.cmo)
    return print(io, ")")
end
function status_summary(KKTvfJ::KKTVectorFieldJacobian; context::Symbol = :default)
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
and let `N` be the product manifold of ``$(_math(:Manifold))×ℝ^m×ℝ^n×ℝ^m``.
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
function Base.show(io::IO, KKTvfAdJ::KKTVectorFieldAdjointJacobian)
    print(io, "KKTVectorFieldAdjointJacobian(")
    print(io, KKTvfAdJ.cmo)
    return print(io, ")")
end
function status_summary(KKTvfAdJ::KKTVectorFieldAdjointJacobian; context::Symbol = :default)
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
and let `N` be the product manifold of ``$(_math(:Manifold))×ℝ^m×ℝ^n×ℝ^m``.
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
function Base.show(io::IO, KKTvfNSq::KKTVectorFieldNormSq)
    print(io, "KKTVectorFieldNormSq(")
    print(io, KKTvfNSq.cmo)
    return print(io, ")")
end
function status_summary(KKTvfNSq::KKTVectorFieldNormSq; context::Symbol = :default)
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

For completeness, the gradient reads, using the [`LagrangianGradient`](@ref) ``L = $(_tex(:grad))_p $(_tex(:Cal, "L"))(p,μ,λ) ∈ T_p$(_math(:Manifold))``,
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
where ``⊙`` denotes the Hadamard (or element wise) product.

# Fields

* `cmo` the [`ConstrainedManifoldObjective`](@ref)

# Constructor

    KKTVectorFieldNormSqGradient(cmo::ConstrainedManifoldObjective)

# Example

Define `grad_f = KKTVectorFieldNormSqGradient(cmo)` for some [`ConstrainedManifoldObjective`](@ref) `cmo`
and let `N` be the product manifold of ``$(_math(:Manifold))×ℝ^m×ℝ^n×ℝ^m``.
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
function Base.show(io::IO, KKTvfNSqGrad::KKTVectorFieldNormSqGradient)
    print(io, "KKTVectorFieldNormSqGradient(")
    print(io, KKTvfNSqGrad.cmo)
    return print(io, ")")
end
function status_summary(KKTvfNSqGrad::KKTVectorFieldNormSqGradient; context::Symbol = :default)
    _is_inline(context) && (return repr(KKTvfNSqGrad))
    return "The gradient of the KKT vector field in normed squared for the constrained objective\n$(_MANOPT_INDENT)$(status_summary(KKTvfNSqGrad.cmo; context = context))"
end

@doc """
    LagrangianCost{CO,T} <: AbstractConstrainedFunction{T}

Implement the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`.

```math
$(_tex(:Cal, "L"))(p; μ, λ) = f(p) + $(_tex(:sum, "i=1", "m")) μ_ig_i(p) + $(_tex(:sum, "j=1", "n")) λ_jh_j(p)
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianCost(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the Lagrangian ``$(_tex(:Cal, "L"))``
you can also call

```
LagrangianCost(co, μ, λ)(M,p)
```
"""
mutable struct LagrangianCost{CO, T} <: AbstractConstrainedFunction{T}
    co::CO
    μ::T
    λ::T
end
function (lc::LagrangianCost)(M, p)
    c = get_cost(M, lc.co, p)
    g = get_inequality_constraint(M, lc.co, p, :)
    h = get_equality_constraint(M, lc.co, p, :)
    (length(g) > 0) && (c += sum(lc.μ .* g))
    (length(h) > 0) && (c += sum(lc.λ .* h))
    return c
end
function show(io::IO, lc::LagrangianCost)
    return print(io, "LagrangianCost\n$(_MANOPT_INDENT)with μ=$(lc.μ), λ=$(lc.λ)")
end

@doc """
    LagrangianGradient{CO,T}

The gradient of the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`
with respect to the variable ``p``. The formula reads

```math
$(_tex(:grad))_p $(_tex(:Cal, "L"))(p; μ, λ)
= $(_tex(:grad)) f(p) + $(_tex(:sum, "i=1", "m")) μ_i $(_tex(:grad)) g_i(p) + $(_tex(:sum, "j=1", "n")) λ_j $(_tex(:grad)) h_j(p)
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianGradient(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the gradient of the Lagrangian ``$(_tex(:grad))_p $(_tex(:Cal, "L"))``
you can also call `LagrangianGradient(co, μ, λ)(M,p)` or `LagrangianGradient(co, μ, λ)(M,X,p)` for the in-place variant.
"""
mutable struct LagrangianGradient{CO, T} <: AbstractConstrainedFunction{T}
    co::CO
    μ::T
    λ::T
end
function (lg::LagrangianGradient)(M, p)
    X = zero_vector(M, p)
    return lg(M, X, p)
end
function (lg::LagrangianGradient)(M, X, p)
    Y = copy(M, p, X)
    get_gradient!(M, X, lg.co, p)
    m = inequality_constraints_length(lg.co)
    n = equality_constraints_length(lg.co)
    for i in 1:m
        get_grad_inequality_constraint!(M, Y, lg.co, p, i)
        copyto!(M, X, p, X + lg.μ[i] * Y)
    end
    for j in 1:n
        get_grad_equality_constraint!(M, Y, lg.co, p, j)
        copyto!(M, X, p, X + lg.λ[j] * Y)
    end
    return X
end
function show(io::IO, lg::LagrangianGradient)
    return print(io, "LagrangianGradient\n$(_MANOPT_INDENT)with μ=$(lg.μ), λ=$(lg.λ)")
end

@doc """
    LagrangianHessian{CO, V, T}

The Hessian of the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`
with respect to the variable ``p``. The formula reads

```math
$(_tex(:Hess))_p $(_tex(:Cal, "L"))(p; μ, λ)[X]
= $(_tex(:Hess)) f(p) + $(_tex(:sum, "i=1", "m")) μ_i $(_tex(:Hess)) g_i(p)[X] + $(_tex(:sum, "j=1", "n")) λ_j $(_tex(:Hess)) h_j(p)[X]
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianHessian(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the Hessian of the Lagrangian ``$(_tex(:Hess))_p $(_tex(:Cal, "L"))``
you can also call `LagrangianHessian(co, μ, λ)(M, p, X)` or `LagrangianHessian(co, μ, λ)(M, Y, p, X)` for the in-place variant.
"""
mutable struct LagrangianHessian{CO, T} <: AbstractConstrainedFunction{T}
    co::CO
    μ::T
    λ::T
end
function (lH::LagrangianHessian)(M, p, X)
    Y = zero_vector(M, p)
    return lH(M, Y, p, X)
end
function (lH::LagrangianHessian)(M, Y, p, X)
    Z = copy(M, p, X)
    get_hessian!(M, Y, lH.co, p, X)
    n = inequality_constraints_length(lH.co)
    m = equality_constraints_length(lH.co)
    for i in 1:n
        get_hess_inequality_constraint!(M, Z, lH.co, p, X, i)
        copyto!(M, Y, p, Y + lH.μ[i] * Z)
    end
    for j in 1:m
        get_hess_equality_constraint!(M, Z, lH.co, p, X, j)
        copyto!(M, Y, p, Y + lH.λ[j] * Z)
    end
    return Y
end
function show(io::IO, lh::LagrangianHessian)
    return print(io, "LagrangianHessian\n$(_MANOPT_INDENT)with μ=$(lh.μ), λ=$(lh.λ)")
end

#
#
# ---
@doc """
    LinearizedDCCost

A functor `(M,q) → ℝ` to represent the inner problem of a [`ManifoldDifferenceOfConvexObjective`](@ref).
This is a cost function of the form

```math
    F_{p_k,X_k}(p) = g(p) - ⟨X_k, $(_tex(:log))_{p_k}p⟩
```
for a point `p_k` and a tangent vector `X_k` at `p_k` (for example outer iterates)
that are stored within this functor as well.

# Fields

* `g` a function
* `pk` a point on a manifold
* `Xk` a tangent vector at `pk`

Both interim values can be set using
`set_parameter!(::LinearizedDCCost, ::Val{:p}, p)`
and `set_parameter!(::LinearizedDCCost, ::Val{:X}, X)`, respectively.

# Constructor
    LinearizedDCCost(g, p, X)
"""
mutable struct LinearizedDCCost{P, T, TG}
    g::TG
    pk::P
    Xk::T
end
(F::LinearizedDCCost)(M, p) = F.g(M, p) - inner(M, F.pk, F.Xk, log(M, F.pk, p))

function set_parameter!(ldc::LinearizedDCCost, ::Val{:p}, p)
    ldc.pk .= p
    return ldc
end
function set_parameter!(ldc::LinearizedDCCost, ::Val{:X}, X)
    ldc.Xk .= X
    return ldc
end

@doc """
    LinearizedDCGrad

A functor `(M,X,p) → ℝ` to represent the gradient of the inner problem of a [`ManifoldDifferenceOfConvexObjective`](@ref).
This is a gradient function of the form

```math
    F_{p_k,X_k}(p) = g(p) - ⟨X_k, $(_tex(:log))_{p_k}p⟩
```

its gradient is given by using ``F=F_1(F_2(p))``, where ``F_1(X) = ⟨X_k,X⟩`` and ``F_2(p) = $(_tex(:log))_{p_k}p``
and the chain rule as well as the adjoint differential of the logarithmic map with respect to its argument for ``D^*F_2(p)``

```math
    $(_tex(:grad)) F(q) = $(_tex(:grad))f(q) - DF_2^*(q)[X]
```

for a point `pk` and a tangent vector `Xk` at `pk` (the outer iterates) that are stored within this functor as well

# Fields

* `grad_g!` the gradient of ``g`` (see also [`LinearizedDCCost`](@ref))
* `pk` a point on a manifold
* `Xk` a tangent vector at `pk`

Both interim values can be set using
`set_parameter!(::LinearizedDCGrad, ::Val{:p}, p)`
and `set_parameter!(::LinearizedDCGrad, ::Val{:X}, X)`, respectively.

# Constructor
    LinearizedDCGrad(grad_g, p, X; evaluation=AllocatingEvaluation())

Where you specify whether `grad_g` is [`AllocatingEvaluation`](@ref) or [`InplaceEvaluation`](@ref),
while this function still provides _both_ signatures.
"""
mutable struct LinearizedDCGrad{P, T, TG}
    grad_g!::TG
    pk::P
    Xk::T
    function LinearizedDCGrad(grad_g::TG, pk::P, Xk::T) where {TG, P, T}
        # TODO: readd evaluatiion and wrap grad
        return new{P, T, TG}(grad_g, pk, Xk)
    end
end
function (grad_f!::LinearizedDCGrad)(M, X, p)
    grad_f!.grad_g!(M, X, p)
    X .-= adjoint_differential_log_argument(M, grad_f!.pk, p, grad_f!.Xk)
    return X
end
function (grad_f!::LinearizedDCGrad)(M, p)
    X = zero_vector(M, p)
    grad_f!.grad_g!(M, X, p)
    X .-= adjoint_differential_log_argument(M, grad_f!.pk, p, grad_f!.Xk)
    return X
end
function set_parameter!(ldcg::LinearizedDCGrad, ::Val{:p}, p)
    ldcg.pk .= p
    return ldcg
end
function set_parameter!(ldcg::LinearizedDCGrad, ::Val{:X}, X)
    ldcg.Xk .= X
    return ldcg
end

@doc """
    ProximalDCCost

A functor `(M, p) → ℝ` to represent the inner cost function of a [`ManifoldDifferenceOfConvexProximalObjective`](@ref).
This is the cost function of the proximal map of `g`.

```math
F_{p_k}(p) = $(_tex(:frac, "1", "2λ"))d_{$(_math(:Manifold))}(p_k,p)^2 + g(p)
```

for a point `pk` and a proximal parameter ``λ``.

# Fields

* `g`  - a function
* `pk` - a point on a manifold
* `λ`  - the prox parameter

Both interim values can be set using
`set_parameter!(::ProximalDCCost, ::Val{:p}, p)`
and `set_parameter!(::ProximalDCCost, ::Val{:λ}, λ)`, respectively.

# Constructor

    ProximalDCCost(g, p, λ)
"""
mutable struct ProximalDCCost{P, TG, R}
    g::TG
    pk::P
    λ::R
end
(F::ProximalDCCost)(M, p) = 1 / (2 * F.λ) * distance(M, p, F.pk)^2 + F.g(M, p)

function set_parameter!(pdcc::ProximalDCCost, ::Val{:p}, p)
    pdcc.pk .= p
    return pdcc
end
function set_parameter!(pdcc::ProximalDCCost, ::Val{:λ}, λ)
    pdcc.λ = λ
    return pdcc
end

@doc """
    ProximalDCGrad

A functor `(M,X,p) → ℝ` to represent the gradient of the inner cost function of a [`ManifoldDifferenceOfConvexProximalObjective`](@ref).
This is the gradient function of the proximal map cost function of `g`. Based on

```math
F_{p_k}(p) = $(_tex(:frac, "1", "2λ"))d_{$(_math(:Manifold))}(p_k,p)^2 + g(p)
```

it reads

```math
$(_tex(:grad)) F_{p_k}(p) = $(_tex(:grad))} g(p) - $(_tex(:frac, "1", "λ"))$(_tex(:log))_p p_k
```

for a point `pk` and a proximal parameter `λ`.

# Fields

* `grad_g`  - a gradient function
* `pk` - a point on a manifold
* `λ`  - the prox parameter

Both interim values can be set using
`set_parameter!(::ProximalDCGrad, ::Val{:p}, p)`
and `set_parameter!(::ProximalDCGrad, ::Val{:λ}, λ)`, respectively.

# Constructor

    ProximalDCGrad(grad_g, pk, λ; evaluation=AllocatingEvaluation())

Where you specify whether `grad_g` is [`AllocatingEvaluation`](@ref) or [`InplaceEvaluation`](@ref),
while this function still always provides _both_ signatures.
"""
mutable struct ProximalDCGrad{P, TG, R}
    grad_g!::TG
    pk::P
    λ::R
    function ProximalDCGrad(grad_g::TG, pk::P, λ::R) where {TG, P, R}
        #TODO: Add eval and a wrapper?
        return new{P, TG, R}(grad_g, pk, λ)
    end
end
function (grad_f!::ProximalDCGrad)(M, X, p)
    grad_f!.grad_g!(M, X, p)
    X .-= 1 / grad_f!.λ * log(M, p, grad_f!.pk)
    return X
end
function (grad_f!::ProximalDCGrad)(M, p)
    X = zero_vector(M, p)
    grad_f!.grad_g!(M, X, p)
    X .-= 1 / grad_f!.λ * log(M, p, grad_f!.pk)
    return X
end
function set_parameter!(pdcg::ProximalDCGrad, ::Val{:p}, p)
    pdcg.pk .= p
    return pdcg
end
function set_parameter!(pdcg::ProximalDCGrad, ::Val{:λ}, λ)
    pdcg.λ = λ
    return pdcg
end
