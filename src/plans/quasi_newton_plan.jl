@doc """
    AbstractQuasiNewtonDirectionUpdate

An abstract representation of an Quasi Newton Update rule to determine the next direction
given current [`QuasiNewtonState`](@ref).

All subtypes should be functors, they should be callable as `H(M,x,d)` to compute a new direction update.
"""
abstract type AbstractQuasiNewtonDirectionUpdate end

get_message(::AbstractQuasiNewtonDirectionUpdate) = ""

"""
    initialize_update!(s::AbstractQuasiNewtonDirectionUpdate)

Initialize direction update. By default no change is made.
"""
initialize_update!(s::AbstractQuasiNewtonDirectionUpdate) = s

@doc """
    AbstractQuasiNewtonUpdateRule

Specify a type for the different [`AbstractQuasiNewtonDirectionUpdate`](@ref)s,
that is for a [`QuasiNewtonMatrixDirectionUpdate`](@ref) there are several different updates to the matrix,
while the default for [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) the most prominent is [`InverseBFGS`](@ref).
"""
abstract type AbstractQuasiNewtonUpdateRule end

@doc """
    BFGS <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemannian BFGS update is used in the Riemannian quasi-Newton method.

Denote by ``$(_tex(:widetilde, "H"))_k^$(_tex(:rm, "BFGS"))`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{p_k}(α_k η_k)``.
Then the update formula reads

```math
H^$(_tex(:rm, "BFGS"))_{k+1} = $(_tex(:widetilde, "H"))^$(_tex(:rm, "BFGS"))_k  + $(_tex(:frac, "y_k y^{$(_tex(:rm, "T"))}_k ", "s^{$(_tex(:rm, "T"))}_k y_k")) - $(_tex(:frac, "$(_tex(:widetilde, "H"))^$(_tex(:rm, "BFGS"))_k s_k s^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "H"))^$(_tex(:rm, "BFGS"))_k ", " s^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "H"))^$(_tex(:rm, "BFGS"))_k s_k"))
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:M)),
```

respectively.
"""
struct BFGS <: AbstractQuasiNewtonUpdateRule end

@doc """
    InverseBFGS <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemannian BFGS update is used in the Riemannian quasi-Newton method.

Denote by ``$(_tex(:widetilde, "B"))_k^$(_tex(:rm, "BFGS"))`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{p_k}(α_k η_k)``.
Then the update formula reads

```math
B^$(_tex(:rm, "BFGS"))_{k+1}  = $(_tex(:Bigl))(
  $(_tex(:Id))_{T_{x_{k+1}} $(_math(:M))} - $(_tex(:frac, "s_k y^{$(_tex(:rm, "T"))}_k ", "s^{$(_tex(:rm, "T"))}_k y_k"))
$(_tex(:Bigr)))
$(_tex(:widetilde, "B"))^$(_tex(:rm, "BFGS"))_k
$(_tex(:Bigl))(
  $(_tex(:Id))_{T_{x_{k+1}} $(_math(:M))} - $(_tex(:frac, "y_k s^{$(_tex(:rm, "T"))}_k ", "s^{$(_tex(:rm, "T"))}_k y_k"))
$(_tex(:Bigr))) + $(_tex(:frac, "s_k s^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:M)),
```

respectively.
"""
struct InverseBFGS <: AbstractQuasiNewtonUpdateRule end

@doc """
    DFP <: AbstractQuasiNewtonUpdateRule

indicates in an [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemannian DFP update is used in the Riemannian quasi-Newton method.

Denote by ``$(_tex(:widetilde, "H"))_k^$(_tex(:rm, "DFP"))`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{p_k}(α_k η_k)``.
Then the update formula reads

```math
H^$(_tex(:rm, "DFP"))_{k+1} = $(_tex(:Bigl))(
  $(_tex(:Id))_{T_{x_{k+1}} $(_math(:M))} - $(_tex(:frac, "y_k s^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
$(_tex(:Bigr)))
$(_tex(:widetilde, "H"))^$(_tex(:rm, "DFP"))_k
$(_tex(:Bigl))(
  $(_tex(:Id))_{T_{x_{k+1}} $(_math(:M))} - $(_tex(:frac, "s_k y^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
$(_tex(:Bigr))) + $(_tex(:frac, "y_k y^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:M)),
```

respectively.
"""
struct DFP <: AbstractQuasiNewtonUpdateRule end

@doc """
    InverseDFP <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemannian DFP update is used in the Riemannian quasi-Newton method.

Denote by ``$(_tex(:widetilde, "B"))_k^$(_tex(:rm, "DFP"))`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{p_k}(α_k η_k)``.
Then the update formula reads

```math
B^$(_tex(:rm, "DFP"))_{k+1} = $(_tex(:widetilde, "B"))^$(_tex(:rm, "DFP"))_k + $(_tex(:frac, "s_k s^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
  - $(_tex(:frac, "$(_tex(:widetilde, "B"))^$(_tex(:rm, "DFP"))_k y_k y^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "B"))^$(_tex(:rm, "DFP"))_k", "y^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "B"))^$(_tex(:rm, "DFP"))_k y_k"))
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:M)),
```

respectively.
"""
struct InverseDFP <: AbstractQuasiNewtonUpdateRule end

@doc """
    SR1 <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemannian SR1 update is used in the Riemannian quasi-Newton method.

Denote by ``$(_tex(:widetilde, "H"))_k^{$(_tex(:rm, "SR1"))}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{p_k}(α_k η_k)``.
Then the update formula reads

```math
H^{$(_tex(:rm, "SR1"))}_{k+1} = $(_tex(:widetilde, "H"))^{$(_tex(:rm, "SR1"))}_k
+ $(
    _tex(
        :frac,
        "(y_k - $(_tex(:widetilde, "H"))^{$(_tex(:rm, "SR1"))}_k s_k) (y_k - $(_tex(:widetilde, "H"))^{$(_tex(:rm, "SR1"))}_k s_k)^{$(_tex(:rm, "T"))}",
        "(y_k - $(_tex(:widetilde, "H"))^{$(_tex(:rm, "SR1"))}_k s_k)^{$(_tex(:rm, "T"))} s_k",
    )
)
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:M)),
```

respectively.

This method can be stabilized by only performing the update if denominator is larger than
``r$(_tex(:norm, "s_k"; index = "x_{k+1}"))$(_tex(:norm, "y_k - $(_tex(:widetilde, "H"))^{$(_tex(:rm, "SR1"))}_k s_k"; index = "x_{k+1}"))``
for some ``r>0``. For more details, see Section 6.2 in [NocedalWright:2006](@cite).

# Constructor
    SR1(r::Float64=-1.0)

Generate the `SR1` update.
"""
struct SR1 <: AbstractQuasiNewtonUpdateRule
    r::Float64
    SR1(r::Float64 = -1.0) = new(r)
end

@doc """
    InverseSR1 <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemannian SR1 update is used in the Riemannian quasi-Newton method.

Denote by ``$(_tex(:widetilde, "B"))_k^{$(_tex(:rm, "SR1"))}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{p_k}(α_k η_k)``.
Then the update formula reads

```math
B^{$(_tex(:rm, "SR1"))}_{k+1} = $(_tex(:widetilde, "B"))^{$(_tex(:rm, "SR1"))}_k
+ $(
    _tex(
        :frac,
        "(s_k - $(_tex(:widetilde, "B"))^{$(_tex(:rm, "SR1"))}_k y_k) (s_k - $(_tex(:widetilde, "B"))^{$(_tex(:rm, "SR1"))}_k y_k)^{$(_tex(:rm, "T"))}",
        "(s_k - $(_tex(:widetilde, "B"))^{$(_tex(:rm, "SR1"))}_k y_k)^{$(_tex(:rm, "T"))} y_k"
    )
)
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:M)),
```

respectively.

This method can be stabilized by only performing the update if denominator is larger than
``r$(_tex(:norm, "y_k"; index = "x_{k+1}"))$(_tex(:norm, "s_k - $(_tex(:widetilde, "H"))^{$(_tex(:rm, "SR1"))}_k y_k"; index = "x_{k+1}"))``
for some ``r>0``. For more details, see Section 6.2 in [NocedalWright:2006](@cite).

# Constructor
    InverseSR1(r::Float64=-1.0)

Generate the `InverseSR1`.
"""
struct InverseSR1 <: AbstractQuasiNewtonUpdateRule
    r::Float64
    InverseSR1(r::Float64 = -1.0) = new(r)
end

@doc """
    Broyden <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemannian Broyden update is used in the Riemannian quasi-Newton method, which is as a convex combination of [`BFGS`](@ref) and [`DFP`](@ref).

Denote by ``$(_tex(:widetilde, "H"))_k^$(_tex(:rm, "Br"))`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{p_k}(α_k η_k)``.
Then the update formula reads

```math
H^{$(_tex(:rm, "Br"))}_{k+1}
=   $(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k - $(
    _tex(
        :frac,
        "$(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k s_k s^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k",
        "s^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k s_k"
    )
) + $(_tex(:frac, "y_k y^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
    + φ_k s^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k s_k
    $(_tex(:Bigl))(
        $(_tex(:frac, "y_k", "s^{$(_tex(:rm, "T"))}_k y_k")) - $(_tex(:frac, "$(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k s_k", "s^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k s_k"))
  $(_tex(:Bigr)))
  $(_tex(:Bigl))(
        $(_tex(:frac, "y_k", "s^{$(_tex(:rm, "T"))}_k y_k")) - $(_tex(:frac, "$(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k s_k", "s^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "H"))^{$(_tex(:rm, "Br"))}_k s_k"))
  $(_tex(:Bigr)))^{$(_tex(:rm, "T"))}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:M)),
```

respectively, and ``φ_k`` is the Broyden factor which is `:constant` by default but can also be set to `:Davidon`.

# Constructor
    Broyden(φ, update_rule::Symbol = :constant)
"""
mutable struct Broyden <: AbstractQuasiNewtonUpdateRule
    φ::Float64
    update_rule::Symbol
end
Broyden(φ::Float64) = Broyden(φ, :constant)

@doc """
    InverseBroyden <: AbstractQuasiNewtonUpdateRule

Indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemannian Broyden update
is used in the Riemannian quasi-Newton method, which is as a convex combination
of [`InverseBFGS`](@ref) and [`InverseDFP`](@ref).

Denote by ``$(_tex(:widetilde, "H"))_k^{$(_tex(:rm, "Br"))}`` the operator concatenated with a vector transport
and its inverse before and after to act on ``x_{k+1} = R_{p_k}(α_k η_k)``.
Then the update formula reads

```math
B^{$(_tex(:rm, "Br"))}_{k+1}
= $(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k
   - $(
    _tex(
        :frac,
        "$(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k y_k y^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k",
        "y^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k y_k"
    )
)
    + $(_tex(:frac, "s_k s^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
    + φ_k y^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k y_k
    $(_tex(:Bigl))(
        $(_tex(:frac, "s_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
        - $(_tex(:frac, "$(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k y_k", "y^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k y_k"))
    $(_tex(:Bigr)))
    $(_tex(:Bigl))(
        $(_tex(:frac, "s_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
        - $(_tex(:frac, "$(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k y_k", "y^{$(_tex(:rm, "T"))}_k $(_tex(:widetilde, "B"))^{$(_tex(:rm, "Br"))}_k y_k"))
     $(_tex(:Bigr)))^{$(_tex(:rm, "T"))}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:M)),
```

respectively, and ``φ_k`` is the Broyden factor which is `:constant` by default but can also be set to `:Davidon`.

# Constructor
    InverseBroyden(φ, update_rule::Symbol = :constant)
"""
mutable struct InverseBroyden <: AbstractQuasiNewtonUpdateRule
    φ::Float64
    update_rule::Symbol
end
InverseBroyden(φ::Float64) = InverseBroyden(φ, :constant)

_doc_QN_H_update = "``H_k ↦ H_{k+1}``"
_doc_QN_B_update = "``B_k ↦ B_{k+1}``"
_doc_QN_H_full_system = """
```math
$(_tex(:text, "Solve"))$(_tex(:quad))$(_tex(:hat, "η_k")) = - H_k $(_tex(:widehat, "$(_tex(:grad))f(p_k)")),
```
"""
_doc_QN_B_full_system = """
```math
$(_tex(:hat, "η_k")) = - B_k $(_tex(:widehat, "$(_tex(:grad))f(p_k)")),
```
"""

"""
    QuasiNewtonPreconditioner{E<:AbstractEvaluationType, F}

Add a preconditioning

# Fields

* `preconditioner::F`: the preconditioner function

# Constructors

    QuasiNewtonPreconditioner(
        preconditioner;
        evaluation::AbstractEvaluationType=AllocatingEvaluation()
    )

Add preconditioning to a gradient problem.

# Input

* `preconditioner`:   preconditioner function, either as a `(M, p, X) -> Y` allocating or `(M, Y, p, X) -> Y` mutating function

# Keyword arguments

$(_var(:Keyword, :evaluation))
"""
struct QuasiNewtonPreconditioner{E <: AbstractEvaluationType, F}
    preconditioner::F
end
function QuasiNewtonPreconditioner(
        preconditioner::F; evaluation::E = AllocatingEvaluation()
    ) where {E <: AbstractEvaluationType, F}
    return QuasiNewtonPreconditioner{E, F}(preconditioner)
end
#
#
# Internally this always works in-place of X
function (qnp::QuasiNewtonPreconditioner{AllocatingEvaluation})(
        X, mp::AbstractManoptProblem, s::AbstractGradientSolverState
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    copyto!(M, X, p, qnp.preconditioner(M, p, X))
    return X
end
function (pg::QuasiNewtonPreconditioner{InplaceEvaluation})(
        X, mp::AbstractManoptProblem, s::AbstractGradientSolverState
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    pg.preconditioner(M, X, p, X)
    return X
end

@doc """
    QuasiNewtonMatrixDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

The `QuasiNewtonMatrixDirectionUpdate` represent a quasi-Newton update rule,
where the operator is stored as a matrix. A distinction is made between the update of the
approximation of the Hessian, $_doc_QN_H_update, and the update of the approximation
of the Hessian inverse, $_doc_QN_B_update.
For the first case, the coordinates of the search direction ``η_k`` with respect to
a basis ``$(_math(:Sequence, "b", "i", "1", "n"))`` are determined by solving a linear system of equations

$_doc_QN_H_full_system

where ``H_k`` is the matrix representing the operator with respect to the basis ``$(_math(:Sequence, "b", "i", "1", "n"))``
and ``$(_tex(:widehat, "$(_tex(:grad)) f(p_k)"))`` represents the coordinates of the gradient of
the objective function ``f`` in ``p_k`` with respect to the basis ``$(_math(:Sequence, "b", "i", "1", "n"))``.
If a method is chosen where Hessian inverse is approximated, the coordinates of the search
direction ``η_k`` with respect to a basis ``$(_math(:Sequence, "b", "i", "1", "n"))`` are obtained simply by
matrix-vector multiplication

$_doc_QN_B_full_system

where ``B_k`` is the matrix representing the operator with respect to the basis ``$(_math(:Sequence, "b", "i", "1", "n"))``
and ``$(_tex(:widehat, "$(_tex(:grad)) f(p_k)"))``. In the end, the search direction ``η_k`` is
generated from the coordinates ``$(_tex(:hat, "η"))_k`` and the vectors of the basis ``$(_math(:Sequence, "b", "i", "1", "n"))``
in both variants.
The [`AbstractQuasiNewtonUpdateRule`](@ref) indicates which quasi-Newton update rule is used.
In all of them, the Euclidean update formula is used to generate the matrix ``H_{k+1}``
and ``B_{k+1}``, and the basis ``$(_math(:Sequence, "b", "i", "1", "n"))`` is transported into the upcoming tangent
space ``T_{p_{k+1}} $(_tex(:Cal, "M"))``, preferably with an isometric vector transport, or generated there.

# Provided functors

* `(mp::AbstractManoptProblem, st::QuasiNewtonState) -> η` to compute the update direction
* `(η, mp::AbstractManoptProblem, st::QuasiNewtonState) -> η` to compute the update direction in-place of `η`

# Fields

* `basis`:                  an `AbstractBasis` to use in the tangent spaces
* `matrix`:                 the matrix which represents the approximating operator.
* `initial_scale`:          when initialising the update, a unit matrix is used as initial approximation, scaled by this factor
* `update`:                 a [`AbstractQuasiNewtonUpdateRule`](@ref).
$(_var(:Field, :vector_transport_method))

# Constructor

    QuasiNewtonMatrixDirectionUpdate(
        M::AbstractManifold,
        update,
        basis::B=default_basis(M),
        m=Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M));
        kwargs...
    )

## Keyword arguments

* `initial_scale=1.0` – this can also be deactivated by passing `nothing`.
$(_var(:Keyword, :vector_transport_method))

Generate the Update rule with defaults from a manifold and the names corresponding to the fields.

# See also

[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref),
[`QuasiNewtonCautiousDirectionUpdate`](@ref),
[`AbstractQuasiNewtonDirectionUpdate`](@ref),
"""
mutable struct QuasiNewtonMatrixDirectionUpdate{
        NT <: AbstractQuasiNewtonUpdateRule,
        B <: AbstractBasis,
        VT <: AbstractVectorTransportMethod,
        M <: AbstractMatrix,
        F <: Union{<:Real, Nothing},
    } <: AbstractQuasiNewtonDirectionUpdate
    basis::B
    matrix::M
    initial_scale::F
    update::NT
    vector_transport_method::VT
end
function status_summary(d::QuasiNewtonMatrixDirectionUpdate)
    return "$(d.update) with $(!isnothing(d.initial_scale) ? "initial scaling $(d.initial_scale) and" : "") vector transport method $(d.vector_transport_method)."
end
function show(io::IO, d::QuasiNewtonMatrixDirectionUpdate)
    s = """
    QuasiNewtonMatrixDirectionUpdate($(d.basis), $(d.matrix), $(d.initial_scale), $(d.update), $(d.vector_transport_method))
    """
    return print(io, s)
end
function QuasiNewtonMatrixDirectionUpdate(
        M::AbstractManifold,
        update::U,
        basis::B = default_basis(M),
        m::MT = Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M));
        initial_scale::F = 1.0,
        vector_transport_method::V = default_vector_transport_method(M),
    ) where {
        U <: AbstractQuasiNewtonUpdateRule,
        MT <: AbstractMatrix,
        B <: AbstractBasis,
        V <: AbstractVectorTransportMethod,
        F <: Union{<:Real, Nothing},
    }
    return QuasiNewtonMatrixDirectionUpdate{U, B, V, MT, F}(
        basis, m, initial_scale, update, vector_transport_method
    )
end
function (d::QuasiNewtonMatrixDirectionUpdate)(mp::AbstractManoptProblem, st)
    r = zero_vector(get_manifold(mp), get_iterate(st))
    return d(r, mp, st)
end
function (d::QuasiNewtonMatrixDirectionUpdate{T})(
        r, mp::AbstractManoptProblem, st
    ) where {T <: Union{InverseBFGS, InverseDFP, InverseSR1, InverseBroyden}}
    M = get_manifold(mp)
    p = get_iterate(st)
    X = get_gradient(st)
    copyto!(M, r, p, X)
    st.preconditioner(r, mp, st)
    get_vector!(M, r, p, -d.matrix * get_coordinates(M, p, r, d.basis), d.basis)
    return r
end
function (d::QuasiNewtonMatrixDirectionUpdate{T})(
        r, mp::AbstractManoptProblem, st
    ) where {T <: Union{BFGS, DFP, SR1, Broyden}}
    M = get_manifold(mp)
    p = get_iterate(st)
    X = get_gradient(st)
    get_vector!(M, r, p, -d.matrix \ get_coordinates(M, p, X, d.basis), d.basis)
    return r
end
function initialize_update!(d::QuasiNewtonMatrixDirectionUpdate)
    copyto!(d.matrix, I)
    return d
end
"""
    hess_val(d::QuasiNewtonMatrixDirectionUpdate, M, p, X)

Evaluate the quadratic form associated with the stored quasi-Newton matrix.
Returns the scalar ``c^{\top} B c`` where ``c`` are the coordinates of the
tangent vector `X` at `p` (in the basis `d.basis`) and ``B`` is `d.matrix`.
"""
function hess_val(d::QuasiNewtonMatrixDirectionUpdate{T}, M::AbstractManifold, p, X) where {T <: Union{BFGS, DFP, SR1, Broyden}}
    c = get_coordinates(M, p, X, d.basis)
    return dot(c, d.matrix, c)
end

"""
    hess_val_eb(d::QuasiNewtonMatrixDirectionUpdate, M, p, b)

Evaluate the quadratic form associated with the stored quasi-Newton matrix.
Returns the scalar ``c^{\top} B c`` where ``c`` are the coordinates of the
unit tangent vector along direction with index `b` at `p` (in the basis `d.basis`)
and ``B`` is `d.matrix`.
"""
function hess_val_eb(d::QuasiNewtonMatrixDirectionUpdate{T}, ::AbstractManifold, p, b) where {T <: Union{BFGS, DFP, SR1, Broyden}}
    return d.matrix[b, b]
end
"""
    hess_val_eb(d::QuasiNewtonMatrixDirectionUpdate, M, p, b, X)

Evaluate the quadratic form associated with the stored quasi-Newton matrix.
Returns the scalar ``c_b^{\top} B c`` where ``c_b`` are the coordinates of the
unit tangent vector along direction with index `b` at `p` (in the basis `d.basis`),
``c`` are the coordinates of the tangent vector `X` at `p` (in the basis `d.basis`)
and ``B`` is `d.matrix`.
"""
function hess_val_eb(d::QuasiNewtonMatrixDirectionUpdate{T}, M::AbstractManifold, p, b, X) where {T <: Union{BFGS, DFP, SR1, Broyden}}
    return dot(d.matrix[b, :], get_coordinates(M, p, X, d.basis))
end

_doc_QN_B = """
```math
$(_tex(:Cal, "B"))_k^{(0)}[⋅]
= $(_tex(:frac, "$(_tex(:inner, "s_{k-1}", "y_{k-1}"; index = "p_k"))", "$(_tex(:inner, "y_{k-1}", "y_{k-1}"; index = "p_k"))"))$(_tex(:Id))_{$(_math(:TpM))}[⋅]
```
"""

@doc """
    QuasiNewtonLimitedMemoryDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

This [`AbstractQuasiNewtonDirectionUpdate`](@ref) represents the limited-memory Riemannian BFGS update,
where the approximating operator is represented by ``m`` stored pairs of tangent
vectors ``$(_math(:Sequence, "$(_tex(:widetilde, "s"))", "i", "k-m", "k-1"))`` and ``$(_math(:Sequence, "$(_tex(:widetilde, "y"))", "i", "k-m", "k-1"))``
in the ``k``-th iteration. For the calculation of the search direction ``X_k``, the generalisation
of the two-loop recursion is used (see [HuangGallivanAbsil:2015](@cite)),
since it only requires inner products and linear combinations of tangent vectors in ``$(_math(:TpM; p = "p_k"))``.
For that the stored pairs of tangent vectors ``s_i, y_i``,
the gradient ``$(_tex(:grad)) f(p_k)`` of the objective function ``f`` in ``p_k``
and the positive definite self-adjoint operator

$(_doc_QN_B)

are used. The two-loop recursion can be understood as that the [`InverseBFGS`](@ref) update
is executed ``m`` times in a row on ``$(_tex(:Cal, "B"))^{(0)}_k[⋅]`` using the tangent vectors ``$(_tex(:widehat, "s"))_i,$(_tex(:widehat, "y"))_i``,
and in the same time the resulting operator ``$(_tex(:Cal, "B"))^{$(_tex(:rm, "LRBFGS"))}_k [⋅]`` is directly applied on ``$(_tex(:grad))f(p_k)``.
When updating there are two cases: if there is still free memory, ``k < m``, the previously
stored vector pairs ``$(_tex(:widehat, "s"))_i,$(_tex(:widehat, "y"))_i`` have to be
transported into the upcoming tangent space ``$(_math(:TpM; p = "p_{k+1}"))``.
If there is no free memory, the oldest pair ``$(_tex(:widehat, "s"))_i,$(_tex(:widehat, "y"))_i``
has to be discarded and then all the remaining vector pairs ``$(_tex(:widehat, "s"))_i,$(_tex(:widehat, "y"))_i``
are transported into the tangent space ``$(_math(:TpM; p = "p_{k+1}"))``.
After that the new values ``s_k = $(_tex(:widehat, "s"))_k = T^{S}_{p_k, α_k η_k}(α_k η_k)`` and ``y_k = $(_tex(:widehat, "y"))_k``
are stored at the beginning. This process ensures that new information about the objective
function is always included and the old, probably no longer relevant, information is discarded.

# Provided functors

* `(mp::AbstractManoptproblem, st::QuasiNewtonState) -> η` to compute the update direction
* `(η, mp::AbstractManoptproblem, st::QuasiNewtonState) -> η` to compute the update direction in-place of `η`

# Fields

* `memory_s`:                the set of the stored (and transported) search directions times step size `` $(_math(:Sequence, _tex(:widehat, "s"), "i", "k-m", "k-1"))``.
* `memory_y`:                set of the stored gradient differences ``$(_math(:Sequence, _tex(:widehat, "y"), "i", "k-m", "k-1"))``.
* `ξ`:                       a variable used in the two-loop recursion.
* `ρ`L                       a variable used in the two-loop recursion.
* `initial_scale`:           initial scaling of the Hessian, deactivate (e.g. when using a preconditioner) by passing `nothing`
$(_var(:Field, :vector_transport_method))
* `message`:                 a string containing a potential warning that might have appeared
* `project!`:                a function to stabilize the update by projecting on the tangent space
* `vector_transport_method`: method for transporting stored s and y directions to the new point
* `nonpositive_curvature_behavior`: how non-positive-definite pairs (s, y) are detected and handled in vector transport.
                             Allowed values are:
                                - `:ignore` (default): pairs whose inner product is zero are
                                  omitted from the current Hessian approximation but are
                                  retained in memory for further iterations. This may lead
                                  to non-positive-definite Hessians and non-descent directions
                                  being selected and thus needs to be handled elsewhere.
                                - `:byrd`: pairs such that `inner(M, p, X_s, Y_s) <= iszero_abstol * norm(M, p, Y_s)^2`
                                  are removed from memory (see [ByrdLuNocedalZhu:1995](@cite),
                                  Eq. (3.9) and its discussion).
* `sy_tol`:                  tolerance for detecting non-positive-definite pairs (X_s, X_y).
                             The pairs may lose positive-definiteness after vector transport.

# Constructor

    QuasiNewtonLimitedMemoryDirectionUpdate(
        M::AbstractManifold,
        x,
        update::AbstractQuasiNewtonUpdateRule,
        memory_size::Int;
        initial_vector=zero_vector(M,x),
        initial_scale::Real=1.0
        project!=copyto!
    )

# See also

[`InverseBFGS`](@ref)
[`QuasiNewtonCautiousDirectionUpdate`](@ref)
[`AbstractQuasiNewtonDirectionUpdate`](@ref)
"""
mutable struct QuasiNewtonLimitedMemoryDirectionUpdate{
        NT <: AbstractQuasiNewtonUpdateRule,
        T,
        F,
        V <: AbstractVector{F},
        G <: Union{F, Nothing},
        VT <: AbstractVectorTransportMethod,
        Proj,
    } <: AbstractQuasiNewtonDirectionUpdate
    memory_s::CircularBuffer{T}
    memory_y::CircularBuffer{T}
    ξ::Vector{F}
    ρ::Vector{F}
    initial_scale::G
    project!::Proj
    vector_transport_method::VT
    nonpositive_curvature_behavior::Symbol
    sy_tol::F
    message::String
end
function QuasiNewtonLimitedMemoryDirectionUpdate(
        M::AbstractManifold,
        p,
        ::NT,
        memory_size::Int;
        initial_vector::T = zero_vector(M, p),
        initial_scale::G = 1.0,
        (project!)::Proj = (copyto!),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        nonpositive_curvature_behavior::Symbol = :ignore,
        sy_tol::Real = 1.0e-8,
    ) where {
        NT <: AbstractQuasiNewtonUpdateRule,
        T,
        VTM <: AbstractVectorTransportMethod,
        Proj,
        G <: Union{<:Real, Nothing},
    }
    s = isnothing(initial_scale) ? (p, initial_vector) : (p, initial_vector, initial_scale)
    mT = allocate_result_type(M, QuasiNewtonLimitedMemoryDirectionUpdate, s)
    m1 = zeros(mT, memory_size)
    m2 = zeros(mT, memory_size)
    _initial_state = !isnothing(initial_scale) ? convert(mT, initial_scale) : initial_scale
    return QuasiNewtonLimitedMemoryDirectionUpdate{
        NT, T, mT, typeof(m1), typeof(_initial_state), VTM, Proj,
    }(
        CircularBuffer{T}(memory_size),
        CircularBuffer{T}(memory_size),
        m1,
        m2,
        _initial_state,
        project!,
        vector_transport_method,
        nonpositive_curvature_behavior,
        sy_tol,
        "",
    )
end
get_message(d::QuasiNewtonLimitedMemoryDirectionUpdate) = d.message
function status_summary(d::QuasiNewtonLimitedMemoryDirectionUpdate{T}) where {T}
    s = "limited memory $T (size $(length(d.memory_s)))"
    !isnothing(d.initial_scale) && (s = "$(s) initial scaling $(d.initial_scale)")
    (d.project! !== copyto!) && (s = "$(s), projections, ")
    s = "$(s)and $(d.vector_transport_method) as vector transport."
    return s
end
function (d::QuasiNewtonLimitedMemoryDirectionUpdate{InverseBFGS})(
        mp::AbstractManoptProblem, st
    )
    r = zero_vector(get_manifold(mp), get_iterate(st))
    return d(r, mp, st)
end
function (d::QuasiNewtonLimitedMemoryDirectionUpdate{InverseBFGS})(
        r, mp::AbstractManoptProblem, st
    )
    isempty(d.message) || (d.message = "") # reset message
    M = get_manifold(mp)
    p = get_iterate(st)
    copyto!(M, r, p, get_gradient(st))
    m = length(d.memory_s)
    if m == 0
        r .*= -1
        return r
    end
    # backward pass
    for i in m:-1:1
        # d.ρ is precomputed in the Hessian update
        d.ξ[i] = inner(M, p, d.memory_s[i], r) * d.ρ[i]
        r .-= d.ξ[i] .* d.memory_y[i]
    end
    last_safe_index = -1
    for i in eachindex(d.ρ)
        if abs(d.ρ[i]) > 0
            last_safe_index = i
        end
    end
    if (last_safe_index == -1)
        d.message = "$(d.message)$(length(d.message) > 0 ? :"\n" : "")"
        d.message = "$(d.message) All memory yield zero inner products, falling back to a gradient step."

        r .*= -1
        return r
    end
    # initial scaling guess
    if !isnothing(d.initial_scale)
        r .*=
            d.initial_scale /
            (d.ρ[last_safe_index] * norm(M, p, d.memory_y[last_safe_index])^2)
    end
    # precon
    st.preconditioner(r, mp, st)
    #
    # forward pass
    for i in eachindex(d.ρ)
        if abs(d.ρ[i]) > 0
            coeff = d.ξ[i] - d.ρ[i] * inner(M, p, d.memory_y[i], r)
            r .+= coeff .* d.memory_s[i]
        end
    end
    # potentially stabilize step by projecting.
    d.project!(M, r, p, r)
    r .*= -1
    return r
end

"""
    initialize_update!(d::QuasiNewtonLimitedMemoryDirectionUpdate)

Initialize the limited memory direction update by emptying the memory buffers.
"""
function initialize_update!(d::QuasiNewtonLimitedMemoryDirectionUpdate)
    empty!(d.memory_s)
    empty!(d.memory_y)
    fill!(d.ρ, 0)
    fill!(d.ξ, 0)
    return d
end

@doc """
    QuasiNewtonCautiousDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

These [`AbstractQuasiNewtonDirectionUpdate`](@ref)s represent any quasi-Newton update rule,
which are based on the idea of a so-called cautious update. The search direction is calculated
as given in [`QuasiNewtonMatrixDirectionUpdate`](@ref) or [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref),
but the update then is only executed if

```math
$(_tex(:frac, "g_{x_{k+1}}(y_k,s_k)", "$(_tex(:norm, "s_k"; index = "x_{k+1}"))^{2}")) ≥ θ $(_tex(:norm, "$(_tex(:grad))f(p_k)"; index = "p_k")),
```

is satisfied, where ``θ`` is a monotone increasing function satisfying ``θ(0) = 0``
and ``θ`` is strictly increasing at ``0``. If this is not the case, the corresponding
update is skipped, which means that for [`QuasiNewtonMatrixDirectionUpdate`](@ref)
the matrix ``H_k`` or ``B_k`` is not updated.
The basis ``$(_math(:Sequence, "b", "i", "1", "n"))`` is nevertheless transported into the upcoming tangent
space ``T_{x_{k+1}} $(_math(:M))``, and for [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)
neither the oldest vector pair ``$(_tex(:widetilde, "s"))_{k−m}``, ``$(_tex(:widetilde, "y"))_{k−m}`` is
discarded nor the newest vector pair ``$(_tex(:widetilde, "s"))_k, $(_tex(:widetilde, "y"))_k`` is added
into storage, but all stored vector pairs ``$(_tex(:set, "$(_tex(:widetilde, "s"))_i, $(_tex(:widetilde, "y"))_i"))_{i=k-m}^{k-1}``
are transported into the tangent space ``T_{x_{k+1}} $(_math(:M))``.
If [`InverseBFGS`](@ref) or [`InverseBFGS`](@ref) is chosen as update, then the resulting
method follows the method of [HuangAbsilGallivan:2018](@cite),
taking into account that the corresponding step size is chosen.

# Provided functors

* `(mp::AbstractManoptProblem, st::QuasiNewtonState) -> η` to compute the update direction
* `(η, mp::AbstractManoptProblem, st::QuasiNewtonState) -> η` to compute the update direction in-place of `η`

# Fields

* `update`: an [`AbstractQuasiNewtonDirectionUpdate`](@ref)
* `θ`:      a monotone increasing function satisfying ``θ(0) = 0`` and ``θ`` is strictly increasing at ``0``.

# Constructor

    QuasiNewtonCautiousDirectionUpdate(U::QuasiNewtonMatrixDirectionUpdate; θ = identity)
    QuasiNewtonCautiousDirectionUpdate(U::QuasiNewtonLimitedMemoryDirectionUpdate; θ = identity)

Generate a cautious update for either a matrix based or a limited memory based update rule.

# See also

[`QuasiNewtonMatrixDirectionUpdate`](@ref)
[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)
"""
mutable struct QuasiNewtonCautiousDirectionUpdate{U, Tθ} <:
    AbstractQuasiNewtonDirectionUpdate where {
        U <: Union{QuasiNewtonMatrixDirectionUpdate, QuasiNewtonLimitedMemoryDirectionUpdate},
    }
    update::U
    θ::Tθ
end
function QuasiNewtonCautiousDirectionUpdate(
        update::U; θ::Function = identity
    ) where {U <: Union{QuasiNewtonMatrixDirectionUpdate, QuasiNewtonLimitedMemoryDirectionUpdate}}
    return QuasiNewtonCautiousDirectionUpdate{U, typeof(θ)}(update, θ)
end
(d::QuasiNewtonCautiousDirectionUpdate)(mp::AbstractManoptProblem, st) = d.update(mp, st)
function (d::QuasiNewtonCautiousDirectionUpdate)(r, mp::AbstractManoptProblem, st)
    return d.update(r, mp, st)
end

# access the inner vector transport method
function get_update_vector_transport(u::AbstractQuasiNewtonDirectionUpdate)
    return u.vector_transport_method
end
function get_update_vector_transport(u::QuasiNewtonCautiousDirectionUpdate)
    return get_update_vector_transport(u.update)
end
function initialize_update!(d::QuasiNewtonCautiousDirectionUpdate)
    initialize_update!(d.update)
    return d
end
