@doc """
    AbstractQuasiNewtonDirectionUpdate

An abstract representation of a Quasi Newton update rule to determine the next direction
given current [`QuasiNewtonState`](@ref).

All subtypes should be functors as well, callable as `H(mp, st)` and in-place as `H(η, mp, st)`,
given an `AbstractManoptProblem` `mp` and a [`QuasiNewtonState`](@ref) `st`, to compute a new update direction.
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

Specify a type for the different [`AbstractQuasiNewtonDirectionUpdate`](@ref)s.

For a [`QuasiNewtonMatrixDirectionUpdate`](@ref) there are several different updates to the matrix,
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
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:Manifold)),
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
  $(_tex(:Id))_{T_{x_{k+1}} $(_math(:Manifold))} - $(_tex(:frac, "s_k y^{$(_tex(:rm, "T"))}_k ", "s^{$(_tex(:rm, "T"))}_k y_k"))
$(_tex(:Bigr)))
$(_tex(:widetilde, "B"))^$(_tex(:rm, "BFGS"))_k
$(_tex(:Bigl))(
  $(_tex(:Id))_{T_{x_{k+1}} $(_math(:Manifold))} - $(_tex(:frac, "y_k s^{$(_tex(:rm, "T"))}_k ", "s^{$(_tex(:rm, "T"))}_k y_k"))
$(_tex(:Bigr))) + $(_tex(:frac, "s_k s^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:Manifold)),
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
  $(_tex(:Id))_{T_{x_{k+1}} $(_math(:Manifold))} - $(_tex(:frac, "y_k s^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
$(_tex(:Bigr)))
$(_tex(:widetilde, "H"))^$(_tex(:rm, "DFP"))_k
$(_tex(:Bigl))(
  $(_tex(:Id))_{T_{x_{k+1}} $(_math(:Manifold))} - $(_tex(:frac, "s_k y^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
$(_tex(:Bigr))) + $(_tex(:frac, "y_k y^{$(_tex(:rm, "T"))}_k", "s^{$(_tex(:rm, "T"))}_k y_k"))
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonState`](@ref)) of

```math
T^{S}_{p_k, α_k η_k}(α_k η_k) $(_tex(:quad))$(_tex(:text, "and"))$(_tex(:quad))
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:Manifold)),
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
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:Manifold)),
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
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:Manifold)),
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
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:Manifold)),
```

respectively.

This method can be stabilized by only performing the update if denominator is larger than
``r$(_tex(:norm, "y_k"; index = "x_{k+1}"))$(_tex(:norm, "s_k - $(_tex(:widetilde, "B"))^{$(_tex(:rm, "SR1"))}_k y_k"; index = "x_{k+1}"))``
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
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:Manifold)),
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

Denote by ``$(_tex(:widetilde, "B"))_k^{$(_tex(:rm, "Br"))}`` the operator concatenated with a vector transport
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
$(_tex(:grad))f(x_{k+1}) - T^{S}_{p_k, α_k η_k}($(_tex(:grad))f(p_k)) ∈ T_{x_{k+1}} $(_math(:Manifold)),
```

respectively, and ``φ_k`` is the Broyden factor which is `:constant` by default but can also be set to `:InverseDavidon`.

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
$(_tex(:text, "Solve"))$(_tex(:quad)) H_k $(_tex(:hat, "η_k")) = - $(_tex(:widehat, "$(_tex(:grad))f(p_k)")),
```
"""
_doc_QN_B_full_system = """
```math
$(_tex(:hat, "η_k")) = - B_k $(_tex(:widehat, "$(_tex(:grad))f(p_k)")),
```
"""

"""
    QuasiNewtonPreconditioner{F}

Add a preconditioning

# Fields

* `preconditioner!::F`: the preconditioner function

# Constructors

    QuasiNewtonPreconditioner(preconditioner)

Add preconditioning to a gradient problem.

# Input

* `preconditioner`:   preconditioner function, either as a `(M, p, X) -> Y` allocating or `(M, Y, p, X) -> Y` mutating function

# Keyword arguments
$(_kwargs(:evaluation))
"""
struct QuasiNewtonPreconditioner{F}
    preconditioner!::F
    function QuasiNewtonPreconditioner(preconditioner::F; evaluation::AbstractEvaluationType = AllocatingEvaluation()) where {F}
        preconditioner_ = maybe_wrap_function(preconditioner, evaluation)
        return new{typeof(preconditioner_)}(preconditioner_)
    end
end
function (pg::QuasiNewtonPreconditioner)(
        X, mp::AbstractManoptProblem, s::AbstractGradientSolverState
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    return pg.preconditioner!(M, X, p, X)
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
space ``$(_math(:TangentSpace; p = "p_{k+1}"))``, preferably with an isometric vector transport, or generated there.

# Provided functors

* `(mp::AbstractManoptProblem, st::QuasiNewtonState) -> η` to compute the update direction
* `(η, mp::AbstractManoptProblem, st::QuasiNewtonState) -> η` to compute the update direction in-place of `η`

# Fields

* `basis`:                  an `AbstractBasis` to use in the tangent spaces
* `matrix`:                 the matrix which represents the approximating operator.
* `initial_scale`:          when initializing the update, a unit matrix is used as initial approximation, scaled by this factor
* `update`:                 a [`AbstractQuasiNewtonUpdateRule`](@ref).
$(_fields(:vector_transport_method))

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
$(_kwargs(:vector_transport_method))

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
function status_summary(d::QuasiNewtonMatrixDirectionUpdate; context::Symbol = :default)
    (context === :short) && return repr(d)
    scale = isnothing(d.initial_scale) ? "deactivated" : "$(d.initial_scale)"
    (context === :inline) &&
        return "A quasi Newton direction update using $(d.update), stored as a matrix."
    return """
    A quasi Newton direction update stored as a matrix

    ## Parameters
    * update rule:             $(_MANOPT_INDENT)$(d.update)
    * basis:                   $(_MANOPT_INDENT)$(d.basis)
    * initial scaling:         $(_MANOPT_INDENT)$(scale)
    * matrix size:             $(_MANOPT_INDENT)$(size(d.matrix, 1))×$(size(d.matrix, 2))
    * vector transport method: $(_MANOPT_INDENT)$(d.vector_transport_method)
    """
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
    copyto!(M, r, p, X)
    st.preconditioner(r, mp, st)
    get_vector!(M, r, p, -d.matrix \ get_coordinates(M, p, r, d.basis), d.basis)
    return r
end
function initialize_update!(d::QuasiNewtonMatrixDirectionUpdate)
    copyto!(d.matrix, I)
    return d
end
"""
    hessian_value_diag(d::QuasiNewtonMatrixDirectionUpdate, M, p, X)

Evaluate the quadratic form associated with the stored quasi-Newton matrix.
Returns the scalar ``c^{$(_tex(:transp))} B c`` where ``c`` are the coordinates of the
tangent vector `X` at `p` (in the basis `d.basis`) and ``B`` is `d.matrix`.
"""
function hessian_value_diag(d::QuasiNewtonMatrixDirectionUpdate{T}, M::AbstractManifold, p, X) where {T <: Union{BFGS, DFP, SR1, Broyden}}
    c = get_coordinates(M, p, X, d.basis)
    return dot(c, d.matrix, c)
end

"""
    hessian_value_diag(d::QuasiNewtonMatrixDirectionUpdate, M, p, X::UnitVector)

Evaluate the quadratic form associated with the stored quasi-Newton matrix.
Returns the scalar ``c^{$(_tex(:transp))} B c`` where ``c`` are the coordinates of the
[`UnitVector`](@ref) `X` at `p` (in the basis `d.basis`) and ``B`` is `d.matrix`.
"""
function hessian_value_diag(d::QuasiNewtonMatrixDirectionUpdate{T}, M::AbstractManifold, p, X::UnitVector) where {T <: Union{BFGS, DFP, SR1, Broyden}}
    b = to_coordinate_index(M, X, d.basis)
    return d.matrix[b, b]
end
"""
    hessian_value(d::QuasiNewtonMatrixDirectionUpdate, M, p, X::UnitVector, Y)

Evaluate the quadratic form associated with the stored quasi-Newton matrix.
Returns the scalar ``c_b^{$(_tex(:transp))} B c`` where ``c_b`` are the coordinates of the
[`UnitVector`](@ref) `X` at `p` (assumed to correspond to the basis `d.basis`),
``c`` are the coordinates of the tangent vector `Y` at `p` (in the basis `d.basis`)
and ``B`` is `d.matrix`.
"""
function hessian_value(d::QuasiNewtonMatrixDirectionUpdate{T}, M::AbstractManifold, p, X::UnitVector, Y) where {T <: Union{BFGS, DFP, SR1, Broyden}}
    b = to_coordinate_index(M, X, d.basis)
    return dot(d.matrix[b, :], get_coordinates(M, p, Y, d.basis))
end

_doc_QN_B = """
```math
$(_tex(:Cal, "B"))_k^{(0)}[⋅]
= $(_tex(:frac, "$(_tex(:inner, "s_{k-1}", "y_{k-1}"; index = "p_k"))", "$(_tex(:inner, "y_{k-1}", "y_{k-1}"; index = "p_k"))"))$(_tex(:Id))_{$(_math(:TangentSpace))}[⋅]
```
"""

@doc """
    QuasiNewtonLimitedMemoryDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

This [`AbstractQuasiNewtonDirectionUpdate`](@ref) represents the limited-memory Riemannian BFGS update.

The approximating operator is represented by ``m`` stored pairs of tangent
vectors ``$(_math(:Sequence, "$(_tex(:widetilde, "s"))", "i", "k-m", "k-1"))`` and ``$(_math(:Sequence, "$(_tex(:widetilde, "y"))", "i", "k-m", "k-1"))``
in the ``k``-th iteration. For the calculation of the search direction ``X_k``, the generalization
of the two-loop recursion is used (see [HuangGallivanAbsil:2015](@cite)),
since it only requires inner products and linear combinations of tangent vectors in ``$(_math(:TangentSpace; p = "p_k"))``.
For that the stored pairs of tangent vectors ``s_i, y_i``,
the gradient ``$(_tex(:grad)) f(p_k)`` of the objective function ``f`` in ``p_k``
and the positive definite self-adjoint operator

$(_doc_QN_B)

are used. The two-loop recursion can be understood as that the [`InverseBFGS`](@ref) update
is executed ``m`` times in a row on ``$(_tex(:Cal, "B"))^{(0)}_k[⋅]`` using the tangent vectors ``$(_tex(:widehat, "s"))_i,$(_tex(:widehat, "y"))_i``,
and in the same time the resulting operator ``$(_tex(:Cal, "B"))^{$(_tex(:rm, "LRBFGS"))}_k [⋅]`` is directly applied on ``$(_tex(:grad))f(p_k)``.
When updating there are two cases: if there is still free memory, ``k < m``, the previously
stored vector pairs ``$(_tex(:widehat, "s"))_i,$(_tex(:widehat, "y"))_i`` have to be
transported into the upcoming tangent space ``$(_math(:TangentSpace; p = "p_{k+1}"))``.
If there is no free memory, the oldest pair ``$(_tex(:widehat, "s"))_i,$(_tex(:widehat, "y"))_i``
has to be discarded and then all the remaining vector pairs ``$(_tex(:widehat, "s"))_i,$(_tex(:widehat, "y"))_i``
are transported into the tangent space ``$(_math(:TangentSpace; p = "p_{k+1}"))``.
After that the new values ``s_k = $(_tex(:widehat, "s"))_k = T^{S}_{p_k, α_k η_k}(α_k η_k)`` and ``y_k = $(_tex(:widehat, "y"))_k``
are stored at the beginning. This process ensures that new information about the objective
function is always included and the old, probably no longer relevant, information is discarded.

# Provided functors

* `(mp::AbstractManoptProblem, st::QuasiNewtonState) -> η` to compute the update direction
* `(η, mp::AbstractManoptProblem, st::QuasiNewtonState) -> η` to compute the update direction in-place of `η`

# Fields

* `memory_s`:                the set of the stored (and transported) search directions times step size `` $(_math(:Sequence, _tex(:widehat, "s"), "i", "k-m", "k-1"))``.
* `memory_y`:                set of the stored gradient differences ``$(_math(:Sequence, _tex(:widehat, "y"), "i", "k-m", "k-1"))``.
* `ξ`:                       a variable used in the two-loop recursion.
* `ρ`:                       a variable used in the two-loop recursion.
* `initial_scale`:           initial scaling of the Hessian, deactivate (e.g. when using a preconditioner) by passing `nothing`
$(_fields(:vector_transport_method))
* `message`:                 a string containing a potential warning that might have appeared
* `project!`:                a function to stabilize the update by projecting on the tangent space
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
        p,
        update::AbstractQuasiNewtonUpdateRule,
        memory_size::Int;
        initial_vector=zero_vector(M, p),
        initial_scale::Real=1.0,
        project!=copyto!,
        vector_transport_method=default_vector_transport_method(M, typeof(p)),
        nonpositive_curvature_behavior::Symbol=:ignore,
        sy_tol::Real=1.0e-8,
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
        M::AbstractManifold, p, ::NT, memory_size::Int;
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
function status_summary(
        d::QuasiNewtonLimitedMemoryDirectionUpdate{T}; context::Symbol = :default
    ) where {T}
    (context === :short) && return repr(d)
    scale = isnothing(d.initial_scale) ? "deactivated" : "$(d.initial_scale)"
    (context === :inline) &&
        return "A limited memory $(T) direction update of memory size $(capacity(d.memory_s))."
    return """
    A limited memory quasi Newton direction update

    ## Parameters
    * update rule:                    $(_MANOPT_INDENT)$(T)
    * memory size:                    $(_MANOPT_INDENT)$(capacity(d.memory_s))
    * currently stored:               $(_MANOPT_INDENT)$(length(d.memory_s))
    * initial scaling:                $(_MANOPT_INDENT)$(scale)
    * projection:                     $(_MANOPT_INDENT)$(d.project! === copyto! ? "none" : "$(d.project!)")
    * nonpositive curvature behavior: $(_MANOPT_INDENT):$(d.nonpositive_curvature_behavior)
    * tolerance sy_tol:               $(_MANOPT_INDENT)$(d.sy_tol)
    * vector transport method:        $(_MANOPT_INDENT)$(d.vector_transport_method)
    """
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

function show(io::IO, qns::QuasiNewtonLimitedMemoryDirectionUpdate)
    return print(io, "QuasiNewtonLimitedMemoryDirectionUpdate with memory size $(capacity(qns.memory_s)) and $(qns.vector_transport_method) as vector transport.")
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
space ``T_{x_{k+1}} $(_math(:Manifold))``, and for [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)
neither the oldest vector pair ``$(_tex(:widetilde, "s"))_{k-m}``, ``$(_tex(:widetilde, "y"))_{k-m}`` is
discarded nor the newest vector pair ``$(_tex(:widetilde, "s"))_k, $(_tex(:widetilde, "y"))_k`` is added
into storage, but all stored vector pairs ``$(_tex(:set, "$(_tex(:widetilde, "s"))_i, $(_tex(:widetilde, "y"))_i"))_{i=k-m}^{k-1}``
are transported into the tangent space ``T_{x_{k+1}} $(_math(:Manifold))``.
If [`InverseBFGS`](@ref) is chosen as update — either within a [`QuasiNewtonMatrixDirectionUpdate`](@ref)
or a [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) — then the resulting
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
function status_summary(d::QuasiNewtonCautiousDirectionUpdate; context::Symbol = :default)
    (context === :short) && return repr(d)
    (context === :inline) &&
        return "A cautious direction update with θ = $(d.θ); internally: $(status_summary(d.update; context = :inline))"
    return """
    A cautious quasi Newton direction update

    ## Parameters
    * θ: $(_MANOPT_INDENT)$(d.θ)

    ## Internal direction update
    $(_in_str(status_summary(d.update; context = context); indent = 1, headers = 1))
    """
end
function show(io::IO, d::QuasiNewtonCautiousDirectionUpdate)
    print(io, "QuasiNewtonCautiousDirectionUpdate with θ = $(d.θ) and internal state:\n")
    return print(io, d.update)
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

#
#
# ---
@doc raw"""
    QuasiNewtonLimitedMemoryBoxDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

A limited memory quasi-Newton direction update with support for box constraints.

It stores an approximation of the Hessian of a scalar function in the compact form
``B_k = θ_k I - W_k M_k W_k^{\mathrm{T}}``,
where ``θ_k > 0`` is the current scaling factor stored in `current_scale`;
while the memory is empty, ``B_k = θ^{-1} I`` holds for the initial scaling guess ``θ > 0``.
Matrix ``M_k = \left(\begin{smallmatrix}M₁₁ & M₂₁^{\mathrm{T}}\\ M₂₁ & M₂₂\end{smallmatrix}\right)``
is stored using its blocks.
Blocks ``W_k`` are (implicitly) composed from `memory_y` and `memory_s` stored in `qn_du`
of type [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref).

Initial scale ``θ`` is stored in the field `initial_scale`; if the memory is not empty,
the current scale is set to ``\frac{\|y_k\|^2}{⟨s_k, y_k⟩ θ}``, where ``k`` is the most recent
index for which ``⟨s_k, y_k⟩`` is not equal to 0.

`last_gcd_result` stores the result of the last generalized Cauchy direction search.

See [ByrdNocedalSchnabel:1994](@cite) for details.
"""
mutable struct QuasiNewtonLimitedMemoryBoxDirectionUpdate{
        TDU <: QuasiNewtonLimitedMemoryDirectionUpdate,
        F <: Real,
        T_HM <: AbstractMatrix,
        V <: AbstractVector,
    } <: AbstractQuasiNewtonDirectionUpdate
    # this approximates inverse Hessian
    qn_du::TDU

    # fields for approximating the Hessian
    current_scale::F
    M_11::T_HM
    M_21::T_HM
    M_22::T_HM
    # buffer for calculating W_k blocks
    buffer_inner_Sk_X::V
    buffer_inner_Sk_Y::V
    buffer_inner_Yk_X::V
    buffer_inner_Yk_Y::V
    last_gcd_result::Symbol
    last_gcd_stepsize::F
end

# a deactivated `initial_scale` (`nothing`, used together with a preconditioner) acts as 1,
# the same convention as `update_hessian!` in src/solvers/quasi_Newton.jl
function _box_initial_scale(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    return isnothing(gh.qn_du.initial_scale) ? one(gh.current_scale) : gh.qn_du.initial_scale
end

function status_summary(
        d::QuasiNewtonLimitedMemoryBoxDirectionUpdate; context::Symbol = :default
    )
    (context === :short) && return repr(d)
    (context === :inline) &&
        return "A limited memory direction update with support for box constraints; internally: $(status_summary(d.qn_du; context = :inline))"
    return """
    A limited memory quasi Newton direction update with support for box constraints

    ## Parameters
    * current scale:                $(_MANOPT_INDENT)$(d.current_scale)
    * last Cauchy direction result: $(_MANOPT_INDENT):$(d.last_gcd_result)
    * last Cauchy step size:        $(_MANOPT_INDENT)$(d.last_gcd_stepsize)

    ## Internal direction update
    $(_in_str(status_summary(d.qn_du; context = context); indent = 1, headers = 1))
    """
end

function get_parameter(d::QuasiNewtonLimitedMemoryBoxDirectionUpdate, ::Val{:max_stepsize})
    if d.last_gcd_result === :found_limited
        return d.last_gcd_stepsize
    else
        return Inf
    end
end

function QuasiNewtonLimitedMemoryBoxDirectionUpdate(
        qn_du::QuasiNewtonLimitedMemoryDirectionUpdate{<:AbstractQuasiNewtonUpdateRule, T, F}
    ) where {T, F <: Real}
    memory_size = capacity(qn_du.memory_s)
    M_11 = zeros(F, memory_size, memory_size)
    M_21 = zeros(F, memory_size, memory_size)
    M_22 = zeros(F, memory_size, memory_size)
    buffer_inner_Sk_X = zeros(F, memory_size)
    buffer_inner_Sk_Y = zeros(F, memory_size)
    buffer_inner_Yk_X = zeros(F, memory_size)
    buffer_inner_Yk_Y = zeros(F, memory_size)
    return QuasiNewtonLimitedMemoryBoxDirectionUpdate{
        typeof(qn_du), F, typeof(M_11), typeof(buffer_inner_Sk_X),
    }(
        qn_du,
        isnothing(qn_du.initial_scale) ? one(F) : convert(F, qn_du.initial_scale),
        M_11,
        M_21,
        M_22,
        buffer_inner_Sk_X,
        buffer_inner_Sk_Y,
        buffer_inner_Yk_X,
        buffer_inner_Yk_Y,
        :not_searched,
        NaN,
    )
end

function initialize_update!(ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    initialize_update!(ha.qn_du)
    ha.last_gcd_result = :not_searched
    return ha
end

function (d::QuasiNewtonLimitedMemoryBoxDirectionUpdate)(
        mp::AbstractManoptProblem, st
    )
    r = zero_vector(get_manifold(mp), get_iterate(st))
    return d(r, mp, st)
end
function (d::QuasiNewtonLimitedMemoryBoxDirectionUpdate)(
        r, mp::AbstractManoptProblem, st
    )
    d.qn_du(r, mp, st)
    M = get_manifold(mp)
    p = get_iterate(st)
    X = get_gradient(st)
    gcd = GeneralizedCauchyDirectionSubsolver(M, p, d)
    d.last_gcd_result, d.last_gcd_stepsize = find_generalized_cauchy_direction!(M, gcd, r, p, r, X)
    return r
end

get_update_vector_transport(u::QuasiNewtonLimitedMemoryBoxDirectionUpdate) = get_update_vector_transport(u.qn_du)

function get_at_bound_index(M::ProductManifold, X, b::Tuple{Int, Any})
    return get_at_bound_index(M.manifolds[b[1]], submanifold_component(M, X, b[1]), b[2])
end

@doc raw"""
    hessian_value_diag(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X)

Compute ``⟨X, B X⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`.
"""
function hessian_value_diag(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X)
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    normX_sqr = norm(M, p, X)^2

    if m == 0 || num_nonzero_rho == 0
        return _box_initial_scale(gh) \ normX_sqr
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.buffer_inner_Yk_X[ii] = inner(M, p, gh.qn_du.memory_y[i], X)
        gh.buffer_inner_Sk_X[ii] = gh.current_scale * inner(M, p, gh.qn_du.memory_s[i], X)

        ii += 1
    end
    buffer_inner_Yk = view(gh.buffer_inner_Yk_X, 1:num_nonzero_rho)
    buffer_inner_Sk = view(gh.buffer_inner_Sk_X, 1:num_nonzero_rho)

    return hessian_value_from_inner_products(gh, normX_sqr, buffer_inner_Yk, buffer_inner_Sk, buffer_inner_Yk, buffer_inner_Sk)
end

@doc raw"""
    hessian_value_diag(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X::UnitVector)

Compute ``⟨X, B X⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`, and `X` is the
[`UnitVector`](@ref).
"""
function hessian_value_diag(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X::UnitVector)
    b = X.index
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    if m == 0 || num_nonzero_rho == 0
        return inv(_box_initial_scale(gh))
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.buffer_inner_Yk_X[ii] = get_at_bound_index(M, gh.qn_du.memory_y[i], b)
        gh.buffer_inner_Sk_X[ii] = gh.current_scale * get_at_bound_index(M, gh.qn_du.memory_s[i], b)

        ii += 1
    end
    buffer_inner_Yk = view(gh.buffer_inner_Yk_X, 1:num_nonzero_rho)
    buffer_inner_Sk = view(gh.buffer_inner_Sk_X, 1:num_nonzero_rho)

    return hessian_value_from_inner_products(gh, one(eltype(gh.qn_du.ρ)), buffer_inner_Yk, buffer_inner_Sk, buffer_inner_Yk, buffer_inner_Sk)
end

@doc raw"""
    hessian_value(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X::UnitVector, Y)

Compute ``⟨X, B Y⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`, where `X` is the
[`UnitVector`](@ref).
"""
function hessian_value(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X::UnitVector, Y)
    b = X.index

    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    Yb = get_at_bound_index(M, Y, b)
    if m == 0 || num_nonzero_rho == 0
        return _box_initial_scale(gh) \ Yb
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.buffer_inner_Yk_X[ii] = get_at_bound_index(M, gh.qn_du.memory_y[i], b)
        gh.buffer_inner_Sk_X[ii] = gh.current_scale * get_at_bound_index(M, gh.qn_du.memory_s[i], b)

        gh.buffer_inner_Yk_Y[ii] = inner(M, p, gh.qn_du.memory_y[i], Y)
        gh.buffer_inner_Sk_Y[ii] = gh.current_scale * inner(M, p, gh.qn_du.memory_s[i], Y)
        ii += 1
    end
    buffer_inner_Yk_X = view(gh.buffer_inner_Yk_X, 1:num_nonzero_rho)
    buffer_inner_Yk_Y = view(gh.buffer_inner_Yk_Y, 1:num_nonzero_rho)
    buffer_inner_Sk_X = view(gh.buffer_inner_Sk_X, 1:num_nonzero_rho)
    buffer_inner_Sk_Y = view(gh.buffer_inner_Sk_Y, 1:num_nonzero_rho)

    return hessian_value_from_inner_products(gh, Yb, buffer_inner_Yk_X, buffer_inner_Sk_X, buffer_inner_Yk_Y, buffer_inner_Sk_Y)
end

@doc raw"""
    update_current_scale!(M::AbstractManifold, p, gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate)

Refresh the scaling factor and blockwise Hessian approximation stored in `gh` using the
nonzero curvature pairs currently in memory.

- Identifies the most recent index with nonzero ``ρ_i`` to scale the initial Hessian guess
    by ``ρ_i‖y_i‖^2 / θ``.
- Builds ``L_k`` and ``S_k^\top S_k`` from the stored ``(s_i, y_i)`` pairs and updates the
    block matrices ``M₁₁``, ``M₂₁``, and ``M₂₂`` via the blockwise inverse formula.
- If all ``ρ_i`` vanish, resets `current_scale` to the inverse of `initial_scale` and
    clears the block matrices.

Returns the mutated `gh`.
"""
function update_current_scale!(M::AbstractManifold, p, gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    m = length(gh.qn_du.memory_s)
    last_safe_index = -1
    for i in eachindex(gh.qn_du.ρ)
        if abs(gh.qn_du.ρ[i]) > 0
            last_safe_index = i
        end
    end

    if (last_safe_index == -1)
        # All memory yield zero inner products
        gh.current_scale = inv(_box_initial_scale(gh))
        gh.M_11 = fill(0.0, 0, 0)
        gh.M_21 = fill(0.0, 0, 0)
        gh.M_22 = fill(0.0, 0, 0)
        return gh
    end

    invA = Diagonal([-ri for ri in gh.qn_du.ρ if !iszero(ri)])
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    Lk = LowerTriangular(zeros(num_nonzero_rho, num_nonzero_rho))

    # total scaling factor for the initial Hessian
    # written this way to avoid floating point overflow (when ynorm is finite but ynorm^2 is Inf)
    # see CUTEst EXPQUAD problem for an example
    ynorm = norm(M, p, gh.qn_du.memory_y[last_safe_index])
    gh.current_scale = ((gh.qn_du.ρ[last_safe_index] * ynorm) * ynorm) / _box_initial_scale(gh)

    tsksk = Symmetric(zeros(num_nonzero_rho, num_nonzero_rho))
    ii = 1
    # fill Dk and Lk
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        jj = 1
        for j in 1:m
            iszero(gh.qn_du.ρ[j]) && continue
            if jj < ii
                Lk[ii, jj] = inner(M, p, gh.qn_du.memory_s[i], gh.qn_du.memory_y[j])
            end
            if ii <= jj
                tsksk.data[ii, jj] = inner(M, p, gh.qn_du.memory_s[i], gh.qn_du.memory_s[j])
            end
            jj += 1
        end
        ii += 1
    end
    tsksk.data .*= gh.current_scale

    # matrix inversion using the blockwise formula for speed
    # Schur complement of -Dk is the only non-diagonal matrix we actually need to inverse in this step
    W1 = Lk * invA
    W2 = W1 * Lk'
    gh.M_22 = inv(Symmetric(tsksk - W2))
    W3 = gh.M_22 * W1
    W4 = W1' * W3

    gh.M_11 = invA + W4
    gh.M_21 = -W3

    return gh
end

@doc raw"""
    hessian_value_from_inner_products(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, iss::Real, cy1, cs1, cy2, cs2)

Evaluate the quadratic form defined by the current blockwise Hessian approximation stored in
`gh`, given precomputed coordinate vectors.

Arguments:
- `iss`: inner product of original vectors.
- `cy1`, `cy2`: coordinates of ``y``-like vectors in the ``Y_k`` basis.
- `cs1`, `cs2`: coordinates of ``s``-like vectors in the scaled ``S_k`` basis.

The result is ``θ·iss - cy₁ᵀ M₁₁ cy₂ - cs₁ᵀ M₂₁ cy₂ - cs₂ᵀ M₂₁ cy₁ - cs₁ᵀ M₂₂ cs₂`` using the blocks
``M₁₁``, ``M₂₁``, ``M₂₂`` stored in `gh` and the current scale ``θ``. Returns the scalar value.
"""
function hessian_value_from_inner_products(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, iss::Real, cy1, cs1, cy2, cs2)
    result = gh.current_scale * iss
    if length(cy1) == 0
        return result
    end
    result -= dot(cy1, gh.M_11, cy2)
    result -= dot(cs1, gh.M_21, cy2) + dot(cs2, gh.M_21, cy1)
    result -= dot(cs1, gh.M_22, cs2)

    return result
end


@doc raw"""
    update_hessian!(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, mp, st, p_old, k)

Update the Hessian approximation `gh` by moving it from the previous point `p_old` to the current iterate
and updating the stored `s` and `y` vectors, respectively.
"""
function update_hessian!(
        gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate,
        mp::AbstractManoptProblem,
        st::AbstractManoptSolverState,
        p_old,
        k::Int,
    )
    (capacity(gh.qn_du.memory_s) == 0) && return gh
    update_hessian!(gh.qn_du, mp, st, p_old, k)
    update_current_scale!(get_manifold(mp), get_iterate(st), gh)
    return gh
end


"""
    abstract type AbstractSegmentHessianUpdater end

Abstract type for methods that calculate f' and f'' in the GCD calculation in subsequent
line segments in [`GeneralizedCauchyDirectionSubsolver`](@ref).
"""
abstract type AbstractSegmentHessianUpdater end

"""
    init_updater!(::AbstractManifold, hessian_segment_updater::AbstractSegmentHessianUpdater, p, d, ha)

Method for initialization of `AbstractSegmentHessianUpdater` `hessian_segment_updater` just before the loop
that examines subsequent intervals for GCD.
"""
init_updater!(::AbstractManifold, hessian_segment_updater::AbstractSegmentHessianUpdater, p, d, ha)

"""
    struct GenericSegmentHessianUpdater <: AbstractSegmentHessianUpdater end

Generic f' and f'' calculation that only relies on `hessian_value` but is relatively slow for
high-dimensional domains.
"""
struct GenericSegmentHessianUpdater{TX} <: AbstractSegmentHessianUpdater
    d_z::TX
    d_tmp::TX
end

function get_default_hessian_segment_updater(M::AbstractManifold, p, ::Any)
    return GenericSegmentHessianUpdater(zero_vector(M, p), zero_vector(M, p))
end

function init_updater!(M::AbstractManifold, hessian_segment_updater::GenericSegmentHessianUpdater, p, d, ha)
    zero_vector!(M, hessian_segment_updater.d_z, p)
    copyto!(M, hessian_segment_updater.d_tmp, d)
    return hessian_segment_updater
end

@doc raw"""
    (upd::GenericSegmentHessianUpdater)(M::AbstractManifold, p, t::Real, dt::Real, b, db, ha)

Calculate Hessian values ``⟨e_b, B d_z⟩`` and ``⟨e_b, B d_tmp⟩`` for the generalized Cauchy
point line search using the generic approach via `hessian_value` with [`UnitVector`](@ref).
``d_z`` start with 0 and is updated in-place by adding `dt * d` to it.
"""
function (upd::GenericSegmentHessianUpdater)(M::AbstractManifold, p, t::Real, dt::Real, b, db, ha)
    upd.d_z .+= dt .* upd.d_tmp
    hv_eb_dz = hessian_value(ha, M, p, UnitVector(b), upd.d_z)
    hv_eb_d = hessian_value(ha, M, p, UnitVector(b), upd.d_tmp)

    set_zero_at_index!(M, upd.d_tmp, b)

    return hv_eb_dz, hv_eb_d
end

"""
    struct LimitedMemorySegmentHessianUpdater{TV <: AbstractVector} <: AbstractSegmentHessianUpdater

Hessian value calculation for generalized Cauchy direction line segments that is optimized for
[`QuasiNewtonLimitedMemoryBoxDirectionUpdate`](@ref). It relies on a specific Hessian structure.
"""
struct LimitedMemorySegmentHessianUpdater{TV <: AbstractVector} <: AbstractSegmentHessianUpdater
    p_s::TV
    p_y::TV
    c_s::TV
    c_y::TV
end

function get_default_hessian_segment_updater(::AbstractManifold, p, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    return LimitedMemorySegmentHessianUpdater(similar(ha.qn_du.ρ), similar(ha.qn_du.ρ), similar(ha.qn_du.ρ), similar(ha.qn_du.ρ))
end

function init_updater!(M::AbstractManifold, hessian_segment_updater::LimitedMemorySegmentHessianUpdater, p, d, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    fill!(hessian_segment_updater.c_s, 0)
    fill!(hessian_segment_updater.c_y, 0)
    ii = 1
    for i in eachindex(ha.qn_du.ρ)
        if iszero(ha.qn_du.ρ[i])
            continue
        end

        hessian_segment_updater.p_s[ii] = ha.current_scale * inner(M, p, ha.qn_du.memory_s[i], d)
        hessian_segment_updater.p_y[ii] = inner(M, p, ha.qn_du.memory_y[i], d)
        ii += 1
    end
    return hessian_segment_updater
end

@doc raw"""
    (hessian_segment_updater::LimitedMemorySegmentHessianUpdater)(
        M::AbstractManifold, p,
        t::Real, dt::Real, b, db, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate
    )

Calculate Hessian values ``⟨e_b, B d_z⟩`` and ``⟨e_b, B d⟩`` for the generalized Cauchy
point line search using the limited-memory block Hessian stored in `ha`.
``d_z`` start with 0 and is updated in-place by adding `dt * d` to it.

## Arguments:

- `M`: manifold.
- `p`: current iterate.
- `t`: current step length from `p`.
- `dt`: step length increment from the last step.
- `b`: bound index of the current segment.
- `db`: search direction component at the bound index `b`.

The updater reuses cached coordinate projections in `hessian_segment_updater` to cheaply
evaluate Hessian quadratic forms via `hessian_value_from_inner_products`.
"""
function (hessian_segment_updater::LimitedMemorySegmentHessianUpdater)(
        M::AbstractManifold, p,
        t::Real, dt::Real, b, db, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate
    )

    m = length(ha.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, ha.qn_du.ρ)

    ii = 1
    for i in 1:m
        iszero(ha.qn_du.ρ[i]) && continue
        # setting _X to w_b from the paper
        ha.buffer_inner_Yk_X[ii] = get_at_bound_index(M, ha.qn_du.memory_y[i], b)
        ha.buffer_inner_Sk_X[ii] = ha.current_scale * get_at_bound_index(M, ha.qn_du.memory_s[i], b)

        ii += 1
    end

    buffer_inner_Yk_eb = view(ha.buffer_inner_Yk_X, 1:num_nonzero_rho)
    buffer_inner_Sk_eb = view(ha.buffer_inner_Sk_X, 1:num_nonzero_rho)

    buffer_inner_cy = view(hessian_segment_updater.c_y, 1:num_nonzero_rho)
    buffer_inner_cs = view(hessian_segment_updater.c_s, 1:num_nonzero_rho)
    buffer_inner_py = view(hessian_segment_updater.p_y, 1:num_nonzero_rho)
    buffer_inner_ps = view(hessian_segment_updater.p_s, 1:num_nonzero_rho)

    buffer_inner_cy .+= dt .* buffer_inner_py
    buffer_inner_cs .+= dt .* buffer_inner_ps

    eb_B_z = hessian_value_from_inner_products(ha, t * db, buffer_inner_Yk_eb, buffer_inner_Sk_eb, buffer_inner_cy, buffer_inner_cs)

    eb_B_d = hessian_value_from_inner_products(ha, db, buffer_inner_Yk_eb, buffer_inner_Sk_eb, buffer_inner_py, buffer_inner_ps)

    buffer_inner_py .-= db .* buffer_inner_Yk_eb
    buffer_inner_ps .-= db .* buffer_inner_Sk_eb

    return eb_B_z, eb_B_d
end

struct ProductIndex{T <: Tuple}
    ranges::T
end

Base.iterate(itr::ProductIndex) = _iterate(itr.ranges, 1, nothing)
Base.iterate(itr::ProductIndex, state) = _iterate(itr.ranges, state...)

function _iterate(ranges, i, st)
    i > length(ranges) && return nothing
    if st === nothing
        it = iterate(ranges[i])
        it === nothing && return _iterate(ranges, i + 1, nothing)
        (j, st2) = it
        return ((i, j), (i, st2))
    else
        it = iterate(ranges[i], st)
        if it === nothing
            return _iterate(ranges, i + 1, nothing)
        else
            (j, st2) = it
            return ((i, j), (i, st2))
        end
    end
end

"""
    to_coordinate_index(M::AbstractManifold, b::UnitVector{Int}, B::AbstractBasis)

Get the index of coordinate equal to 1 of [`UnitVector`](@ref) `b` with respect to
`AbstractBasis` `B`.
"""
to_coordinate_index(::AbstractManifold, b::UnitVector{Int}, ::AbstractBasis) = b.index
"""
    to_coordinate_index(M::ProductManifold, b::UnitVector{Tuple{Int, Int}}, B::AbstractBasis)

Get the index of coordinate equal to 1 of [`UnitVector`](@ref) `b` with respect to
`AbstractBasis` `B`.
"""
function to_coordinate_index(M::ProductManifold, b::UnitVector{Tuple{Int, Int}}, B::AbstractBasis)
    i, j = b.index
    offset = sum(k -> number_of_coordinates(M.manifolds[k], B), 1:(i - 1); init = 0)
    return offset + j
end

Base.length(itr::ProductIndex) = sum(length, itr.ranges)


"""
    get_bounds_index(::AbstractManifold)

Get the bound indices of manifold `M`. Standard manifolds don't have bounds, so
`Base.OneTo(0)`, that is an empty range, is returned.
"""
get_bounds_index(M::AbstractManifold) = Base.OneTo(0)
function get_bounds_index(M::ProductManifold)
    ranges = map(get_bounds_index, M.manifolds)
    iter = ProductIndex(ranges)
    return iter
end

"""
    get_stepsize_bound(M::AbstractManifold, x, d, i)

Get the upper bound on moving in direction `d` from point `p` on manifold `M`, for the
bound index `i`.
"""
get_stepsize_bound(M::AbstractManifold, p, d, i)
function get_stepsize_bound(M::ProductManifold, p, d, i::Tuple{Int, Any})
    i1, i2 = i
    return get_stepsize_bound(M.manifolds[i1], submanifold_component(M, p, i1), submanifold_component(M, d, i1), i2)
end

"""
    set_zero_at_index!(M::ProductManifold, d, i::Tuple{Int,Any})

Set the element of the `i[1]`th component of `d` at bound index `i[2]` to zero.
"""
function set_zero_at_index!(M::ProductManifold, d, i::Tuple{Int, Any})
    i1, i2 = i
    set_zero_at_index!(M.manifolds[i1], submanifold_component(M, d, i1), i2)
    return d
end

"""
    set_stepsize_bound!(M::AbstractManifold, d_out, p, d, t_current::Real)

Limit the per-component stepsize in `d_out` to the bound imposed by the box constraints.

For each component at index `i` in the tangent vector `d_out`, if the stepsize bound in
direction `d` for that component is less than `t_current`, set the element of `d_out` to
the distance from `p[i]` to the bound in the direction of `d[i]`. If the stepsize bound is
non-positive, set the element to 0.

By default it does not modify `d_out` because most manifolds don't have direction-specific
stepsize bounds, and general anisotropic bounds are handled differently.
"""
function set_stepsize_bound!(::AbstractManifold, d_out, p, d, t_current::Real)
    return d_out
end

function set_stepsize_bound!(M::ProductManifold, d_out, p, d, t_current::Real)
    map(
        (N, d_out_c, p_c, d_c) -> set_stepsize_bound!(N, d_out_c, p_c, d_c, t_current),
        M.manifolds,
        submanifold_components(M, d_out),
        submanifold_components(M, p),
        submanifold_components(M, d),
    )
    return d_out
end

@doc raw"""
    GeneralizedCauchyDirectionSubsolver{TM <: AbstractManifold, TP, T_HA <: AbstractQuasiNewtonDirectionUpdate, TFU <: AbstractSegmentHessianUpdater}

Helper container for generalized Cauchy direction search. Stores the manifold `M`, cached
original descent direction (`d_original`), the quasi-Newton direction update `ha`, and the
`hessian_segment_updater`, which computes certain values of the Hessian while advancing segments.
Instances are reused across segments during [`find_generalized_cauchy_direction!`](@ref) to
avoid allocations.
"""
struct GeneralizedCauchyDirectionSubsolver{
        TX,
        T_HA, TFU <: AbstractSegmentHessianUpdater, TFT <: Tuple{<:Real, Any}, TBI,
        TO <: Base.Order.Ordering,
    }
    d_original::TX
    ha::T_HA
    hessian_segment_updater::TFU
    F_list::Vector{TFT}
    bounds_indices::TBI
    ordering::TO
end

function GeneralizedCauchyDirectionSubsolver(
        M::AbstractManifold, p, ha;
        hessian_segment_updater::AbstractSegmentHessianUpdater = get_default_hessian_segment_updater(M, p, ha)
    )
    bounds_indices = get_bounds_index(M)
    TInd = eltype(bounds_indices)
    TF = number_eltype(p)
    F_list = Tuple{TF, TInd}[]
    sizehint!(F_list, length(bounds_indices) + 1)
    ordering = Base.By(first)
    return GeneralizedCauchyDirectionSubsolver(
        zero_vector(M, p), ha,
        hessian_segment_updater, F_list, bounds_indices, ordering
    )
end

function collect_isotropic_limits!(::AbstractManifold, ::Vector{<:Tuple{TF, Any}}, p, d)::Tuple{Bool, TF} where {TF <: Real}
    return false, convert(TF, Inf)
end

function collect_isotropic_limits!(M::ProductManifold, F_list::Vector{<:Tuple{TF, Any}}, p, d)::Tuple{Bool, TF} where {TF <: Real}
    has_finite_limit = false
    smallest_positive_limit = Inf
    map(M.manifolds, submanifold_components(M, p), submanifold_components(M, d)) do Mi, p_i, d_i
        if !has_anisotropic_max_stepsize(Mi)
            max_step = Manopt.max_stepsize(Mi, p_i)
            if isfinite(max_step)
                tms = max_step / norm(Mi, p_i, d_i)
                push!(F_list, (tms, -1))
                has_finite_limit = true
                if tms < smallest_positive_limit
                    smallest_positive_limit = tms
                end
            end
        end
    end
    return has_finite_limit, smallest_positive_limit
end

"""
    find_generalized_cauchy_direction!(
        M::AbstractManifold,
        gcd::GeneralizedCauchyDirectionSubsolver, d_out, p, d, X
    )

Find generalized Cauchy direction looking from point `p` on manifold `M` in direction `d`
and save it to `d_out`. Gradient of the objective at `p` is `X`.

The function returns a pair (status, max_stepsize) where `status` is a symbol describing
the result of the search, and `max_stepsize` is the maximum stepsize that can be taken in
the direction `d_out`.

The `status` can be one of the following:
* `:found_limited` if the point was found and we can perform a step of length at most 1
  in direction `d_out` afterwards,
* `:found_unlimited` if the point was found and we can perform a step of length at most
  `max_stepsize(M, p)` in direction `d_out` afterwards,
* `:not_found` if the search cannot be performed in direction `d`.
"""
function find_generalized_cauchy_direction!(
        M::AbstractManifold,
        gcd::GeneralizedCauchyDirectionSubsolver{
            <:Any, <:Any,
            <:AbstractSegmentHessianUpdater, <:Tuple{TF, Any},
        },
        d_out, p, d, X
    ) where {TF <: Real}
    copyto!(M, gcd.d_original, d)
    copyto!(M, d_out, d)

    ordering = gcd.ordering
    F_list = gcd.F_list
    empty!(F_list)

    bounds_indices = gcd.bounds_indices

    # isotropic limits
    has_finite_limit, smallest_positive_limit = collect_isotropic_limits!(M, F_list, p, d)
    # anisotropic limits
    for i in bounds_indices
        sbi = get_stepsize_bound(M, p, d, i)::TF

        if sbi > 0
            push!(F_list, (sbi, i))
            if sbi < smallest_positive_limit
                smallest_positive_limit = sbi
            end
        end
        has_finite_limit |= isfinite(sbi)
    end

    # In this case we can't move in the direction `d` at all, though it's usually not
    # a problem relevant to the end user because it can be handled by step_solver! that
    # uses the GCD subsolver.
    if isempty(F_list)
        return (:not_found, NaN)
    end
    heapify!(F_list, ordering)

    f_prime = inner(M, p, X, d)
    f_double_prime = hessian_value_diag(gcd.ha, M, p, d)

    if iszero(f_prime) || iszero(f_double_prime)
        return (:not_found, NaN)
    end

    dt_min = -f_prime / f_double_prime
    t_old = 0.0

    t_current, b = heappop!(F_list, ordering)
    dt = t_current - t_old

    init_updater!(M, gcd.hessian_segment_updater, p, d, gcd.ha)
    # b can be -1 if it corresponds to the max stepsize limit on the manifold part
    while dt_min > dt && b != -1
        db = get_at_bound_index(M, d, b)::TF
        gb = get_at_bound_index(M, X, b)::TF

        hv_eb_dz, hv_eb_d = gcd.hessian_segment_updater(M, p, t_current, dt, b, db, gcd.ha)::Tuple{TF, TF}

        f_prime += dt * f_double_prime - db * (gb + hv_eb_dz)
        f_double_prime += (2 * -db * hv_eb_d) + db^2 * hessian_value_diag(gcd.ha, M, p, UnitVector(b))

        t_old = t_current

        # If f_prime is 0, we've found the local minimizer (GCD)
        if iszero(f_prime) || iszero(f_double_prime)
            # It means that GCD is at the beginning of the t_current, so we want to set dt_min to 0 (stay in the point)
            dt_min = 0.0
            break
        end

        dt_min = -f_prime / f_double_prime
        isempty(F_list) && break

        t_current, b = heappop!(F_list, ordering)
        dt = t_current - t_old
    end

    dt_min = max(dt_min, 0.0)
    t_old = t_old + dt_min
    d_out .*= t_old
    # by construction, there is no bound achievable before stepsize 1.0 in direction d_out
    # there first bound after that is achieved at smallest_positive_limit / t_old
    max_feasible_stepsize = max(1.0, smallest_positive_limit / t_old)

    set_stepsize_bound!(M, d_out, p, gcd.d_original, t_old)
    if has_finite_limit
        return (:found_limited, max_feasible_stepsize)
    else
        return (:found_unlimited, Inf)
    end
end

"""
    struct MaxStepsizeInDirectionSubsolver end

Helper container for finding the maximum stepsize in a direction. Stores the list of bounds
`F_list` and the bound indices `bounds_indices`.

## Constructor

    MaxStepsizeInDirectionSubsolver(M::AbstractManifold, p)

Initialize the `MaxStepsizeInDirectionSubsolver` for manifold `M` and point `p`. The `F_list`
is initialized to be empty and will be populated during the search for the maximum stepsize
in a direction. Floating point type of the elements bounds in `F_list` is determined by the
number type of `p`.

The `MaxStepsizeInDirectionSubsolver` can be reused for multiple different points and
directions on the same manifold, but it is not thread-safe.
"""
struct MaxStepsizeInDirectionSubsolver{TFT <: Tuple{<:Real, Any}, TBI}
    F_list::Vector{TFT}
    bounds_indices::TBI
end
function MaxStepsizeInDirectionSubsolver(M::AbstractManifold, p)
    bounds_indices = get_bounds_index(M)
    TInd = eltype(bounds_indices)
    TF = number_eltype(p)
    F_list = Tuple{TF, TInd}[]
    sizehint!(F_list, length(bounds_indices) + 1)
    return MaxStepsizeInDirectionSubsolver{Tuple{TF, TInd}, typeof(bounds_indices)}(F_list, bounds_indices)
end

"""
    find_max_stepsize_in_direction(M::AbstractManifold, gcd::MaxStepsizeInDirectionSubsolver, p, d)

Find the maximum stepsize that can be performed from point `p` in direction `d`.

The function returns a pair (status, max_stepsize) where `status` is a symbol describing
the result of the search, and `max_stepsize` is the maximum stepsize that can be taken from
`p` in the direction `d`.

The `status` can be one of the following:
* `:found_limited` if a finite bound exists; `max_stepsize` is then the smallest positive
  stepsize bound,
* `:found_unlimited` if no finite bound exists; `max_stepsize` is then `Inf`,
* `:not_found` if no positive stepsize bound exists at all; `max_stepsize` is then `NaN`.
"""
function find_max_stepsize_in_direction(
        M::AbstractManifold,
        sdf::MaxStepsizeInDirectionSubsolver{<:Tuple{TF, Any}},
        p, d
    ) where {TF <: Real}

    F_list = sdf.F_list
    empty!(F_list)
    bounds_indices = sdf.bounds_indices

    # isotropic limits
    has_finite_limit, smallest_positive_limit = collect_isotropic_limits!(M, F_list, p, d)
    # anisotropic limits
    for i in bounds_indices
        sbi = get_stepsize_bound(M, p, d, i)::TF

        if sbi > 0
            push!(F_list, (sbi, i))
            if sbi < smallest_positive_limit
                smallest_positive_limit = sbi
            end
        end
        has_finite_limit |= isfinite(sbi)
    end

    if isempty(F_list)
        return (:not_found, NaN)
    end
    if has_finite_limit
        return (:found_limited, smallest_positive_limit)
    else
        return (:found_unlimited, Inf)
    end

end

function show(io::IO, qns::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    print(io, "QuasiNewtonLimitedMemoryBoxDirectionUpdate with internal state:\n")
    return print(io, qns.qn_du)
end
