@doc raw"""
    AbstractQuasiNewtonDirectionUpdate

An abstract representation of an Quasi Newton Update rule to determine the next direction
given current [`QuasiNewtonOptions`](@ref).

All subtypes should be functors, i.e. one should be able to call them as `H(M,x,d)` to compute a new direction update.
"""
abstract type AbstractQuasiNewtonDirectionUpdate end

@doc raw"""
    AbstractQuasiNewtonUpdateRule

Specify a type for the different [`AbstractQuasiNewtonDirectionUpdate`](@ref)s.
"""
abstract type AbstractQuasiNewtonUpdateRule end

@doc raw"""
    BFGS <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian BFGS update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{H}_k^\mathrm{BFGS}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
H^\mathrm{BFGS}_{k+1} = \widetilde{H}^\mathrm{BFGS}_k  + \frac{y_k y^{\mathrm{T}}_k }{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{H}^\mathrm{BFGS}_k s_k s^{\mathrm{T}}_k \widetilde{H}^\mathrm{BFGS}_k }{s^{\mathrm{T}}_k \widetilde{H}^\mathrm{BFGS}_k s_k}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonOptions`](@ref)) of

```math
T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
\operatorname{grad}f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(\operatorname{grad}f(x_k)) ∈ T_{x_{k+1}} \mathcal{M},
```

respectively.
"""
struct BFGS <: AbstractQuasiNewtonUpdateRule end
@doc raw"""
    InverseBFGS <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemanian BFGS update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{B}_k^\mathrm{BFGS}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
B^\mathrm{BFGS}_{k+1}  = \Bigl(
  \mathrm{id}_{T_{x_{k+1}} \mathcal{M}} - \frac{s_k y^{\mathrm{T}}_k }{s^{\mathrm{T}}_k y_k}
\Bigr)
\widetilde{B}^\mathrm{BFGS}_k
\Bigl(
  \mathrm{id}_{T_{x_{k+1}} \mathcal{M}} - \frac{y_k s^{\mathrm{T}}_k }{s^{\mathrm{T}}_k y_k}
\Bigr) + \frac{s_k s^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonOptions`](@ref)) of

```math
T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
\operatorname{grad}f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(\operatorname{grad}f(x_k)) ∈ T_{x_{k+1}} \mathcal{M},
```

respectively.

"""
struct InverseBFGS <: AbstractQuasiNewtonUpdateRule end
@doc raw"""
    DFP <: AbstractQuasiNewtonUpdateRule

indicates in an [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian DFP update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{H}_k^\mathrm{DFP}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
H^\mathrm{DFP}_{k+1} = \Bigl(
  \mathrm{id}_{T_{x_{k+1}} \mathcal{M}} - \frac{y_k s^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
\Bigr)
\widetilde{H}^\mathrm{DFP}_k
\Bigl(
  \mathrm{id}_{T_{x_{k+1}} \mathcal{M}} - \frac{s_k y^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
\Bigr) + \frac{y_k y^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonOptions`](@ref)) of

```math
T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
\operatorname{grad}f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(\operatorname{grad}f(x_k)) ∈ T_{x_{k+1}} \mathcal{M},
```

respectively.

"""
struct DFP <: AbstractQuasiNewtonUpdateRule end
@doc raw"""
    InverseDFP <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemanian DFP update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{B}_k^\mathrm{DFP}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
B^\mathrm{DFP}_{k+1} = \widetilde{B}^\mathrm{DFP}_k
+ \frac{s_k s^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
- \frac{\widetilde{B}^\mathrm{DFP}_k y_k y^{\mathrm{T}}_k \widetilde{B}^\mathrm{DFP}_k}{y^{\mathrm{T}}_k \widetilde{B}^\mathrm{DFP}_k y_k}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonOptions`](@ref)) of

```math
T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
\operatorname{grad}f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(\operatorname{grad}f(x_k)) ∈ T_{x_{k+1}} \mathcal{M},
```

respectively.

"""
struct InverseDFP <: AbstractQuasiNewtonUpdateRule end
@doc raw"""
    SR1 <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian SR1 update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{H}_k^\mathrm{SR1}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
H^\mathrm{SR1}_{k+1} = \widetilde{H}^\mathrm{SR1}_k
+ \frac{
  (y_k - \widetilde{H}^\mathrm{SR1}_k s_k) (y_k - \widetilde{H}^\mathrm{SR1}_k s_k)^{\mathrm{T}}
}{
(y_k - \widetilde{H}^\mathrm{SR1}_k s_k)^{\mathrm{T}} s_k
}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonOptions`](@ref)) of

```math
T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
\operatorname{grad}f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(\operatorname{grad}f(x_k)) ∈ T_{x_{k+1}} \mathcal{M},
```

respectively.


This method can be stabilized by only performing the update if denominator is larger than
``r\lVert s_k\rVert_{x_{k+1}}\lVert y_k - \widetilde{H}^\mathrm{SR1}_k s_k \rVert_{x_{k+1}}``
for some ``r>0``. For more details, see Section 6.2 in [^NocedalWright2006]

[^NocedalWright2006]:
    > Nocedal, J., Wright, S.: Numerical Optimization, Second Edition, Springer, 2006.
    > doi: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)

# Constructor
    SR1(r::Float64=-1.0)

Generate the `SR1` update, which by default does not include the check (since the default sets ``t<0```)

"""
struct SR1 <: AbstractQuasiNewtonUpdateRule
    r::Float64
    SR1(r::Float64=-1.0) = new(r)
end

@doc raw"""
    InverseSR1 <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemanian SR1 update is used in the Riemannian quasi-Newton method.


We denote by ``\widetilde{B}_k^\mathrm{SR1}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
B^\mathrm{SR1}_{k+1} = \widetilde{B}^\mathrm{SR1}_k
+ \frac{
  (s_k - \widetilde{B}^\mathrm{SR1}_k y_k) (s_k - \widetilde{B}^\mathrm{SR1}_k y_k)^{\mathrm{T}}
}{
  (s_k - \widetilde{B}^\mathrm{SR1}_k y_k)^{\mathrm{T}} y_k
}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonOptions`](@ref)) of

```math
T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
\operatorname{grad}f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(\operatorname{grad}f(x_k)) ∈ T_{x_{k+1}} \mathcal{M},
```

respectively.


This method can be stabilized by only performing the update if denominator is larger than
``r\lVert y_k\rVert_{x_{k+1}}\lVert s_k - \widetilde{H}^\mathrm{SR1}_k y_k \rVert_{x_{k+1}}``
for some ``r>0``. For more details, see Section 6.2 in [^NocedalWright2006].

# Constructor
    InverseSR1(r::Float64=-1.0)

Generate the `InverseSR1` update, which by default does not include the check,
since the default sets ``t<0```.
"""
struct InverseSR1 <: AbstractQuasiNewtonUpdateRule
    r::Float64
    InverseSR1(r::Float64=-1.0) = new(r)
end

@doc raw"""
    Broyden <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian Broyden update is used in the Riemannian quasi-Newton method, which is as a convex combination of [`BFGS`](@ref) and [`DFP`](@ref).

We denote by ``\widetilde{H}_k^\mathrm{Br}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
H^\mathrm{Br}_{k+1} = \widetilde{H}^\mathrm{Br}_k
  - \frac{\widetilde{H}^\mathrm{Br}_k s_k s^{\mathrm{T}}_k \widetilde{H}^\mathrm{Br}_k}{s^{\mathrm{T}}_k \widetilde{H}^\mathrm{Br}_k s_k} + \frac{y_k y^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
  + φ_k s^{\mathrm{T}}_k \widetilde{H}^\mathrm{Br}_k s_k
  \Bigl(
        \frac{y_k}{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{H}^\mathrm{Br}_k s_k}{s^{\mathrm{T}}_k \widetilde{H}^\mathrm{Br}_k s_k}
  \Bigr)
  \Bigl(
        \frac{y_k}{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{H}^\mathrm{Br}_k s_k}{s^{\mathrm{T}}_k \widetilde{H}^\mathrm{Br}_k s_k}
  \Bigr)^{\mathrm{T}}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonOptions`](@ref)) of

```math
T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
\operatorname{grad}f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(\operatorname{grad}f(x_k)) ∈ T_{x_{k+1}} \mathcal{M},
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

@doc raw"""
    InverseBroyden <: AbstractQuasiNewtonUpdateRule

Indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian Broyden update
is used in the Riemannian quasi-Newton method, which is as a convex combination
of [`InverseBFGS`](@ref) and [`InverseDFP`](@ref).

We denote by ``\widetilde{H}_k^\mathrm{Br}`` the operator concatenated with a vector transport
and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
B^\mathrm{Br}_{k+1} = \widetilde{B}^\mathrm{Br}_k
 - \frac{\widetilde{B}^\mathrm{Br}_k y_k y^{\mathrm{T}}_k \widetilde{B}^\mathrm{Br}_k}{y^{\mathrm{T}}_k \widetilde{B}^\mathrm{Br}_k y_k}
   + \frac{s_k s^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
 + φ_k y^{\mathrm{T}}_k \widetilde{B}^\mathrm{Br}_k y_k
 \Bigl(
     \frac{s_k}{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{B}^\mathrm{Br}_k y_k}{y^{\mathrm{T}}_k \widetilde{B}^\mathrm{Br}_k y_k}
    \Bigr) \Bigl(
        \frac{s_k}{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{B}^\mathrm{Br}_k y_k}{y^{\mathrm{T}}_k \widetilde{B}^\mathrm{Br}_k y_k}
 \Bigr)^{\mathrm{T}}
```

where ``s_k`` and ``y_k`` are the coordinate vectors with respect to the current basis (from [`QuasiNewtonOptions`](@ref)) of

```math
T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
\operatorname{grad}f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(\operatorname{grad}f(x_k)) ∈ T_{x_{k+1}} \mathcal{M},
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

@doc raw"""
    QuasiNewtonOptions <: Options

These Quasi Newton [`Options`](@ref) represent any quasi-Newton based method and can be
used with any update rule for the direction.

# Fields
* `x` – the current iterate, a point on a manifold
* `gradient` – the current gradient
* `sk` – the current step
* `yk` the current gradient difference
* `direction_update` - an [`AbstractQuasiNewtonDirectionUpdate`](@ref) rule.
* `retraction_method` – an `AbstractRetractionMethod`
* `stop` – a [`StoppingCriterion`](@ref)

# See also
[`GradientProblem`](@ref)
"""
mutable struct QuasiNewtonOptions{
    P,
    T,
    U<:AbstractQuasiNewtonDirectionUpdate,
    SC<:StoppingCriterion,
    S<:Stepsize,
    RTR<:AbstractRetractionMethod,
    VT<:AbstractVectorTransportMethod,
} <: AbstractGradientOptions
    x::P
    gradient::T
    sk::T
    yk::T
    direction_update::U
    retraction_method::RTR
    stepsize::S
    stop::SC
    vector_transport_method::VT
end
function QuasiNewtonOptions(
    x::P,
    gradient::T,
    direction_update::U,
    stop::SC,
    stepsize::S;
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
) where {P,T,U<:AbstractQuasiNewtonDirectionUpdate,SC<:StoppingCriterion,S<:Stepsize}
    return QuasiNewtonOptions{
        P,T,U,SC,S,typeof(retraction_method),typeof(vector_transport_method)
    }(
        x,
        gradient,
        deepcopy(gradient),
        deepcopy(gradient),
        direction_update,
        retraction_method,
        stepsize,
        stop,
        vector_transport_method,
    )
end

@doc raw"""
    QuasiNewtonMatrixDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

These [`AbstractQuasiNewtonDirectionUpdate`](@ref)s represent any quasi-Newton update rule, where the operator is stored as a matrix. A distinction is made between the update of the approximation of the Hessian, ``H_k \mapsto H_{k+1}``, and the update of the approximation of the Hessian inverse, ``B_k \mapsto B_{k+1}``. For the first case, the coordinates of the search direction ``η_k`` with respect to a basis ``\{b_i\}^{n}_{i=1}`` are determined by solving a linear system of equations, i.e.

```math
\text{Solve} \quad \hat{η_k} = - H_k \widehat{\operatorname{grad}f(x_k)}
```

where ``H_k`` is the matrix representing the operator with respect to the basis ``\{b_i\}^{n}_{i=1}`` and ``\widehat{\operatorname{grad}f(x_k)}`` represents the coordinates of the gradient of the objective function ``f`` in ``x_k`` with respect to the basis ``\{b_i\}^{n}_{i=1}``.
If a method is chosen where Hessian inverse is approximated, the coordinates of the search direction ``η_k`` with respect to a basis ``\{b_i\}^{n}_{i=1}`` are obtained simply by matrix-vector multiplication, i.e.

```math
\hat{η_k} = - B_k \widehat{\operatorname{grad}f(x_k)}
```

where ``B_k`` is the matrix representing the operator with respect to the basis ``\{b_i\}^{n}_{i=1}`` and ``\widehat{\operatorname{grad}f(x_k)}`` as above. In the end, the search direction ``η_k`` is generated from the coordinates ``\hat{eta_k}`` and the vectors of the basis ``\{b_i\}^{n}_{i=1}`` in both variants.
The [`AbstractQuasiNewtonUpdateRule`](@ref) indicates which quasi-Newton update rule is used. In all of them, the Euclidean update formula is used to generate the matrix ``H_{k+1}`` and ``B_{k+1}``, and the basis ``\{b_i\}^{n}_{i=1}`` is transported into the upcoming tangent space ``T_{x_{k+1}} \mathcal{M}``, preferably with an isometric vector transport, or generated there.

# Fields
* `basis` – the basis.
* `matrix` – the matrix which represents the approximating operator.
* `scale` – indicates whether the initial matrix (= identity matrix) should be scaled before the first update.
* `update` – a [`AbstractQuasiNewtonUpdateRule`](@ref).
* `vector_transport_method` – an `AbstractVectorTransportMethod`

# See also
[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)
[`QuasiNewtonCautiousDirectionUpdate`](@ref)
[`AbstractQuasiNewtonDirectionUpdate`](@ref)
"""
mutable struct QuasiNewtonMatrixDirectionUpdate{
    NT<:AbstractQuasiNewtonUpdateRule,
    B<:AbstractBasis,
    VT<:AbstractVectorTransportMethod,
    M<:AbstractMatrix,
} <: AbstractQuasiNewtonDirectionUpdate
    basis::B
    matrix::M
    scale::Bool
    update::NT
    vector_transport_method::VT
end
function QuasiNewtonMatrixDirectionUpdate(
    update::AbstractQuasiNewtonUpdateRule,
    basis::B,
    m::M,
    ;
    scale::Bool=true,
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
) where {M<:AbstractMatrix,B<:AbstractBasis}
    return QuasiNewtonMatrixDirectionUpdate{
        typeof(update),B,typeof(vector_transport_method),M
    }(
        basis, m, scale, update, vector_transport_method
    )
end
function (d::QuasiNewtonMatrixDirectionUpdate{T})(
    p, o
) where {T<:Union{InverseBFGS,InverseDFP,InverseSR1,InverseBroyden}}
    return get_vector(
        p.M, o.x, -d.matrix * get_coordinates(p.M, o.x, o.gradient, d.basis), d.basis
    )
end
function (d::QuasiNewtonMatrixDirectionUpdate{T})(
    p, o
) where {T<:Union{BFGS,DFP,SR1,Broyden}}
    return get_vector(
        p.M, o.x, -d.matrix \ get_coordinates(p.M, o.x, o.gradient, d.basis), d.basis
    )
end

@doc raw"""
    QuasiNewtonLimitedMemoryDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

This [`AbstractQuasiNewtonDirectionUpdate`](@ref) represents the limited-memory Riemanian BFGS update, where the approximating  operator is represented by ``m`` stored pairs of tangent vectors ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}`` in the ``k``-th iteration.
For the calculation of the search direction ``η_k``, the generalisation of the two-loop recursion is used (see [^HuangGallivanAbsil2015]), since it only requires inner products and linear combinations of tangent vectors in ``T_{x_k} \mathcal{M}``. For that the stored pairs of tangent vectors ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}``, the gradient ``\operatorname{grad}f(x_k)`` of the objective function ``f`` in ``x_k`` and the positive definite self-adjoint operator

```math
\mathcal{B}^{(0)}_k[⋅] = \frac{g_{x_k}(s_{k-1}, y_{k-1})}{g_{x_k}(y_{k-1}, y_{k-1})} \; \mathrm{id}_{T_{x_k} \mathcal{M}}[⋅]
```

are used. The two-loop recursion can be understood as that the [`InverseBFGS`](@ref) update is executed ``m`` times in a row on ``\mathcal{B}^{(0)}_k[⋅]`` using the tangent vectors ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}``, and in the same time the resulting operator ``\mathcal{B}^{LRBFGS}_k [⋅]`` is directly applied on ``\operatorname{grad}f(x_k)``.
When updating there are two cases: if there is still free memory, i.e. ``k < m``, the previously stored vector pairs ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}`` have to be transported into the upcoming tangent space ``T_{x_{k+1}} \mathcal{M}``; if there is no free memory, the oldest pair ``\{ \widetilde{s}_{k−m}, \widetilde{y}_{k−m}\}`` has to be discarded and then all the remaining vector pairs ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m+1}^{k-1}`` are transported into the tangent space ``T_{x_{k+1}} \mathcal{M}``. After that we calculate and store ``s_k = \widetilde{s}_k = T^{S}_{x_k, α_k η_k}(α_k η_k)`` and ``y_k = \widetilde{y}_k``. This process ensures that new information about the objective function is always included and the old, probably no longer relevant, information is discarded.

# Fields
* `method` – the maximum number of vector pairs stored.
* `memory_s` – the set of the stored (and transported) search directions times step size ``\{ \widetilde{s}_i\}_{i=k-m}^{k-1}``.
* `memory_y` – set of the stored gradient differences ``\{ \widetilde{y}_i\}_{i=k-m}^{k-1}``.
* `ξ` – a variable used in the two-loop recursion.
* `ρ` – a variable used in the two-loop recursion.
* `scale` –
* `vector_transport_method` – a `AbstractVectorTransportMethod`

# See also
[`InverseBFGS`](@ref)
[`QuasiNewtonCautiousDirectionUpdate`](@ref)
[`AbstractQuasiNewtonDirectionUpdate`](@ref)

[^HuangGallivanAbsil2015]:
    > Huang, Wen and Gallivan, K. A. and Absil, P.-A., A Broyden Class of Quasi-Newton Methods for Riemannian Optimization,
    > SIAM J. Optim., 25 (2015), pp. 1660-1685.
    > doi: [10.1137/140955483](https://doi.org/10.1137/140955483)
"""
mutable struct QuasiNewtonLimitedMemoryDirectionUpdate{
    NT<:AbstractQuasiNewtonUpdateRule,T,VT<:AbstractVectorTransportMethod
} <: AbstractQuasiNewtonDirectionUpdate
    method::NT
    memory_s::CircularBuffer{T}
    memory_y::CircularBuffer{T}
    ξ::Vector{Float64}
    ρ::Vector{Float64}
    scale::Float64
    vector_transport_method::VT
end
function QuasiNewtonLimitedMemoryDirectionUpdate(
    method::NT,
    ::T,
    memory_size::Int;
    scale::Bool=true,
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
) where {NT<:AbstractQuasiNewtonUpdateRule,T}
    return QuasiNewtonLimitedMemoryDirectionUpdate{NT,T,typeof(vector_transport_method)}(
        method,
        CircularBuffer{T}(memory_size),
        CircularBuffer{T}(memory_size),
        zeros(memory_size),
        zeros(memory_size),
        scale,
        vector_transport_method,
    )
end
function (d::QuasiNewtonLimitedMemoryDirectionUpdate{InverseBFGS})(p, o)
    r = deepcopy(o.gradient)
    m = length(d.memory_s)
    m == 0 && return -r
    for i in m:-1:1
        d.ρ[i] = 1 / inner(p.M, o.x, d.memory_s[i], d.memory_y[i]) # 1 sk 2 yk
        d.ξ[i] = inner(p.M, o.x, d.memory_s[i], r) * d.ρ[i]
        r .= r .- d.ξ[i] .* d.memory_y[i]
    end
    r .= 1 / (d.ρ[m] * norm(p.M, o.x, last(d.memory_y))^2) .* r
    for i in 1:m
        r .= r .+ (d.ξ[i] - d.ρ[i] * inner(p.M, o.x, d.memory_y[i], r)) .* d.memory_s[i]
    end
    return -project(p.M, o.x, r)
end

@doc raw"""
    QuasiNewtonCautiousDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

These [`AbstractQuasiNewtonDirectionUpdate`](@ref)s represent any quasi-Newton update rule, which are based on the idea of a so-called cautious update. The search direction is calculated as given in [`QuasiNewtonMatrixDirectionUpdate`](@ref) or [`LimitedMemoryQuasiNewctionDirectionUpdate`]. But the update given in [`QuasiNewtonMatrixDirectionUpdate`](@ref) or [`LimitedMemoryQuasiNewctionDirectionUpdate`] is only executed if

```math
\frac{g_{x_{k+1}}(y_k,s_k)}{\lVert s_k \rVert^{2}_{x_{k+1}}} \geq \theta(\lVert \operatorname{grad}f(x_k) \rVert_{x_k}),
```

is satisfied, where ``\theta`` is a monotone increasing function satisfying ``\theta(0) = 0`` and ``\theta`` is strictly increasing at ``0``. If this is not the case, the corresponding update will be skipped, which means that for [`QuasiNewtonMatrixDirectionUpdate`](@ref) the matrix ``H_k`` or ``B_k`` is not updated, but the basis ``\{b_i\}^{n}_{i=1}`` is nevertheless transported into the upcoming tangent space ``T_{x_{k+1}} \mathcal{M}``, and for [`LimitedMemoryQuasiNewctionDirectionUpdate`] neither the oldest vector pair ``\{ \widetilde{s}_{k−m}, \widetilde{y}_{k−m}\}`` is discarded nor the newest vector pair ``\{ \widetilde{s}_{k}, \widetilde{y}_{k}\}`` is added into storage, but all stored vector pairs ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}`` are transported into the tangent space ``T_{x_{k+1}} \mathcal{M}``.
If [`InverseBFGS`](@ref) or [`InverseBFGS`](@ref) is chosen as update, then the resulting method follows the method of [^HuangAbsilGallivan2018], taking into account that the corresponding step size is chosen.


# Fields
* `update` – an [`AbstractQuasiNewtonDirectionUpdate`](@ref)
* `θ` – a monotone increasing function satisfying ``θ(0) = 0`` and ``θ`` is strictly increasing at ``0``.

# See also
[`QuasiNewtonMatrixDirectionUpdate`](@ref)
[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)

[^HuangAbsilGallivan2018]:
    > Huang, Wen and Absil, P.-A and Gallivan, Kyle, A Riemannian BFGS Method Without Differentiated Retraction for Nonconvex Optimization Problems,
    > SIAM J. Optim., 28 (2018), pp. 470-495.
    > doi: [10.1137/17M1127582](https://doi.org/10.1137/17M1127582)
"""
mutable struct QuasiNewtonCautiousDirectionUpdate{U} <:
               AbstractQuasiNewtonDirectionUpdate where {
    U<:Union{
        QuasiNewtonMatrixDirectionUpdate,QuasiNewtonLimitedMemoryDirectionUpdate{T}
    },
} where {T<:AbstractQuasiNewtonUpdateRule}
    update::U
    θ::Function
end
function QuasiNewtonCautiousDirectionUpdate(
    update::U; θ::Function=x -> x
) where {
    U<:Union{
        QuasiNewtonMatrixDirectionUpdate,QuasiNewtonLimitedMemoryDirectionUpdate{T}
    },
} where {T<:AbstractQuasiNewtonUpdateRule}
    return QuasiNewtonCautiousDirectionUpdate{U}(update, θ)
end
(d::QuasiNewtonCautiousDirectionUpdate)(p, o) = d.update(p, o)

# access the inner vector transport method
function get_update_vector_transport(u::AbstractQuasiNewtonDirectionUpdate)
    return u.vector_transport_method
end
function get_update_vector_transport(u::QuasiNewtonCautiousDirectionUpdate)
    return get_update_vector_transport(u.update)
end
