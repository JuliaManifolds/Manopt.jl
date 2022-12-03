@doc raw"""
    primal_dual_semismooth_Newton(M, N, cost, x0, ξ0, m, n, prox_F, diff_prox_F, prox_G_dual, diff_prox_dual_G, linearized_operator, adjoint_linearized_operator)

Perform the Primal-Dual Riemannian Semismooth Newton algorithm.

Given a `cost` function $\mathcal E\colon\mathcal M \to \overline{ℝ}$ of the form
```math
\mathcal E(x) = F(x) + G( Λ(x) ),
```
where $F\colon\mathcal M \to \overline{ℝ}$, $G\colon\mathcal N \to \overline{ℝ}$,
and $\Lambda\colon\mathcal M \to \mathcal N$. The remaining input parameters are

* `x,ξ` primal and dual start points $x\in\mathcal M$ and $\xi\in T_n\mathcal N$
* `m,n` base points on $\mathcal M$ and $\mathcal N$, respectively.
* `linearized_forward_operator` the linearization $DΛ(⋅)[⋅]$ of the operator $Λ(⋅)$.
* `adjoint_linearized_operator` the adjoint $DΛ^*$ of the linearized operator $DΛ(m)\colon T_{m}\mathcal M \to T_{Λ(m)}\mathcal N$
* `prox_F, prox_G_Dual` the proximal maps of $F$ and $G^\ast_n$
* `diff_prox_F, diff_prox_dual_G` the (Clarke Generalized) differentials of the proximal maps of $F$ and $G^\ast_n$

For more details on the algorithm, see[^DiepeveenLellmann2021].

# Optional Parameters

* `primal_stepsize` – (`1/sqrt(8)`) proximal parameter of the primal prox
* `Λ` (`missing`) the exact operator, that is required if `Λ(m)=n` does not hold;
`missing` indicates, that the forward operator is exact.
* `dual_stepsize` – (`1/sqrt(8)`) proximal parameter of the dual prox
* `reg_param` – (`1e-5`) regularisation parameter for the Newton matrix
Note that this changes the arguments the `forward_operator` will be called.
* `stopping_criterion` – (`stopAtIteration(50)`) a [`StoppingCriterion`](@ref)
* `update_primal_base` – (`missing`) function to update `m` (identity by default/missing)
* `update_dual_base` – (`missing`) function to update `n` (identity by default/missing)
* `retraction_method` – (`default_retraction_method(M)`) the rectraction to use
* `inverse_retraction_method` - (`default_inverse_retraction_method(M)`) an inverse retraction to use.
* `vector_transport_method` - (`default_vector_transport_method(M)`) a vector transport to use

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details

[^DiepeveenLellmann2021]:
    > W. Diepeveen, J. Lellmann:
    > _An Inexact Semismooth Newton Method on Riemannian Manifolds with Application to Duality-Based Total Variation Denoising_,
    > SIAM Journal on Imaging Sciences, 2021.
    > doi: [10.1137/21M1398513](https://doi.org/10.1137/21M1398513)
"""
function primal_dual_semismooth_Newton(
    M::AbstractManifold,
    N::AbstractManifold,
    cost::TF,
    x::P,
    ξ::T,
    m::P,
    n::Q,
    prox_F::Function,
    diff_prox_F::Function,
    prox_G_dual::Function,
    diff_prox_G_dual::Function,
    linearized_forward_operator::Function,
    adjoint_linearized_operator::Function;
    Λ::Union{Function,Missing}=missing,
    kwargs...,
) where {TF,P,T,Q}
    x_res = copy(M, x)
    ξ_res = copy(N, n, ξ)
    m_res = copy(M, m)
    n_res = copy(N, n)
    return primal_dual_semismooth_Newton!(
        M,
        N,
        cost,
        x_res,
        ξ_res,
        m_res,
        n_res,
        prox_F,
        diff_prox_F,
        prox_G_dual,
        diff_prox_G_dual,
        linearized_forward_operator,
        adjoint_linearized_operator;
        Λ=Λ,
        kwargs...,
    )
end
@doc raw"""
    primal_dual_semismooth_Newton(M, N, cost, x0, ξ0, m, n, prox_F, diff_prox_F, prox_G_dual, diff_prox_G_dual, linearized_forward_operator, adjoint_linearized_operator)

Perform the Riemannian Primal-dual Riemannian semismooth Newton algorithm in place of `x`, `ξ`, and potentially `m`,
`n` if they are not fixed. See [`primal_dual_semismooth_Newton`](@ref) for details and optional parameters.
"""
function primal_dual_semismooth_Newton!(
    M::mT,
    N::nT,
    cost::Function,
    x::P,
    ξ::T,
    m::P,
    n::Q,
    prox_F::Function,
    diff_prox_F::Function,
    prox_G_dual::Function,
    diff_prox_G_dual::Function,
    linearized_forward_operator::Function,
    adjoint_linearized_operator::Function;
    dual_stepsize=1 / sqrt(8),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    Λ::Union{Function,Missing}=missing,
    primal_stepsize=1 / sqrt(8),
    reg_param=1e-5,
    stopping_criterion::StoppingCriterion=StopAfterIteration(50),
    update_primal_base::Union{Function,Missing}=missing,
    update_dual_base::Union{Function,Missing}=missing,
    retraction_method::RM=default_retraction_method(M),
    inverse_retraction_method::IRM=default_inverse_retraction_method(M),
    vector_transport_method::VTM=default_vector_transport_method(M),
    kwargs...,
) where {
    mT<:AbstractManifold,
    nT<:AbstractManifold,
    P,
    Q,
    T,
    RM<:AbstractRetractionMethod,
    IRM<:AbstractInverseRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
}
    p = PrimalDualSemismoothNewtonProblem(
        M,
        N,
        cost,
        prox_F,
        diff_prox_F,
        prox_G_dual,
        diff_prox_G_dual,
        linearized_forward_operator,
        adjoint_linearized_operator;
        Λ=Λ,
        evaluation=evaluation,
    )
    o = PrimalDualSemismoothNewtonState(
        M,
        m,
        n,
        x,
        ξ,
        primal_stepsize,
        dual_stepsize,
        reg_param;
        stopping_criterion=stopping_criterion,
        update_primal_base=update_primal_base,
        update_dual_base=update_dual_base,
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
        vector_transport_method=vector_transport_method,
    )
    o = decorate_options(o; kwargs...)
    return get_solver_return(solve!(p, o))
end

function initialize_solver!(
    ::PrimalDualSemismoothNewtonProblem, ::PrimalDualSemismoothNewtonState
) end

function step_solver!(
    p::PrimalDualSemismoothNewtonProblem, o::PrimalDualSemismoothNewtonState, iter
)
    # do step
    primal_dual_step!(p, o)
    o.m = ismissing(o.update_primal_base) ? o.m : o.update_primal_base(p, o, iter)
    if !ismissing(o.update_dual_base)
        n_old = deepcopy(o.n)
        o.n = o.update_dual_base(p, o, iter)
        o.ξ = vector_transport_to(p.N, n_old, o.ξ, o.n, o.vector_transport_method)
    end
    return o
end

function primal_dual_step!(
    p::PrimalDualSemismoothNewtonProblem, o::PrimalDualSemismoothNewtonState
)

    # construct X
    X = construct_primal_dual_residual_vector(p, o)

    # construct matrix
    ∂X = construct_primal_dual_residual_covariant_derivative_matrix(p, o)
    ∂X += o.reg_param * sparse(I, size(∂X))  # prevent singular matrix at solution

    # solve matrix -> find coordinates
    d_coords = ∂X \ -X

    dims = manifold_dimension(p.M)
    dx_coords = d_coords[1:dims]
    dξ_coords = d_coords[(dims + 1):end]

    # compute step
    dx = get_vector(p.M, o.x, dx_coords, DefaultOrthonormalBasis())
    dξ = get_vector(p.N, o.n, dξ_coords, DefaultOrthonormalBasis())

    # do step
    o.x = retract(p.M, o.x, dx, o.retraction_method)
    return o.ξ = o.ξ + dξ
end

raw"""
    construct_primal_dual_residual_vector(p, o)

Constructs the vector representation of $X(p^{(k)}, ξ_{n}^{(k)}) \in \mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}$
"""
function construct_primal_dual_residual_vector(
    p::PrimalDualSemismoothNewtonProblem, o::PrimalDualSemismoothNewtonState
)

    # Compute primal vector
    x_update = get_primal_prox(
        p,
        # o.x,
        o.primal_stepsize,
        retract(
            p.M,
            o.x,
            vector_transport_to(
                p.M,
                o.m,
                -o.primal_stepsize * (adjoint_linearized_operator(p, o.m, o.n, o.ξ)),
                o.x,
                o.vector_transport_method,
            ),
            o.retraction_method,
        ),
    )

    primal_vector = -inverse_retract(p.M, o.x, x_update, o.inverse_retraction_method)

    X₁ = get_coordinates(p.M, o.x, primal_vector, DefaultOrthonormalBasis())

    # Compute dual vector
    # (1) compute update direction
    ξ_update = linearized_forward_operator(
        p, o.m, inverse_retract(p.M, o.m, o.x, o.inverse_retraction_method), o.n
    )
    # (2) if p.Λ is missing, we assume that n = Λ(m) and do  not PT, otherwise we do
    ξ_update = if ismissing(p.Λ!!)
        ξ_update
    else
        vector_transport_to(
            p.N, forward_operator(p, o.m), ξ_update, o.n, o.vector_transport_method
        )
    end
    # (3) to the dual update
    ξ_update = get_dual_prox(p, o.n, o.dual_stepsize, o.ξ + o.dual_stepsize * ξ_update)

    dual_vector = o.ξ - ξ_update

    X₂ = get_coordinates(p.N, o.n, dual_vector, DefaultOrthonormalBasis())

    X = [X₁; X₂]

    return X
end

raw"""
onstruct_primal_dual_residual_covariant_derivative_matrix(p, o)

Constructs the matrix representation of $V^{(k)}:\mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}\rightarrow \mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}$
"""
function construct_primal_dual_residual_covariant_derivative_matrix(
    p::PrimalDualSemismoothNewtonProblem, o::PrimalDualSemismoothNewtonState
)

    # construct bases
    Θ = get_basis(p.M, o.x, DefaultOrthonormalBasis())
    Ξ = get_basis(p.N, o.n, DefaultOrthonormalBasis())

    dims = manifold_dimension(p.M)
    dualdims = manifold_dimension(p.N)

    # we assume here that a parallel transport is already in the next operator
    qξ = -o.primal_stepsize * adjoint_linearized_operator(p, o.m, o.n, o.ξ)
    qₚ = shortest_geodesic(p.M, o.m, o.x, 1 / 2)
    qb = retract(p.M, o.m, qξ, o.retraction_method)
    q₅ = 2 * inverse_retract(p.M, qb, qₚ, o.inverse_retraction_method)
    q₄ = retract(p.M, qb, q₅, o.retraction_method)
    q₃ = -inverse_retract(p.M, o.x, q₄, o.inverse_retraction_method)
    q₂ = retract(p.M, o.x, q₃, o.retraction_method)
    q₁ = get_primal_prox(p, o.primal_stepsize, q₂)  # TODO hier gebleven met debuggen

    # (1) compute update direction
    η₁ = linearized_forward_operator(
        p, o.m, inverse_retract(p.M, o.m, o.x, o.inverse_retraction_method), o.n
    )
    # (2) if p.Λ is missing, we assume that n = Λ(m) and do  not PT, otherwise we do
    η₁ = if ismissing(p.Λ!!)
        η₁
    else
        vector_transport_to(p.N, forward_operator(p, o.m), η₁, o.n, o.vector_transport_method)
    end
    # (3) to the dual update
    η₁ = o.ξ + o.dual_stepsize * η₁
    # construct ∂X₁₁ and ∂X₂₁
    ∂X₁₁ = spzeros(dims, dims)
    ∂X₂₁ = spzeros(dualdims, dims)

    Mdims = prod(manifold_dimension(p.M))
    debug11 = false
    debug21 = true
    for j in 1:Mdims
        eⱼ = zeros(Mdims)
        eⱼ[j] = 1
        Θⱼ = get_vector(p.M, o.m, eⱼ, Θ)
        Gⱼ = differential_geodesic_endpoint(p.M, o.m, o.x, 1 / 2, Θⱼ)
        Fⱼ = 2 * differential_log_argument(p.M, qb, qₚ, Gⱼ)
        Eⱼ = differential_exp_argument(p.M, qb, q₅, Fⱼ)
        D₂ⱼ = -differential_log_argument(p.M, o.x, q₄, Eⱼ)
        D₁ⱼ = -differential_log_basepoint(p.M, o.x, q₄, Θⱼ)
        Dⱼ = D₁ⱼ + D₂ⱼ
        C₂ⱼ = differential_exp_argument(p.M, o.x, q₃, Dⱼ)

        C₁ⱼ = differential_exp_basepoint(p.M, o.x, q₃, Θⱼ)
        Cⱼ = C₁ⱼ + C₂ⱼ
        Bⱼ = get_differential_primal_prox(p, o.primal_stepsize, q₂, Cⱼ)
        A₂ⱼ = -differential_log_argument(p.M, o.x, q₁, Bⱼ)
        A₁ⱼ = -differential_log_basepoint(p.M, o.x, q₁, Θⱼ)
        Aⱼ = A₁ⱼ + A₂ⱼ

        ∂X₁₁j = get_coordinates(p.M, o.x, Aⱼ, DefaultOrthonormalBasis())
        sp_∂X₁₁j = sparsevec(∂X₁₁j)
        dropzeros!(sp_∂X₁₁j)
        ∂X₁₁[:, j] = sp_∂X₁₁j

        Mⱼ = differential_log_argument(p.M, o.m, o.x, Θⱼ)
        Kⱼ = if ismissing(p.Λ!!)
            o.dual_stepsize * linearized_forward_operator(p, o.m, Mⱼ, o.n)
        else
            o.dual_stepsize * vector_transport_to(
                p.N,
                forward_operator(p, o.m),
                linearized_forward_operator(p, o.m, Mⱼ, o.n),
                o.n,
                o.vector_transport_method,
            )
        end
        Jⱼ = get_differential_dual_prox(p, o.n, o.dual_stepsize, η₁, Kⱼ)
        ∂X₂₁j = get_coordinates(p.N, o.n, -Jⱼ, DefaultOrthonormalBasis())

        sp_∂X₂₁j = sparsevec(∂X₂₁j)
        dropzeros!(sp_∂X₂₁j)
        ∂X₂₁[:, j] = sp_∂X₂₁j
    end

    # construct ∂X₁₂ and ∂X₂₂
    ∂X₁₂ = spzeros(dims, dualdims)
    ∂X₂₂ = spzeros(dualdims, dualdims)

    Ndims = prod(manifold_dimension(p.N))
    for j in 1:Ndims
        eⱼ = zeros(Ndims)
        eⱼ[j] = 1
        Ξⱼ = get_vector(p.N, o.n, eⱼ, Ξ)
        hⱼ = -o.primal_stepsize * adjoint_linearized_operator(p, o.m, o.n, Ξⱼ) # officially ∈ T*mM, but embedded in TmM
        Hⱼ = vector_transport_to(p.M, o.m, hⱼ, o.x)
        C₂ⱼ = differential_exp_argument(p.M, o.x, q₃, Hⱼ)
        Bⱼ = get_differential_primal_prox(p, o.primal_stepsize, q₂, C₂ⱼ)
        A₂ⱼ = -differential_log_argument(p.M, o.x, q₁, Bⱼ)

        ∂X₁₂j = get_coordinates(p.M, o.m, A₂ⱼ, DefaultOrthonormalBasis())

        sp_∂X₁₂j = sparsevec(∂X₁₂j)
        dropzeros!(sp_∂X₁₂j)
        ∂X₁₂[:, j] = sp_∂X₁₂j

        Jⱼ = get_differential_dual_prox(p, o.n, o.dual_stepsize, η₁, Ξⱼ)
        Iⱼ = Ξⱼ - Jⱼ

        ∂X₂₂j = get_coordinates(p.N, o.n, Iⱼ, DefaultOrthonormalBasis())

        sp_∂X₂₂j = sparsevec(∂X₂₂j)
        dropzeros!(sp_∂X₂₂j)
        ∂X₂₂[:, j] = sp_∂X₂₂j
    end

    return [∂X₁₁ ∂X₁₂; ∂X₂₁ ∂X₂₂]
end
