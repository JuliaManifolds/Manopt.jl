_doc_PDSN_formula = raw"""
Given a `cost` function ``\mathcal E: \mathcal M → \overline{ℝ}`` of the form
```math
\mathcal E(p) = F(p) + G( Λ(p) ),
```
where ``F: \mathcal M → \overline{ℝ}``, ``G: \mathcal N → \overline{ℝ}``,
and ``Λ: \mathcal M → \mathcal N``. The remaining input parameters are
"""

_doc_PDSN = """
    primal_dual_semismooth_Newton(M, N, cost, p, X, m, n, prox_F, diff_prox_F, prox_G_dual, diff_prox_dual_G, linearized_operator, adjoint_linearized_operator)

Perform the Primal-Dual Riemannian semismooth Newton algorithm.

$(_doc_PDSN_formula)

* `p, X`:                          primal and dual start points ``p∈$(_math(:M))`` and ``X ∈ T_n$(_tex(:Cal, "N"))``
* `m,n`:                           base points on ``$(_math(:M))`` and ``$(_tex(:Cal, "N"))`, respectively.
* `linearized_forward_operator`:   the linearization ``DΛ(⋅)[⋅]`` of the operator ``Λ(⋅)``.
* `adjoint_linearized_operator`:   the adjoint ``DΛ^*`` of the linearized operator ``DΛ(m):  $(_math(:TpM; p = "m")) → $(_math(:TpM; M = "N", p = "Λ(m)"))``
* `prox_F, prox_G_Dual`:           the proximal maps of ``F`` and ``G^$(_tex(:ast))_n``
* `diff_prox_F, diff_prox_dual_G`: the (Clarke Generalized) differentials of the proximal maps of ``F`` and ``G^$(_tex(:ast))_n``

For more details on the algorithm, see [DiepeveenLellmann:2021](@cite).

# Keyword arguments

* `dual_stepsize=1/sqrt(8)`: proximal parameter of the dual prox
$(_var(:Keyword, :evaluation))
$(_var(:Keyword, :inverse_retraction_method))
* `Λ=missing`: the exact operator, that is required if `Λ(m)=n` does not hold;
  `missing` indicates, that the forward operator is exact.
* `primal_stepsize=1/sqrt(8)`: proximal parameter of the primal prox
* `reg_param=1e-5`: regularisation parameter for the Newton matrix
  Note that this changes the arguments the `forward_operator` is called.
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(50)`"))
* `update_primal_base=missing`: function to update `m` (identity by default/missing)
* `update_dual_base=missing`: function to update `n` (identity by default/missing)
$(_var(:Keyword, :vector_transport_method))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_PDSN)"
function primal_dual_semismooth_Newton(
        M::AbstractManifold,
        N::AbstractManifold,
        cost::TF,
        p::P,
        X::T,
        m::P,
        n::Q,
        prox_F::Function,
        diff_prox_F::Function,
        prox_G_dual::Function,
        diff_prox_G_dual::Function,
        linearized_forward_operator::Function,
        adjoint_linearized_operator::Function;
        Λ::Union{Function, Missing} = missing,
        kwargs...,
    ) where {TF, P, T, Q}
    x_res = copy(M, p)
    ξ_res = copy(N, n, X)
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
        Λ = Λ,
        kwargs...,
    )
end

@doc "$(_doc_PDSN)"
function primal_dual_semismooth_Newton!(
        M::mT,
        N::nT,
        cost::Function,
        p::P,
        X::T,
        m::P,
        n::Q,
        prox_F::Function,
        diff_prox_F::Function,
        prox_G_dual::Function,
        diff_prox_G_dual::Function,
        linearized_forward_operator::Function,
        adjoint_linearized_operator::Function;
        dual_stepsize = 1 / sqrt(8),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        Λ::Union{Function, Missing} = missing,
        primal_stepsize = 1 / sqrt(8),
        reg_param = 1.0e-5,
        stopping_criterion::StoppingCriterion = StopAfterIteration(50),
        update_primal_base::Union{Function, Missing} = missing,
        update_dual_base::Union{Function, Missing} = missing,
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        kwargs...,
    ) where {
        mT <: AbstractManifold,
        nT <: AbstractManifold,
        P,
        Q,
        T,
        RM <: AbstractRetractionMethod,
        IRM <: AbstractInverseRetractionMethod,
        VTM <: AbstractVectorTransportMethod,
    }
    pdmsno = PrimalDualManifoldSemismoothNewtonObjective(
        cost,
        prox_F,
        diff_prox_F,
        prox_G_dual,
        diff_prox_G_dual,
        linearized_forward_operator,
        adjoint_linearized_operator;
        Λ = Λ,
        evaluation = evaluation,
    )
    dpdmsno = decorate_objective!(M, pdmsno; kwargs...)
    tmp = TwoManifoldProblem(M, N, dpdmsno)
    pdsn = PrimalDualSemismoothNewtonState(
        M;
        m = m,
        n = n,
        p = p,
        X = X,
        primal_stepsize = primal_stepsize,
        dual_stepsize = dual_stepsize,
        regularization_parameter = reg_param,
        stopping_criterion = stopping_criterion,
        update_primal_base = update_primal_base,
        update_dual_base = update_dual_base,
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
        vector_transport_method = vector_transport_method,
    )
    dpdsn = decorate_state!(pdsn; kwargs...)
    solve!(tmp, dpdsn)
    return get_solver_return(get_objective(tmp), dpdsn)
end

function initialize_solver!(::TwoManifoldProblem, ::PrimalDualSemismoothNewtonState) end

function step_solver!(tmp::TwoManifoldProblem, pdsn::PrimalDualSemismoothNewtonState, iter)
    N = get_manifold(tmp, 2)
    # do step
    primal_dual_step!(tmp, pdsn)
    pdsn.m = if ismissing(pdsn.update_primal_base)
        pdsn.m
    else
        pdsn.update_primal_base(tmp, pdsn, iter)
    end
    if !ismissing(pdsn.update_dual_base)
        n_old = deepcopy(pdsn.n)
        pdsn.n = pdsn.update_dual_base(tmp, pdsn, iter)
        pdsn.X = vector_transport_to(N, n_old, pdsn.X, pdsn.n, pdsn.vector_transport_method)
    end
    return pdsn
end

function primal_dual_step!(tmp::TwoManifoldProblem, pdsn::PrimalDualSemismoothNewtonState)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    # construct X
    X = construct_primal_dual_residual_vector(tmp, pdsn)

    # construct matrix
    ∂X = construct_primal_dual_residual_covariant_derivative_matrix(tmp, pdsn)
    ∂X += pdsn.regularization_parameter * sparse(I, size(∂X))  # prevent singular matrix at solution

    # solve matrix -> find coordinates
    d_coords = ∂X \ -X

    dims = manifold_dimension(M)
    dx_coords = d_coords[1:dims]
    dξ_coords = d_coords[(dims + 1):end]

    # compute step
    dx = get_vector(M, pdsn.p, dx_coords, DefaultOrthonormalBasis())
    dξ = get_vector(N, pdsn.n, dξ_coords, DefaultOrthonormalBasis())

    # do step
    pdsn.p = retract(M, pdsn.p, dx, pdsn.retraction_method)
    return pdsn.X = pdsn.X + dξ
end

raw"""
    construct_primal_dual_residual_vector(p, o)

Constructs the vector representation of ``X(p^{(k)}, ξ_{n}^{(k)}) ∈ \mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}``
"""
function construct_primal_dual_residual_vector(
        tmp::TwoManifoldProblem, pdsn::PrimalDualSemismoothNewtonState
    )
    obj = get_objective(tmp)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    # Compute primal vector
    p_update = get_primal_prox(
        tmp,
        pdsn.primal_stepsize,
        retract(
            M,
            pdsn.p,
            vector_transport_to(
                M,
                pdsn.m,
                -pdsn.primal_stepsize *
                    (adjoint_linearized_operator(tmp, pdsn.m, pdsn.n, pdsn.X)),
                pdsn.p,
                pdsn.vector_transport_method,
            ),
            pdsn.retraction_method,
        ),
    )

    primal_vector = -inverse_retract(M, pdsn.p, p_update, pdsn.inverse_retraction_method)

    X₁ = get_coordinates(M, pdsn.p, primal_vector, DefaultOrthonormalBasis())

    # Compute dual vector
    # (1) compute update direction
    ξ_update = linearized_forward_operator(
        tmp,
        pdsn.m,
        inverse_retract(M, pdsn.m, pdsn.p, pdsn.inverse_retraction_method),
        pdsn.n,
    )
    # (2) if p.Λ is missing, assume that n = Λ(m) and do not PT
    ξ_update = if !hasproperty(obj, :Λ!!) || ismissing(obj.Λ!!)
        ξ_update
    else
        vector_transport_to(
            N,
            forward_operator(tmp, pdsn.m),
            ξ_update,
            pdsn.n,
            pdsn.vector_transport_method,
        )
    end
    # (3) the dual update
    ξ_update = get_dual_prox(
        tmp, pdsn.n, pdsn.dual_stepsize, pdsn.X + pdsn.dual_stepsize * ξ_update
    )
    dual_vector = pdsn.X - ξ_update
    X₂ = get_coordinates(N, pdsn.n, dual_vector, DefaultOrthonormalBasis())
    return [X₁; X₂]
end

raw"""
onstruct_primal_dual_residual_covariant_derivative_matrix(p, o)

Constructs the matrix representation of ``V^{(k)}:\mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}\rightarrow \mathcal{T}_{p^{(k)}} \mathcal{M} \times \mathcal{T}_{n}^{*} \mathcal{N}``
"""
function construct_primal_dual_residual_covariant_derivative_matrix(
        tmp::TwoManifoldProblem, pdsn::PrimalDualSemismoothNewtonState
    )
    obj = get_objective(tmp)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    # construct bases
    Θ = get_basis(M, pdsn.p, DefaultOrthonormalBasis())
    Ξ = get_basis(N, pdsn.n, DefaultOrthonormalBasis())

    dims = manifold_dimension(M)
    dualdims = manifold_dimension(N)

    # assume here that a parallel transport is already in the next operator
    qξ = -pdsn.primal_stepsize * adjoint_linearized_operator(tmp, pdsn.m, pdsn.n, pdsn.X)
    qₚ = shortest_geodesic(M, pdsn.m, pdsn.p, 1 / 2)
    qb = retract(M, pdsn.m, qξ, pdsn.retraction_method)
    q₅ = 2 * inverse_retract(M, qb, qₚ, pdsn.inverse_retraction_method)
    q₄ = retract(M, qb, q₅, pdsn.retraction_method)
    q₃ = -inverse_retract(M, pdsn.p, q₄, pdsn.inverse_retraction_method)
    q₂ = retract(M, pdsn.p, q₃, pdsn.retraction_method)
    q₁ = get_primal_prox(tmp, pdsn.primal_stepsize, q₂)

    # (1) compute update direction
    η₁ = linearized_forward_operator(
        tmp,
        pdsn.m,
        inverse_retract(M, pdsn.m, pdsn.p, pdsn.inverse_retraction_method),
        pdsn.n,
    )
    # (2) if p.Λ is missing, assume that n = Λ(m) and do  not PT
    η₁ = if !hasproperty(obj, :Λ!!) || ismissing(obj.Λ!!)
        η₁
    else
        vector_transport_to(
            N, forward_operator(tmp, pdsn.m), η₁, pdsn.n, pdsn.vector_transport_method
        )
    end
    # (3) to the dual update
    η₁ = pdsn.X + pdsn.dual_stepsize * η₁
    # construct ∂X₁₁ and ∂X₂₁
    ∂X₁₁ = spzeros(dims, dims)
    ∂X₂₁ = spzeros(dualdims, dims)

    Mdims = prod(manifold_dimension(M))
    for j in 1:Mdims
        eⱼ = zeros(Mdims)
        eⱼ[j] = 1
        Θⱼ = get_vector(M, pdsn.m, eⱼ, Θ)
        Gⱼ = differential_shortest_geodesic_endpoint(M, pdsn.m, pdsn.p, 1 / 2, Θⱼ)
        Fⱼ = 2 * differential_log_argument(M, qb, qₚ, Gⱼ)
        Eⱼ = differential_exp_argument(M, qb, q₅, Fⱼ)
        D₂ⱼ = -differential_log_argument(M, pdsn.p, q₄, Eⱼ)
        D₁ⱼ = -differential_log_basepoint(M, pdsn.p, q₄, Θⱼ)
        Dⱼ = D₁ⱼ + D₂ⱼ
        C₂ⱼ = differential_exp_argument(M, pdsn.p, q₃, Dⱼ)

        C₁ⱼ = differential_exp_basepoint(M, pdsn.p, q₃, Θⱼ)
        Cⱼ = C₁ⱼ + C₂ⱼ
        Bⱼ = get_differential_primal_prox(tmp, pdsn.primal_stepsize, q₂, Cⱼ)
        A₂ⱼ = -differential_log_argument(M, pdsn.p, q₁, Bⱼ)
        A₁ⱼ = -differential_log_basepoint(M, pdsn.p, q₁, Θⱼ)
        Aⱼ = A₁ⱼ + A₂ⱼ

        ∂X₁₁j = get_coordinates(M, pdsn.p, Aⱼ, DefaultOrthonormalBasis())
        sp_∂X₁₁j = sparsevec(∂X₁₁j)
        dropzeros!(sp_∂X₁₁j)
        ∂X₁₁[:, j] = sp_∂X₁₁j

        Mⱼ = differential_log_argument(M, pdsn.m, pdsn.p, Θⱼ)
        Kⱼ = if !hasproperty(obj, :Λ!!) || ismissing(obj.Λ!!)
            pdsn.dual_stepsize * linearized_forward_operator(tmp, pdsn.m, Mⱼ, pdsn.n)
        else
            pdsn.dual_stepsize * vector_transport_to(
                N,
                forward_operator(tmp, pdsn.m),
                linearized_forward_operator(tmp, pdsn.m, Mⱼ, pdsn.n),
                pdsn.n,
                pdsn.vector_transport_method,
            )
        end
        Jⱼ = get_differential_dual_prox(tmp, pdsn.n, pdsn.dual_stepsize, η₁, Kⱼ)
        ∂X₂₁j = get_coordinates(N, pdsn.n, -Jⱼ, DefaultOrthonormalBasis())

        sp_∂X₂₁j = sparsevec(∂X₂₁j)
        dropzeros!(sp_∂X₂₁j)
        ∂X₂₁[:, j] = sp_∂X₂₁j
    end

    # construct ∂X₁₂ and ∂X₂₂
    ∂X₁₂ = spzeros(dims, dualdims)
    ∂X₂₂ = spzeros(dualdims, dualdims)

    Ndims = prod(manifold_dimension(N))
    for j in 1:Ndims
        eⱼ = zeros(Ndims)
        eⱼ[j] = 1
        Ξⱼ = get_vector(N, pdsn.n, eⱼ, Ξ)
        hⱼ = -pdsn.primal_stepsize * adjoint_linearized_operator(tmp, pdsn.m, pdsn.n, Ξⱼ) # officially ∈ T*mM, but embedded in TmM
        Hⱼ = vector_transport_to(M, pdsn.m, hⱼ, pdsn.p)
        C₂ⱼ = differential_exp_argument(M, pdsn.p, q₃, Hⱼ)
        Bⱼ = get_differential_primal_prox(tmp, pdsn.primal_stepsize, q₂, C₂ⱼ)
        A₂ⱼ = -differential_log_argument(M, pdsn.p, q₁, Bⱼ)

        ∂X₁₂j = get_coordinates(M, pdsn.m, A₂ⱼ, DefaultOrthonormalBasis())

        sp_∂X₁₂j = sparsevec(∂X₁₂j)
        dropzeros!(sp_∂X₁₂j)
        ∂X₁₂[:, j] = sp_∂X₁₂j

        Jⱼ = get_differential_dual_prox(tmp, pdsn.n, pdsn.dual_stepsize, η₁, Ξⱼ)
        Iⱼ = Ξⱼ - Jⱼ

        ∂X₂₂j = get_coordinates(N, pdsn.n, Iⱼ, DefaultOrthonormalBasis())

        sp_∂X₂₂j = sparsevec(∂X₂₂j)
        dropzeros!(sp_∂X₂₂j)
        ∂X₂₂[:, j] = sp_∂X₂₂j
    end

    return [∂X₁₁ ∂X₁₂; ∂X₂₁ ∂X₂₂]
end
