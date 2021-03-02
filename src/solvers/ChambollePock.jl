@doc raw"""
    ChambollePock(M, N, cost, x0, ξ0, m, n, prox_F, prox_G_dual, forward_operator, adjoint_DΛ)

Perform the Riemannian Chambolle–Pock algorithm.

Given a `cost` function $\mathcal E\colon\mathcal M → ℝ$ of the form
```math
\mathcal E(x) = F(x) + G( Λ(x) ),
```
where $F\colon\mathcal M → ℝ$, $G\colon\mathcal N → ℝ$,
and $\Lambda\colon\mathcal M → \mathcal N$. The remaining input parameters are

* `x,ξ` primal and dual start points $x\in\mathcal M$ and $\xi∈T_n\mathcal N$
* `m,n` base points on $\mathcal M$ and $\mathcal N$, respectively.
* `forward_operator` the operator $Λ(⋅)$ or its linearization $DΛ(⋅)[⋅]$, depending on whether `:exact` or `:linearized` is chosen.
* `adjoint_linearized_operator` the adjoint $DΛ^*$ of the linearized operator $DΛ(m)\colon T_{m}\mathcal M → T_{Λ(m)}\mathcal N$
* `prox_F, prox_G_Dual` the proximal maps of $F$ and $G^\ast_n$

By default, this performs the exact Riemannian Chambolle Pock algorithm, see the opional parameter
`DΛ` for ther linearized variant.

For more details on the algorithm, see[^BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez2020].

# Optional Parameters

* `acceleration` – (`0.05`)
* `dual_stepsize` – (`1/sqrt(8)`) proximnal parameter of the primal prox
* `Λ` (`missing`) the exact operator, that is required if the forward operator is linearized;
  `missing` indicates, that the forward operator is exact.
* `primal_stepsize` – (`1/sqrt(8)`) proximnal parameter of the dual prox
* `relaxation` – (`1.`)
* `relax` – (`:primal`) whether to relax the primal or dual
* `variant` - (`:exact` if `Λ` is missing, otherwise `:linearized`) variant to use.
  Note that this changes the arguments the `forward_operator` will be called.
* `stopping_criterion` – (`stopAtIteration(100)`) a [`StoppingCriterion`](@ref)
* `update_primal_base` – (`missing`) function to update `m` (identity by default/missing)
* `update_dual_base` – (`missing`) function to update `n` (identity by default/missing)
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use
* `inverse_retraction_method` - (`LogarithmicInverseRetraction()`) an inverse retraction to use.
* `vector_transport_method` - (`ParallelTransport()`) a vector transport to use

[^BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez2020]:
    > R. Bergmann, R. Herzog, M. Silva Louzeiro, D. Tenbrinck, J. Vidal-Núñez:
    > _Fenchel Duality Theory and a Primal-Dual Algorithm on Riemannian Manifolds_,
    > arXiv: [1908.02022](http://arxiv.org/abs/1908.02022)
    > accepted for publication in Foundations of Computational Mathematics
"""
function ChambollePock(
    M::Manifold,
    N::Manifold,
    cost::Function,
    x::P,
    ξ::T,
    m::P,
    n::Q,
    prox_F::Function,
    prox_G_dual::Function,
    adjoint_linear_operator::Function;
    Λ::Union{Function,Missing}=missing,
    linearized_forward_operator::Union{Function,Missing}=missing,
    kwargs...,
) where {P,T,Q}
    x_res = allocate(x)
    copyto!(x_res, x)
    ξ_res = allocate(ξ)
    copyto!(ξ_res, ξ)
    m_res = allocate(m)
    copyto!(m_res, m)
    n_res = allocate(n)
    copyto!(n_res, n)
    return ChambollePock!(
        M,
        N,
        cost,
        x_res,
        ξ_res,
        m_res,
        n_res,
        prox_F,
        prox_G_dual,
        adjoint_linear_operator;
        Λ=Λ,
        linearized_forward_operator=linearized_forward_operator,
        kwargs...,
    )
end
@doc raw"""
    ChambollePock(M, N, cost, x, ξ, m, n, prox_F, prox_G_dual, forward_operator, adjoint_DΛ)

Perform the Riemannian Chambolle–Pock algorithm in place of `x`, `ξ`, and potenitally `m`,
`n` if they are not fixed. See [`ChambollePock`](@ref) for details and optional parameters.
"""
function ChambollePock!(
    M::Manifold,
    N::Manifold,
    cost::Function,
    x::P,
    ξ::T,
    m::P,
    n::Q,
    prox_F::Function,
    prox_G_dual::Function,
    adjoint_linear_operator::Function;
    Λ::Union{Function,Missing}=missing,
    linearized_forward_operator::Union{Function,Missing}=missing,
    acceleration=0.05,
    dual_stepsize=1 / sqrt(8),
    primal_stepsize=1 / sqrt(8),
    relaxation=1.0,
    relax::Symbol=:primal,
    stopping_criterion::StoppingCriterion=StopAfterIteration(200),
    update_primal_base::Union{Function,Missing}=missing,
    update_dual_base::Union{Function,Missing}=missing,
    retraction_method::RM=ExponentialRetraction(),
    inverse_retraction_method::IRM=LogarithmicInverseRetraction(),
    vector_transport_method::VTM=ParallelTransport(),
    variant=ismissing(Λ) ? :exact : :linearized,
    return_options=false,
    kwargs...,
) where {
    P,
    Q,
    T,
    RM<:AbstractRetractionMethod,
    IRM<:AbstractInverseRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
}
    p = PrimalDualProblem(
        M,
        N,
        cost,
        prox_F,
        prox_G_dual,
        adjoint_linear_operator;
        linearized_forward_operator=linearized_forward_operator,
        Λ=Λ,
    )
    o = ChambollePockOptions(
        m,
        n,
        x,
        ξ,
        primal_stepsize,
        dual_stepsize;
        acceleration=acceleration,
        relaxation=relaxation,
        stopping_criterion=stopping_criterion,
        relax=relax,
        update_primal_base=update_primal_base,
        update_dual_base=update_dual_base,
        variant=variant,
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
        vector_transport_method=vector_transport_method,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end

function initialize_solver!(::PrimalDualProblem, ::ChambollePockOptions) end

function step_solver!(p::PrimalDualProblem, o::ChambollePockOptions, iter)
    primal_dual_step!(p, o, Val(o.relax))
    o.m = ismissing(o.update_primal_base) ? o.m : o.update_primal_base(p, o, iter)
    if !ismissing(o.update_dual_base)
        n_old = deepcopy(o.n)
        o.n = o.update_dual_base(p, o, iter)
        vector_transport_to!(p.N, o.ξ, n_old, o.ξ, o.n, o.vector_transport_method)
        vector_transport_to!(p.N, o.ξbar, n_old, o.ξbar, o.n, o.vector_transport_method)
    end
    return o
end
#
# Variant 1: primal relax
#
function primal_dual_step!(p::PrimalDualProblem, o::ChambollePockOptions, ::Val{:primal})
    dual_update!(p, o, o.xbar, Val(o.variant))
    if ismissing(p.Λ!!)
        ptξn = o.ξ
    else
        ptξn = vector_transport_to(
            p.N, o.n, o.ξ, forward_operator(p, o.m), o.vector_transport_method
        )
    end
    xOld = o.x
    o.x = get_primal_prox!(
        p,
        o.x,
        o.m,
        o.primal_stepsize,
        retract(
            p.M,
            o.x,
            vector_transport_to(
                p.M,
                o.m,
                -o.primal_stepsize * (adjoint_linearized_operator(p, o.m, o.n, ptξn)),
                o.x,
                o.vector_transport_method,
            ),
            o.retraction_method,
        ),
    )
    update_prox_parameters!(o)
    retract!(
        p.M,
        o.xbar,
        o.x,
        -o.relaxation * inverse_retract(p.M, o.x, xOld, o.inverse_retraction_method),
        o.retraction_method,
    )
    return o
end
#
# Variant 2: dual relax
#
function primal_dual_step!(p::PrimalDualProblem, o::ChambollePockOptions, ::Val{:dual})
    if ismissing(p.Λ!!)
        ptξbar = o.ξbar
    else
        ptξbar = vector_transport_to(
            p.N, o.n, o.ξbar, forward_operator(p, o.m), o.vector_transport_method
        )
    end
    get_primal_prox!(
        p,
        o.x,
        o.m,
        o.primal_stepsize,
        retract(
            p.M,
            o.x,
            vector_transport_to(
                p.M,
                o.m,
                -o.primal_stepsize * (adjoint_linearized_operator(p, o.m, o.n, ptξbar)),
                o.x,
                o.vector_transport_method,
            ),
            o.retraction_method,
        ),
    )
    ξ_old = deepcopy(o.ξ)
    dual_update!(p, o, o.x, Val(o.variant))
    update_prox_parameters!(o)
    o.ξbar = o.ξ + o.relaxation * (o.ξ - ξ_old)
    return o
end
#
# Dual step: linearized
# depending on whether its primal relaxed or dual relaxed we start from start=o.x or start=o.xbar here
#
function dual_update!(
    p::PrimalDualProblem, o::ChambollePockOptions, start::P, ::Val{:linearized}
) where {P}
    # (1) compute update direction
    ξ_update = linearized_forward_operator(
        p, o.m, inverse_retract(p.M, o.m, start, o.inverse_retraction_method)
    )
    # (2) if p.Λ is missing, we assume that n = Λ(m) and do  not PT, otherwise we do
    (!ismissing(p.Λ!!)) && vector_transport_to!(
        p.N,
        ξ_update,
        forward_operator(p, o.m),
        ξ_update,
        o.n,
        o.vector_transport_method,
    )
    # (3) to the dual update
    get_dual_prox!(p, o.ξ, o.n, o.dual_stepsize, o.ξ + o.dual_stepsize * ξ_update)
    return o
end
#
# Dual step: exact
# depending on whether its primal relaxed or dual relaxed we start from start=o.x or start=o.xbar here
#
function dual_update!(
    p::PrimalDualProblem, o::ChambollePockOptions, start::P, ::Val{:exact}
) where {P}
    ξ_update = inverse_retract(
        p.N, o.n, forward_operator(p, start), o.inverse_retraction_method
    )
    get_dual_prox!(p, o.ξ, o.n, o.dual_stepsize, o.ξ + o.dual_stepsize * ξ_update)
    return o
end

@doc raw"""
    update_prox_parameters!(o)
update the prox parameters as described in Algorithm 2 of Chambolle, Pock, 2010, i.e.

1. ``θ_{n} = \frac{1}{\sqrt{1+2γτ_n}}``
2. ``τ_{n+1} = θ_nτ_n``
3. ``σ_{n+1} = \frac{σ_n}{θ_n}``
"""
function update_prox_parameters!(o::O) where {O<:PrimalDualOptions}
    if o.acceleration > 0
        o.relaxation = 1 / sqrt(1 + 2 * o.acceleration * o.primal_stepsize)
        o.primal_stepsize = o.primal_stepsize * o.relaxation
        o.dual_stepsize = o.dual_stepsize / o.relaxation
    end
    return o
end
