@doc raw"""
    ChambollePock(
        M, N, cost, x0, ξ0, m, n, prox_F, prox_G_dual, adjoint_linear_operator;
        forward_operator=missing,
        linearized_forward_operator=missing,
        evaluation=AllocatingEvaluation()
    )

Perform the Riemannian Chambolle–Pock algorithm.

Given a `cost` function $\mathcal E:\mathcal M → ℝ$ of the form
```math
\mathcal E(x) = F(x) + G( Λ(x) ),
```
where $F:\mathcal M → ℝ$, $G:\mathcal N → ℝ$,
and $Λ:\mathcal M → \mathcal N$. The remaining input parameters are

* `x,ξ` primal and dual start points $x∈\mathcal M$ and $ξ∈T_n\mathcal N$
* `m,n` base points on $\mathcal M$ and $\mathcal N$, respectively.
* `adjoint_linearized_operator` the adjoint $DΛ^*$ of the linearized operator $DΛ(m): T_{m}\mathcal M → T_{Λ(m)}\mathcal N$
* `prox_F, prox_G_Dual` the proximal maps of $F$ and $G^\ast_n$

note that depending on the [`AbstractEvaluationType`](@ref) `evaluation` the last three parameters
as well as the forward_operator `Λ` and the `linearized_forward_operator` can be given as
allocating functions `(Manifolds, parameters) -> result`  or as mutating functions
`(Manifold, result, parameters)` -> result` to spare allocations.

By default, this performs the exact Riemannian Chambolle Pock algorithm, see the optional parameter
`DΛ` for their linearized variant.

For more details on the algorithm, see[^BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez2020].

# Optional Parameters

* `acceleration` – (`0.05`)
* `dual_stepsize` – (`1/sqrt(8)`) proximal parameter of the primal prox
* `evaluation` ([`AllocatingEvaluation`](@ref)`()) specify whether the proximal maps and operators are
  allocating functions `(Manifolds, parameters) -> result`  or given as mutating functions
  `(Manifold, result, parameters)` -> result` to spare allocations.
* `Λ` (`missing`) the (forward) operator $Λ(⋅)$ (required for the `:exact` variant)
* `linearized_forward_operator` (`missing`) its linearization $DΛ(⋅)[⋅]$ (required for the `:linearized` variant)
* `primal_stepsize` – (`1/sqrt(8)`) proximal parameter of the dual prox
* `relaxation` – (`1.`)
* `relax` – (`:primal`) whether to relax the primal or dual
* `variant` - (`:exact` if `Λ` is missing, otherwise `:linearized`) variant to use.
  Note that this changes the arguments the `forward_operator` will be called.
* `stopping_criterion` – (`stopAtIteration(100)`) a [`StoppingCriterion`](@ref)
* `update_primal_base` – (`missing`) function to update `m` (identity by default/missing)
* `update_dual_base` – (`missing`) function to update `n` (identity by default/missing)
* `retraction_method` – (`default_retraction_method(M)`) the rectraction to use
* `inverse_retraction_method` - (`default_inverse_retraction_method(M)`) an inverse retraction to use.
* `vector_transport_method` - (`default_vector_transport_method(M)`) a vector transport to use

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details

[^BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez2020]:
    > R. Bergmann, R. Herzog, M. Silva Louzeiro, D. Tenbrinck, J. Vidal-Núñez:
    > _Fenchel Duality Theory and a Primal-Dual Algorithm on Riemannian Manifolds_,
    > Foundations of Computational Mathematics, 2021.
    > doi: [10.1007/s10208-020-09486-5](http://dx.doi.org/10.1007/s10208-020-09486-5)
    > arXiv: [1908.02022](http://arxiv.org/abs/1908.02022)
"""
function ChambollePock(
    M::AbstractManifold,
    N::AbstractManifold,
    cost::TF,
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
) where {TF,P,T,Q}
    x_res = copy(M, x)
    ξ_res = copy(N, n, ξ)
    m_res = copy(M, m)
    n_res = copy(N, n)
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
    ChambollePock(M, N, cost, x0, ξ0, m, n, prox_F, prox_G_dual, adjoint_linear_operator)

Perform the Riemannian Chambolle–Pock algorithm in place of `x`, `ξ`, and potentially `m`,
`n` if they are not fixed. See [`ChambollePock`](@ref) for details and optional parameters.
"""
function ChambollePock!(
    M::AbstractManifold,
    N::AbstractManifold,
    cost::TF,
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
    retraction_method::RM=default_retraction_method(M),
    inverse_retraction_method::IRM=default_inverse_retraction_method(M),
    vector_transport_method::VTM=default_vector_transport_method(M),
    variant=ismissing(Λ) ? :exact : :linearized,
    kwargs...,
) where {
    TF,
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
        M,
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
    return get_solver_return(solve!(p, o))
end

function initialize_solver!(::PrimalDualProblem, ::ChambollePockOptions) end

function step_solver!(p::PrimalDualProblem, o::ChambollePockOptions, iter)
    primal_dual_step!(p, o, Val(o.relax))
    o.m = ismissing(o.update_primal_base) ? o.m : o.update_primal_base(p, o, iter)
    if !ismissing(o.update_dual_base)
        n_old = deepcopy(o.n)
        o.n = o.update_dual_base(p, o, iter)
        vector_transport_to!(p.N, o.ξ, n_old, o.ξ, o.n, o.vector_transport_method_dual)
        vector_transport_to!(
            p.N, o.ξbar, n_old, o.ξbar, o.n, o.vector_transport_method_dual
        )
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
            p.N, o.n, o.ξ, forward_operator(p, o.m), o.vector_transport_method_dual
        )
    end
    xOld = o.x
    o.x = get_primal_prox!(
        p,
        o.x,
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
            p.N, o.n, o.ξbar, forward_operator(p, o.m), o.vector_transport_method_dual
        )
    end
    get_primal_prox!(
        p,
        o.x,
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
        p, o.m, inverse_retract(p.M, o.m, start, o.inverse_retraction_method), o.n
    )
    # (2) if p.Λ is missing, we assume that n = Λ(m) and do  not PT, otherwise we do
    (!ismissing(p.Λ!!)) && vector_transport_to!(
        p.N,
        ξ_update,
        forward_operator(p, o.m),
        ξ_update,
        o.n,
        o.vector_transport_method_dual,
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
        p.N, o.n, forward_operator(p, start), o.inverse_retraction_method_dual
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
