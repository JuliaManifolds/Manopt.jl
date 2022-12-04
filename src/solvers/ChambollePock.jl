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
    o = ChambollePockState(
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
    o = decorate_state(o; kwargs...)
    return get_solver_return(solve!(p, o))
end

function initialize_solver!(::PrimalDualProblem, ::ChambollePockState) end

function step_solver!(p::PrimalDualProblem, s::ChambollePockState, iter)
    primal_dual_step!(p, s, Val(s.relax))
    s.m = ismissing(s.update_primal_base) ? s.m : s.update_primal_base(p, s, iter)
    if !ismissing(s.update_dual_base)
        n_old = deepcopy(s.n)
        s.n = s.update_dual_base(p, s, iter)
        vector_transport_to!(p.N, s.ξ, n_old, s.ξ, s.n, s.vector_transport_method_dual)
        vector_transport_to!(
            p.N, s.ξbar, n_old, s.ξbar, s.n, s.vector_transport_method_dual
        )
    end
    return s
end
#
# Variant 1: primal relax
#
function primal_dual_step!(p::PrimalDualProblem, s::ChambollePockState, ::Val{:primal})
    dual_update!(p, s, s.xbar, Val(s.variant))
    if ismissing(p.Λ!!)
        ptξn = s.ξ
    else
        ptξn = vector_transport_to(
            p.N, s.n, s.ξ, forward_operator(p, s.m), s.vector_transport_method_dual
        )
    end
    xOld = s.x
    s.x = get_primal_prox!(
        p,
        s.x,
        s.primal_stepsize,
        retract(
            p.M,
            s.x,
            vector_transport_to(
                p.M,
                s.m,
                -s.primal_stepsize * (adjoint_linearized_operator(p, s.m, s.n, ptξn)),
                s.x,
                s.vector_transport_method,
            ),
            s.retraction_method,
        ),
    )
    update_prox_parameters!(s)
    retract!(
        p.M,
        s.xbar,
        s.x,
        -s.relaxation * inverse_retract(p.M, s.x, xOld, s.inverse_retraction_method),
        s.retraction_method,
    )
    return s
end
#
# Variant 2: dual relax
#
function primal_dual_step!(p::PrimalDualProblem, s::ChambollePockState, ::Val{:dual})
    if ismissing(p.Λ!!)
        ptξbar = s.ξbar
    else
        ptξbar = vector_transport_to(
            p.N, s.n, s.ξbar, forward_operator(p, s.m), s.vector_transport_method_dual
        )
    end
    get_primal_prox!(
        p,
        s.x,
        s.primal_stepsize,
        retract(
            p.M,
            s.x,
            vector_transport_to(
                p.M,
                s.m,
                -s.primal_stepsize * (adjoint_linearized_operator(p, s.m, s.n, ptξbar)),
                s.x,
                s.vector_transport_method,
            ),
            s.retraction_method,
        ),
    )
    ξ_old = deepcopy(s.ξ)
    dual_update!(p, s, s.x, Val(s.variant))
    update_prox_parameters!(s)
    s.ξbar = s.ξ + s.relaxation * (s.ξ - ξ_old)
    return s
end
#
# Dual step: linearized
# depending on whether its primal relaxed or dual relaxed we start from start=o.x or start=o.xbar here
#
function dual_update!(
    p::PrimalDualProblem, s::ChambollePockState, start::P, ::Val{:linearized}
) where {P}
    # (1) compute update direction
    ξ_update = linearized_forward_operator(
        p, s.m, inverse_retract(p.M, s.m, start, s.inverse_retraction_method), s.n
    )
    # (2) if p.Λ is missing, we assume that n = Λ(m) and do  not PT, otherwise we do
    (!ismissing(p.Λ!!)) && vector_transport_to!(
        p.N,
        ξ_update,
        forward_operator(p, s.m),
        ξ_update,
        s.n,
        s.vector_transport_method_dual,
    )
    # (3) to the dual update
    get_dual_prox!(p, s.ξ, s.n, s.dual_stepsize, s.ξ + s.dual_stepsize * ξ_update)
    return s
end
#
# Dual step: exact
# depending on whether its primal relaxed or dual relaxed we start from start=o.x or start=o.xbar here
#
function dual_update!(
    p::PrimalDualProblem, s::ChambollePockState, start::P, ::Val{:exact}
) where {P}
    ξ_update = inverse_retract(
        p.N, s.n, forward_operator(p, start), s.inverse_retraction_method_dual
    )
    get_dual_prox!(p, s.ξ, s.n, s.dual_stepsize, s.ξ + s.dual_stepsize * ξ_update)
    return s
end

@doc raw"""
    update_prox_parameters!(o)
update the prox parameters as described in Algorithm 2 of Chambolle, Pock, 2010, i.e.

1. ``θ_{n} = \frac{1}{\sqrt{1+2γτ_n}}``
2. ``τ_{n+1} = θ_nτ_n``
3. ``σ_{n+1} = \frac{σ_n}{θ_n}``
"""
function update_prox_parameters!(s::O) where {O<:AbstractPrimalDualSolverState}
    if s.acceleration > 0
        s.relaxation = 1 / sqrt(1 + 2 * s.acceleration * s.primal_stepsize)
        s.primal_stepsize = s.primal_stepsize * s.relaxation
        s.dual_stepsize = s.dual_stepsize / s.relaxation
    end
    return s
end
