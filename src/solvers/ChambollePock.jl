@doc raw"""
    ChambollePockState <: AbstractPrimalDualSolverState

stores all options and variables within a linearized or exact Chambolle Pock.
The following list provides the order for the constructor, where the previous iterates are
initialized automatically and values with a default may be left out.

* `m` - base point on ``\mathcal M``
* `n` - base point on ``\mathcal N``
* `p` - an initial point on ``x^{(0)} ∈\mathcal M`` (and its previous iterate)
* `X` - an initial tangent vector ``X^{(0)}∈T^*\mathcal N`` (and its previous iterate)
* `pbar` - the relaxed iterate used in the next dual update step (when using `:primal` relaxation)
* `Xbar` - the relaxed iterate used in the next primal update step (when using `:dual` relaxation)
* `primal_stepsize` – (`1/sqrt(8)`) proximal parameter of the primal prox
* `dual_stepsize` – (`1/sqrt(8)`) proximal parameter of the dual prox
* `acceleration` – (`0.`) acceleration factor due to Chambolle & Pock
* `relaxation` – (`1.`) relaxation in the primal relaxation step (to compute `pbar`)
* `relax` – (`:primal`) which variable to relax (`:primal` or `:dual`)
* `stop` - a [`StoppingCriterion`](@ref)
* `variant` – (`exact`) whether to perform an `:exact` or `:linearized` Chambolle-Pock
* `update_primal_base` (`(p,o,i) -> o.m`) function to update the primal base
* `update_dual_base` (`(p,o,i) -> o.n`) function to update the dual base
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) the retraction to use
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) an inverse
  retraction to use on the manifold ``\mathcal M``.
* `inverse_retraction_method_dual` - (`default_inverse_retraction_method(N, typeof(n))`)
  an inverse retraction to use on manifold ``\mathcal N``.
* `vector_transport_method` - (`default_vector_transport_method(M, typeof(p))`) a vector transport to
  use on the manifold ``\mathcal M``.
* `vector_transport_method_dual` - (`default_vector_transport_method(N, typeof(n))`) a
  vector transport to use on manifold ``\mathcal N``.

where for the last two the functions a [`AbstractManoptProblem`](@ref)` p`,
[`AbstractManoptSolverState`](@ref)` o` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing` in the linearized case).

# Constructor
    ChambollePockState(M::AbstractManifold,
        m::P, n::Q, p::P, X::T, primal_stepsize::Float64, dual_stepsize::Float64;
        kwargs...
    )
where all other fields from above are keyword arguments with their default values given in brackets,
as well as `N=TangentBundle(M)`
"""
mutable struct ChambollePockState{
    P,
    Q,
    T,
    RM<:AbstractRetractionMethod,
    IRM<:AbstractInverseRetractionMethod,
    IRM_Dual<:AbstractInverseRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
    VTM_Dual<:AbstractVectorTransportMethod,
} <: AbstractPrimalDualSolverState
    m::P
    n::Q
    p::P
    pbar::P
    X::T
    Xbar::T
    primal_stepsize::Float64
    dual_stepsize::Float64
    acceleration::Float64
    relaxation::Float64
    relax::Symbol
    stop::StoppingCriterion
    variant::Symbol
    update_primal_base::Union{Function,Missing}
    update_dual_base::Union{Function,Missing}
    retraction_method::RM
    inverse_retraction_method::IRM
    inverse_retraction_method_dual::IRM_Dual
    vector_transport_method::VTM
    vector_transport_method_dual::VTM_Dual

    function ChambollePockState(
        M::AbstractManifold,
        m::P,
        n::Q,
        p::P,
        X::T;
        N=TangentBundle(M),
        primal_stepsize::Float64=1 / sqrt(8),
        dual_stepsize::Float64=1 / sqrt(8),
        acceleration::Float64=0.0,
        relaxation::Float64=1.0,
        relax::Symbol=:primal,
        stopping_criterion::StoppingCriterion=StopAfterIteration(300),
        variant::Symbol=:exact,
        update_primal_base::Union{Function,Missing}=missing,
        update_dual_base::Union{Function,Missing}=missing,
        retraction_method::RM=default_retraction_method(M, typeof(p)),
        inverse_retraction_method::IRM=default_inverse_retraction_method(M, typeof(p)),
        inverse_retraction_method_dual::IRM_Dual=default_inverse_retraction_method(
            N, typeof(p)
        ),
        vector_transport_method::VTM=default_vector_transport_method(M, typeof(n)),
        vector_transport_method_dual::VTM_Dual=default_vector_transport_method(
            N, typeof(n)
        ),
    ) where {
        P,
        Q,
        T,
        RM<:AbstractRetractionMethod,
        IRM<:AbstractInverseRetractionMethod,
        IRM_Dual<:AbstractInverseRetractionMethod,
        VTM<:AbstractVectorTransportMethod,
        VTM_Dual<:AbstractVectorTransportMethod,
    }
        return new{P,Q,T,RM,IRM,IRM_Dual,VTM,VTM_Dual}(
            m,
            n,
            p,
            copy(M, p),
            X,
            copy(N, X),
            primal_stepsize,
            dual_stepsize,
            acceleration,
            relaxation,
            relax,
            stopping_criterion,
            variant,
            update_primal_base,
            update_dual_base,
            retraction_method,
            inverse_retraction_method,
            inverse_retraction_method_dual,
            vector_transport_method,
            vector_transport_method_dual,
        )
    end
end
function show(io::IO, cps::ChambollePockState)
    i = get_count(cps, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(cps.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Chambolle-Pock Algorithm
    $Iter
    ## Parameters
    * primal_stepsize:  $(cps.primal_stepsize)
    * dual_stepsize:    $(cps.dual_stepsize)
    * acceleration:     $(cps.acceleration)
    * relaxation:       $(cps.relaxation)
    * relax:            $(cps.relax)
    * variant:          :$(cps.variant)
    * retraction_method:              $(cps.retraction_method)
    * inverse_retraction_method:      $(cps.inverse_retraction_method)
    * vector_transport_method:        $(cps.vector_transport_method)
    * inverse_retraction_method_dual: $(cps.inverse_retraction_method_dual)
    * vector_transport_method_dual:   $(cps.vector_transport_method_dual)

    ## Stopping Criterion
    $(status_summary(cps.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
get_solver_result(apds::AbstractPrimalDualSolverState) = get_iterate(apds)
get_iterate(apds::AbstractPrimalDualSolverState) = apds.p
function set_iterate!(apds::AbstractPrimalDualSolverState, p)
    apds.p = p
    return apds
end

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
\mathcal E(p) = F(p) + G( Λ(p) ),
```
where $F:\mathcal M → ℝ$, $G:\mathcal N → ℝ$,
and $Λ:\mathcal M → \mathcal N$. The remaining input parameters are

* `p, X` primal and dual start points $x∈\mathcal M$ and $ξ∈T_n\mathcal N$
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
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) the rectraction to use
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction to use.
* `vector_transport_method` - (`default_vector_transport_method(M, typeof(p))`) a vector transport to use

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details

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
    p::P,
    X::T,
    m::P,
    n::Q,
    prox_F::Function,
    prox_G_dual::Function,
    adjoint_linear_operator::Function;
    Λ::Union{Function,Missing}=missing,
    linearized_forward_operator::Union{Function,Missing}=missing,
    kwargs...,
) where {TF,P,T,Q}
    q = copy(M, p)
    Y = copy(N, n, X)
    m2 = copy(M, m)
    n2 = copy(N, n)
    return ChambollePock!(
        M,
        N,
        cost,
        q,
        Y,
        m2,
        n2,
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
    p::P,
    X::T,
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
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    inverse_retraction_method::IRM=default_inverse_retraction_method(M, typeof(p)),
    vector_transport_method::VTM=default_vector_transport_method(M, typeof(p)),
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
    pdmo = PrimalDualManifoldObjective(
        cost,
        prox_F,
        prox_G_dual,
        adjoint_linear_operator;
        linearized_forward_operator=linearized_forward_operator,
        Λ=Λ,
    )
    dpdmo = decorate_objective!(M, pdmo; kwargs...)
    tmp = TwoManifoldProblem(M, N, dpdmo)
    o = ChambollePockState(
        M,
        m,
        n,
        p,
        X;
        N=N,
        primal_stepsize=primal_stepsize,
        dual_stepsize=dual_stepsize,
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
    o = decorate_state!(o; kwargs...)
    return get_solver_return(solve!(tmp, o))
end

function initialize_solver!(::TwoManifoldProblem, ::ChambollePockState) end

function step_solver!(tmp::TwoManifoldProblem, cps::ChambollePockState, iter)
    N = get_manifold(tmp, 2)
    primal_dual_step!(tmp, cps, Val(cps.relax))
    cps.m =
        ismissing(cps.update_primal_base) ? cps.m : cps.update_primal_base(tmp, cps, iter)
    if !ismissing(cps.update_dual_base)
        n_old = deepcopy(cps.n)
        cps.n = cps.update_dual_base(tmp, cps, iter)
        vector_transport_to!(
            N, cps.X, n_old, cps.X, cps.n, cps.vector_transport_method_dual
        )
        vector_transport_to!(
            N, cps.Xbar, n_old, cps.Xbar, cps.n, cps.vector_transport_method_dual
        )
    end
    return cps
end
#
# Variant 1: primal relax
#
function primal_dual_step!(tmp::TwoManifoldProblem, cps::ChambollePockState, ::Val{:primal})
    dual_update!(tmp, cps, cps.pbar, Val(cps.variant))
    obj = get_objective(tmp)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    if !hasproperty(obj, :Λ!!) || ismissing(obj.Λ!!)
        ptXn = cps.X
    else
        ptXn = vector_transport_to(
            N, cps.n, cps.X, forward_operator(tmp, cps.m), cps.vector_transport_method_dual
        )
    end
    p_old = cps.p
    cps.p = get_primal_prox!(
        tmp,
        cps.p,
        cps.primal_stepsize,
        retract(
            M,
            cps.p,
            vector_transport_to(
                M,
                cps.m,
                -cps.primal_stepsize *
                (adjoint_linearized_operator(tmp, cps.m, cps.n, ptXn)),
                cps.p,
                cps.vector_transport_method,
            ),
            cps.retraction_method,
        ),
    )
    update_prox_parameters!(cps)
    retract!(
        M,
        cps.pbar,
        cps.p,
        -cps.relaxation * inverse_retract(M, cps.p, p_old, cps.inverse_retraction_method),
        cps.retraction_method,
    )
    return cps
end
#
# Variant 2: dual relax
#
function primal_dual_step!(tmp::TwoManifoldProblem, cps::ChambollePockState, ::Val{:dual})
    obj = get_objective(tmp)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    if !hasproperty(obj, :Λ!!) || ismissing(obj.Λ!!)
        ptXbar = cps.Xbar
    else
        ptXbar = vector_transport_to(
            N,
            cps.n,
            cps.Xbar,
            forward_operator(tmp, cps.m),
            cps.vector_transport_method_dual,
        )
    end
    get_primal_prox!(
        tmp,
        cps.p,
        cps.primal_stepsize,
        retract(
            M,
            cps.p,
            vector_transport_to(
                M,
                cps.m,
                -cps.primal_stepsize *
                (adjoint_linearized_operator(tmp, cps.m, cps.n, ptXbar)),
                cps.p,
                cps.vector_transport_method,
            ),
            cps.retraction_method,
        ),
    )
    X_old = deepcopy(cps.X)
    dual_update!(tmp, cps, cps.p, Val(cps.variant))
    update_prox_parameters!(cps)
    cps.Xbar = cps.X + cps.relaxation * (cps.X - X_old)
    return cps
end
#
# Dual step: linearized
# depending on whether its primal relaxed or dual relaxed we start from start=o.x or start=o.xbar here
#
function dual_update!(
    tmp::TwoManifoldProblem, cps::ChambollePockState, start::P, ::Val{:linearized}
) where {P}
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    obj = get_objective(tmp)
    # (1) compute update direction
    X_update = linearized_forward_operator(
        tmp, cps.m, inverse_retract(M, cps.m, start, cps.inverse_retraction_method), cps.n
    )
    # (2) if p.Λ is missing, we assume that n = Λ(m) and do  not PT, otherwise we do
    (hasproperty(obj, :Λ!!) && !ismissing(obj.Λ!!)) && vector_transport_to!(
        N,
        X_update,
        forward_operator(tmp, cps.m),
        X_update,
        cps.n,
        cps.vector_transport_method_dual,
    )
    # (3) to the dual update
    get_dual_prox!(
        tmp, cps.X, cps.n, cps.dual_stepsize, cps.X + cps.dual_stepsize * X_update
    )
    return cps
end
#
# Dual step: exact
# depending on whether its primal relaxed or dual relaxed we start from start=o.x or start=o.xbar here
#
function dual_update!(
    tmp::TwoManifoldProblem, cps::ChambollePockState, start::P, ::Val{:exact}
) where {P}
    N = get_manifold(tmp, 2)
    ξ_update = inverse_retract(
        N, cps.n, forward_operator(tmp, start), cps.inverse_retraction_method_dual
    )
    get_dual_prox!(
        tmp, cps.X, cps.n, cps.dual_stepsize, cps.X + cps.dual_stepsize * ξ_update
    )
    return cps
end

@doc raw"""
    update_prox_parameters!(o)
update the prox parameters as described in Algorithm 2 of Chambolle, Pock, 2010, i.e.

1. ``θ_{n} = \frac{1}{\sqrt{1+2γτ_n}}``
2. ``τ_{n+1} = θ_nτ_n``
3. ``σ_{n+1} = \frac{σ_n}{θ_n}``
"""
function update_prox_parameters!(pds::S) where {S<:AbstractPrimalDualSolverState}
    if pds.acceleration > 0
        pds.relaxation = 1 / sqrt(1 + 2 * pds.acceleration * pds.primal_stepsize)
        pds.primal_stepsize = pds.primal_stepsize * pds.relaxation
        pds.dual_stepsize = pds.dual_stepsize / pds.relaxation
    end
    return pds
end
