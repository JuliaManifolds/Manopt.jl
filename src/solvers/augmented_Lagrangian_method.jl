@doc raw"""
    augmented_Lagrangian_method(M, F, gradF, sub_problem, sub_options, G, H, gradG, gradH)

perform the augmented Lagrangian method (ALM)[^LiuBoumal2020][^source_code]. The aim of the ALM is to find the solution of the [`ConstrainedProblem`](@ref)
```math
\begin{aligned}
\min_{x ∈\mathcal{M}} &f(x)\\
\text{subject to } &g_i(x)\leq 0 \quad ∀ i= 1, …, m,\\
\quad &h_j(x)=0 \quad ∀ j=1,…,p,
\end{aligned}
```
where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^p`` are twice continuously differentiable functions from `M` to ℝ.
For that, in every step ``k`` of the algorithm, the augmented Lagrangian function
```math
\mathcal{L}_{ρ^{(k-1)}}(x, λ^{(k-1)}, γ^{(k-1)}) = f(x) + \frac{ρ^{(k-1)}}{2} (\sum_{j=1}^p (h_j(x)+\frac{γ_j^{(k-1)}}{ρ^{(k-1)}})^2 + \sum_{i=1}^m \max\left\{0,\frac{λ_i^{(k-1)}}{ρ^{(k-1)}}+ g_i(x)\right\}^2)
```
is minimized over all ``x ∈\mathcal{M}``, where ``λ^{(k-1)}=[λ_1^{(k-1)}, …, λ_m^{(k-1)}]^T`` and ``γ^{(k-1)}=[γ_1^{(k-1)}, …, γ_p^{(k-1)}]^T`` are the current iterations of the Lagrange multipliers and ``ρ^{(k-1)}`` is the current penalty parameter.

Then, the Lagrange multipliers are updated by 
```math
γ_j^{(k)} =\operatorname{clip}_{[γ_{\min},γ_{\max}]} (γ_j^{(k-1)} + ρ^{(k-1)} h_j(x^{(k)})) \text{for all} j=1,…,p,
```
and
```math
λ_i^{(k)} =\operatorname{clip}_{[0,λ_{\max}]} (λ_i^{(k-1)} + ρ^{(k-1)} g_i(x^{(k)})) \text{for all}  i=1,…,m,
```
where ``γ_{\min} \leq γ_{\max}`` and ``λ_{\max}`` are the multiplier boundaries. 

Next, we update the accuracy tolerance ``ϵ`` by setting
```math
ϵ^{(k)}=\max\{ϵ_{\min}, θ_ϵ ϵ^{(k-1)}\},
```
where ``ϵ_{\min}`` is the lowest value ``ϵ`` is allowed to become and ``θ_ϵ ∈ (0,1)`` is constant scaling factor.

Last, we update the penalty parameter ``ρ``. For this, we define
```math
σ^{(k)}=\max_{j=1,…,p, i=1,…,m} \{\|h_j(x^{(k)})\|, \|\max_{i=1,…,m}\{g_i(x^{(k)}), -\frac{λ_i^{(k-1)}}{ρ^{(k-1)}} \}\| \}.
```
Then, we update `ρ` according to
```math
ρ^{(k)} = \begin{cases}
ρ^{(k-1)}/θ_ρ,  & \text{if } σ^{(k)}\leq θ_ρ σ^{(k-1)} ,\\
ρ^{(k-1)}, & \text{else,}
\end{cases}
```
where ``θ_ρ \in (0,1)`` is a constant scaling factor.

[^LiuBoumal2020]:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > In: Applied Mathematics & Optimization, vol 82, 949–981 (2020),
    > doi [10.1007/s00245-019-09564-3](https://doi.org/10.1007/s00245-019-09564-3)

[^source_code]:
    > original source code to the paper:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > src: [https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints](https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints)


# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF` – the gradient of the cost function

# Optional
* `G` – the inequality constraints
* `H` – the equality constraints 
* `gradG` – the gradient of the inequality constraints
* `gradH` – the gradient of the equality constraints
* `x` – initial point
* `sub_problem` – problem for the subsolver
* `sub_options` – options of the subproblem
* `max_inner_iter` – (`200`) the maximum number of iterations the subsolver should perform in each iteration 
* `num_outer_itertgn` – (`30`)
* `ϵ` – (`1e-3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `γ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the equality constraints
* `γ_min` – (`- γ_max`) a lower bound for the Lagrange multiplier belonging to the equality constraints
* `λ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the inequality constraints
* `λ` – (`ones(len(`[`get_inequality_constraints`](@ref)`(p,x))`) the Lagrange multiplier with respect to the inequality constraints
* `γ` – (`ones(len(`[`get_equality_constraints`](@ref)`(p,x))`) the Lagrange multiplier with respect to the equality constraints
* `ρ` – (`1.0`) the penalty parameter
* `τ` – (`0.8`) factor for the improvement of the evaluation of the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `θ_ϵ` – (`(ϵ_min/ϵ)^(1/num_outer_itertgn)`) the scaling factor of the accuracy tolerance
* `oldacc` – (`Inf`) evaluation of the penalty from the last iteration
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min), `[`StopWhenChangeLess`](@ref)`(1e-6)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `return_options` – (`false`) – if activated, the extended result, i.e. the complete [`Options`](@ref) are returned. This can be used to access recorded values. If set to false (default) just the optimal value `x` is returned.

# Output
* `x` – the resulting point of ALM
OR
* `options` – the options returned by the solver (see `return_options`)
"""
function augmented_Lagrangian_method(
    M::AbstractManifold,
    F::TF,
    gradF::TGF;
    G::Function=x->[], 
    H::Function=x->[],
    gradG::Function=x->[],
    gradH::Function=x->[],
    x=random_point(M), 
    sub_problem::Problem = GradientProblem(M,F,gradF),
    sub_options::Options = GradientDescentOptions(M,x),
    kwargs...,
) where {TF, TGF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return augmented_Lagrangian_method!(M, F, gradF; G=G, H=H, gradG=gradG, gradH=gradH, x=x_res, sub_problem=sub_problem, sub_options=sub_options,kwargs...)
end
@doc raw"""
    augmented_Lagrangian_method!(M, F, gradF, sub_problem, sub_options, G, H, gradG, gradH)

perform the augmented Lagrangian method (ALM)[^LiuBoumal2020][^source_code]. The aim of the ALM is to find the solution of the [`ConstrainedProblem`](@ref)
```math
\begin{aligned}
\min_{x ∈\mathcal{M}} &f(x)\\
\text{subject to } &g_i(x)\leq 0 \quad ∀ i= 1, …, m,\\
\quad &h_j(x)=0 \quad ∀ j=1,…,p,
\end{aligned}
```
where `M` is a Riemannian manifold, and ``f``, ``\{g_i\}_{i=1}^m`` and ``\{h_j\}_{j=1}^p`` are twice continuously differentiable functions from `M` to ℝ.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF` – the gradient of the cost function
## Optional 
* `G` – the inequality constraints
* `H` – the equality constraints 
* `gradG` – the gradient of the inequality constraints
* `gradH` – the gradient of the equality constraints

For more options, especially `x` for the initial point and `ρ` for the penalty parameter, see [`augmented_Lagrangian_method`](@ref).
"""
function augmented_Lagrangian_method!(
    M::AbstractManifold,
    F::TF,
    gradF::TGF;
    G::Function=x->[],
    H::Function=x->[],
    gradG::Function=x->[],
    gradH::Function=x->[],
    x=random_point(M),
    sub_problem::Problem = GradientProblem(M,F,gradF),
    sub_options::Options = GradientDescentOptions(M,x),
    max_inner_iter::Int=200,
    num_outer_itertgn::Int=30,
    ϵ::Real=1e-3, #(starting)tolgradnorm
    ϵ_min::Real=1e-6, #endingtolgradnorm
    γ_max::Real=20.0,
    γ_min::Real=-γ_max,
    λ_max::Real=20.0,
    λ::Vector=ones(length(G(M,x))),
    γ::Vector=ones(length(H(M,x))),
    ρ::Real=1.0, 
    τ::Real=0.8,
    θ_ρ::Real=0.3, 
    θ_ϵ::Real=(ϵ_min/ϵ)^(1/num_outer_itertgn), 
    oldacc::Real=Inf, 
    stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(1e-6)), 
    return_options=false,
    kwargs...,
) where {TF, TGF}
    p = ConstrainedProblem(M, F, gradF, G, gradG, H, gradH)
    o = ALMOptions(
        M,
        p,
        x,
        sub_problem,
        sub_options;
        max_inner_iter = max_inner_iter,
        num_outer_itertgn = num_outer_itertgn,
        ϵ = ϵ,
        ϵ_min = ϵ_min,
        γ_max = γ_max,
        γ_min = γ_min,
        λ_max = λ_max,
        λ = λ,
        γ = γ,
        ρ = ρ,
        τ = τ,
        θ_ρ = θ_ρ,
        stopping_criterion = stopping_criterion,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end

#
# Solver functions
#
function initialize_solver!(p::ConstrainedProblem, o::ALMOptions)
    o.θ_ϵ = (o.ϵ_min/o.ϵ)^(1/o.num_outer_itertgn)
    o.old_acc = Inf
    return o
end
function step_solver!(p::ConstrainedProblem, o::ALMOptions, iter)
    # use subsolver to minimize the augmented Lagrangian within a tolerance ϵ and with max_inner_iter
    #cost = get_Lagrangian_cost_function(p, o) 
    #cost = (M,x) -> get_Lagrangian_cost_function(x, p, o)
    cost = LagrangeCost(p.cost, p.G, p.H, o.ρ, o.λ, o.γ)
    grad = LagrangeGrad(p.cost, p.gradF, p.G, p.gradG, p.H, p.gradH, o.ρ, o.λ, o.γ)
    
    # o.x = gradient_descent(p.M, cost, grad, o.x, stepsize=ArmijoLinesearch(), stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.ϵ)))
    
    ### Sphere
    ## ArmijoLinesearch
    # o.x = quasi_Newton(p.M, cost, grad, o.x, stepsize=ArmijoLinesearch(1.0,ExponentialRetraction(), 0.95, 0.1, 1e-20), stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.ϵ),StopWhenStepSizeLess(1e-16)))
    ## WolfePowellLinesearch
    o.x = quasi_Newton(p.M, cost, grad, o.x, stepsize=WolfePowellLinesearch(p.M,10^(-4),0.999,retraction_method=ExponentialRetraction(), vector_transport_method=ParallelTransport(), linesearch_stopsize=1e-10), stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.ϵ),StopWhenStepSizeLess(1e-16)))
    
    ### Stiefel 
    ## ArmijoLinesearch
    #o.x = quasi_Newton(p.M, cost, grad, o.x, retraction_method=QRRetraction(), vector_transport_method=ProjectionTransport(), stepsize=ArmijoLinesearch(1.0,QRRetraction(), 0.95, 0.1, 1e-20), stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.ϵ),StopWhenStepSizeLess(1e-16)))
    ## WolfePowellLinesearch
    #o.x = quasi_Newton(p.M, cost, grad, o.x, retraction_method=QRRetraction(), vector_transport_method=ProjectionTransport(), stepsize=WolfePowellLinesearch(QRRetraction(),ProjectionTransport(),10^(-4),0.999), stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.ϵ),StopWhenStepSizeLess(1e-16)))
    # o.x = quasi_Newton(p.M, cost, grad, o.x, retraction_method=QRRetraction(), vector_transport_method=ProjectionTransport(), stepsize=WolfePowellLinesearch(p.M,10^(-4),0.999,retraction_method=QRRetraction(),vector_transport_method=ProjectionTransport(),linesearch_stopsize=1e-20), stopping_criterion=StopWhenAny(StopAfterIteration(o.max_inner_iter),StopWhenGradientNormLess(o.ϵ),StopWhenStepSizeLess(1e-16)))


    
    # update multipliers
    cost_ineq = get_inequality_constraints(p, o.x)
    n_ineq_constraint = size(cost_ineq,1)
    o.λ = convert(Vector{Float64},min.(ones(n_ineq_constraint).* o.λ_max, max.(o.λ + o.ρ .* cost_ineq, zeros(n_ineq_constraint))))
    cost_eq = get_equality_constraints(p, o.x)
    n_eq_constraint = size(cost_eq,1)
    o.γ = convert(Vector{Float64},min.(ones(n_eq_constraint).* o.γ_max , max.(ones(n_eq_constraint) .* o.γ_min, o.γ + o.ρ .* cost_eq)))

    # get new evaluation of penalty
    new_acc = max(maximum(abs.(max.(-o.λ./o.ρ, cost_ineq)),init=0), maximum(abs.(cost_eq),init=0))

    # update ρ if necessary
    if iter == 1 || new_acc > o.τ * o.old_acc 
        o.ρ = o.ρ/o.θ_ρ 
    end
    o.old_acc = new_acc

    # update the tolerance ϵ
    o.ϵ = max(o.ϵ_min, o.ϵ * o.θ_ϵ)
end
get_solver_result(o::ALMOptions) = o.x

mutable struct LagrangeCost{F,G,H,R,T}
    f::F
    g::G
    h::H
    ρ::R
    λ::T
    γ::T
end
function LagrangeCost(f::F,g::G,h::H,ρ::R,λ::T,γ::T) where {F,G,H,R,T}
    return LagrangeCost{F,G,H,R,T}(f,g,h,ρ,λ,γ)
end 
function (L::LagrangeCost)(M::AbstractManifold,x::P) where {P}
    inequality_constraints = L.g(M,x)
    equality_constraints = L.h(M,x)
    num_inequality_constraints = size(inequality_constraints,1)
    num_equality_constraints = size(equality_constraints,1)
    if num_inequality_constraints != 0 
        cost_ineq = sum(max.(zeros(num_inequality_constraints), L.λ ./ L.ρ .+ inequality_constraints).^2)
    end
    if num_equality_constraints != 0
        cost_eq = sum((equality_constraints .+ L.γ./L.ρ).^2)
    end
    if num_inequality_constraints != 0
        if num_equality_constraints != 0
            return L.f(M,x) + (L.ρ/2) * (cost_ineq + cost_eq)
        else
            return L.f(M,x) + (L.ρ/2) * cost_ineq
        end
    else
        if num_equality_constraints != 0
            return L.f(M,x) + (L.ρ/2) * cost_eq
        else
            return L.f(M,x) 
        end
    end
end

mutable struct LagrangeGrad{F,GF,G,GG,H,GH,R,T}
    f::F
    gradF::GF
    g::G
    gradG::GG
    h::H
    gradH::GH
    ρ::R
    λ::T
    γ::T
end
function LagrangeGrad(f::F,gradF::GF,g::G,gradG::GG,h::H,gradH::GH,ρ::R,λ::T,γ::T) where {F,GF,G,GG,H,GH,R,T}
    return LagrangeGrad{F,GF,G,GG,H,GH,R,T}(f,gradF,g,gradG,h,gradH,ρ,λ,γ)
end 
function (LG::LagrangeGrad)(M::AbstractManifold,x::P) where {P}
    inequality_constraints = LG.g(M,x)
    equality_constraints = LG.h(M,x)
    num_inequality_constraints = size(inequality_constraints,1)
    num_equality_constraints = size(equality_constraints,1)
    if num_inequality_constraints != 0 
        grad_ineq = sum(
            ((inequality_constraints .* LG.ρ .+ LG.λ) .* LG.gradG(M,x)).*(inequality_constraints .+ LG.λ./LG.ρ .>0)
            )
    end
    if num_equality_constraints != 0
        grad_eq = sum((equality_constraints .* LG.ρ .+ LG.γ) .* LG.gradH(M,x))
    end
    if num_inequality_constraints != 0
        if num_equality_constraints != 0
            return LG.gradF(M,x) + grad_ineq + grad_eq
        else
            return LG.gradF(M,x) + grad_ineq
        end
    else
        if num_equality_constraints != 0
            return LG.gradF(M,x) + grad_eq
        else
            return LG.gradF(M,x) 
        end
    end
end
