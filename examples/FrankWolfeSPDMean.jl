### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ e2dd0dd2-0444-11ed-1e55-5b3126a23f62
begin
    using Pkg
    Pkg.activate()#use branch in dev mode for this case until FW is a registered Manopt version.
    using Manopt, Manifolds, LinearAlgebra, Plots, Random
end

# ╔═╡ 285f33c1-e8ed-45d4-ba33-9e5eb17ce422
md""" # Constraint Mean using Frank–Wolfe

This example illustrates the use of the Riemannian Frank-Wolfe algorithm by reimplementing the example given in [^WeberSra2022], Section 4.2, that is the geometric mean (in Alg. 3, the GM variant).
"""

# ╔═╡ c355af21-9912-4de2-9a3e-21c3dd7d2fdd
md"""
We first define some variables.
* `n` is the dimension of our SPD matrices
* `N` is the number of data matrices we will generate
"""

# ╔═╡ 397f4aa4-bd8b-4f98-9c5a-c3514e50ef37
begin
    Random.seed!(42)
    # dimension of the SPD matrices
    n = 50
    # number of data points
    N = 100
end

# ╔═╡ 8e59bb69-e073-40cc-a304-d1256463dcbb
md"""
We now generate the $N data points and $N random weights that sum up to 1.
"""

# ╔═╡ 1a96dee3-f2b2-42a4-ac5a-2af399e86463
@doc raw"""
	harmonic_mean(pts,w=1/N.*ones(N))

for `N` points `pts` on the SPD matrices, this computes the weighted harmonic mean, i.e.

```math
	\biggl( \sum_{i=1}^N w_i p_i^{-1} \biggr)^{-1}
```
"""
function harmonic_mean(pts, w=1 / length(pts) .* ones(length(pts)))
    return inv(sum([wi * inv(pi) for (wi, pi) in zip(w, pts)]))
end

# ╔═╡ a56c69d9-768b-4ce5-99d7-986820b8b53c
@doc raw"""
	arithmetic_mean(pts,w=1/N.*ones(N))

for `N` points `pts` on the SPD matrices, this computes the weighted arithmetic mean, i.e.

```math
	\sum_{i=1}^N w_i p_i
```
"""
function arithmetic_mean(pts, w=1 / length(pts) .* ones(length(pts)))
    return sum([wi * pi for (wi, pi) in zip(w, pts)])
end

# ╔═╡ 58594aa5-1dc8-4193-914f-98ab4bfcdc03
M = SymmetricPositiveDefinite(n)

# ╔═╡ 09e304f8-b153-4076-81d7-a3a01644d65a
begin
    data = [random_point(M) for _ in 1:N]
    weights = rand(N)
    weights ./= sum(weights)
end

# ╔═╡ b2fdb060-31c6-4d2c-9e04-e5fa0caab5d0
function weighted_mean_cost(M, p)
    return sum([wi * distance(M, di, p)^2 for (wi, di) in zip(weights, data)])
end

# ╔═╡ e5fc5216-5aab-4638-9444-02dd9b1cb4e3
function grad_weighted_mean(M, p)
    q = SPDPoint(p)
    return sum([wi * grad_distance(M, di, q) for (wi, di) in zip(weights, data)])
end

# ╔═╡ 6c9c3984-2de8-4f4e-b8e9-e747059043cf
function grad_weighted_mean!(M, X, p)
    q = SPDPoint(p)
    zero_vector!(M, X, p)
    Y = zero_vector(M, p)
    for (wi, di) in zip(weights, data)
        grad_distance!(M, Y, di, q)
        X .+= wi .* Y
    end
    return X
end

# ╔═╡ 701ace87-ef6e-42a6-9e81-563c3abc55b4
@doc raw"""
	FW_oracle!(M::SymmetricPositiveDefinite, q, L, U, p, X)

Given a lower bound `L` and an upper bound `U` (spd matrices),
a point `p` and a tangent vector (e.g. the gradient at p),
this oracle solves the subproblem related to the constraint problem

```math
	\operatorname{arg\,min}_{L\preceq q \preceq U} ⟨ X, \log_p q⟩
```
which has a closed form solution, cf. (38) in [^WeberSra2022] computed in place of `q`
"""
function FW_oracle!(M::SymmetricPositiveDefinite, q, L, U, p, X)
    (p_sqrt, p_sqrt_inv) = Manifolds._sqrt_and_sqrt_inv(p)

    e2 = eigen(p_sqrt * X * p_sqrt)
    D = Diagonal(1.0 .* (e2.values .< 0))
    Q = e2.vectors

    Uprime = Q' * p_sqrt_inv * U * p_sqrt_inv * Q
    Lprime = Q' * p_sqrt_inv * L * p_sqrt_inv * Q
    P = cholesky(Hermitian(Uprime - Lprime))
    z = P.U' * D * P.U + Lprime
    copyto!(M, q, p_sqrt * Q * z * Q' * p_sqrt)
    return q
end

# ╔═╡ 11130019-505c-4557-933a-ab034d6b5b7b
function FW_oracle!(M::SymmetricPositiveDefinite, q::SPDPoint, L, U, p, X)
    (p_sqrt, p_sqrt_inv) = Manifolds._sqrt_and_sqrt_inv(p)

    e2 = eigen(p_sqrt * X * p_sqrt)
    D = Diagonal(1.0 .* (e2.values .< 0))
    Q = e2.vectors

    Uprime = Q' * p_sqrt_inv * U * p_sqrt_inv * Q
    Lprime = Q' * p_sqrt_inv * L * p_sqrt_inv * Q
    P = cholesky(Hermitian(Uprime - Lprime))
    z = P.U' * D * P.U + Lprime
    Q = p_sqrt * Q * z * Q' * p_sqrt
    !ismissing(q.p) && copyto!(q.p, Q)
    q.eigen .= eigen(Q)
    if !is_missing(q.sqrt) && !ismissing(q.sqrt_inv)
        copyto!.([q.sqrt, q.sqrt_inv], _sqrt_and_sqrt_inv(Q))
    else
        !ismissing(q.sqrt) && copyto!(q.sqrt, _sqrt(Q))
        !ismissing(q.sqrt_inv) && copyto!(q.sqrt_inv, _sqrt_inv(Q))
    end
    return q
end

# ╔═╡ 41ad71e7-708f-42e9-a92b-902c6324215f
H = harmonic_mean(data, weights);

# ╔═╡ d77613d4-ede8-44c7-bc3f-46ab4c828b90
A = arithmetic_mean(data, weights);

# ╔═╡ 17ec6a97-d7af-4b35-b388-523537e88a0f
special_oracle!(M, q, p, X) = FW_oracle!(M, q, H, A, p, X)

# ╔═╡ 43d14777-70dc-4e76-a481-4c5409f03115
statsM = @timed qT = mean(M, data, weights);

# ╔═╡ e6d19915-c495-4693-bdda-f83dc6d5ddfc
cT = weighted_mean_cost(M, qT)

# ╔═╡ 3d369862-dc1c-4e06-acde-9eb1baa1a425
statsF = @timed oF = Frank_Wolfe_algorithm(
    M,
    weighted_mean_cost,
    grad_weighted_mean!,
    data[1];
    subtask=special_oracle!,
    debug=[
        :Iteration,
        :Cost,
        (:Change, " | Change: %1.5e | "),
        DebugGradientNorm(; format=" | grad F |: %1.5e |"),
        "\n",
        :Stop,
        50,
    ],
    record=[:Iteration, :Iterate, :Cost],
    evaluation=MutatingEvaluation(),
    return_options=true,
);

# ╔═╡ cd22d1df-260b-44a8-9a0a-d349f3f588e1
qF = get_solver_result(oF);

# ╔═╡ b0974627-ae5e-432e-99b9-10b3b7463615
cF = weighted_mean_cost(M, qF)

# ╔═╡ 9c883e38-bc39-418c-8414-ba36656329f0
q1 = copy(M, data[1]);

# ╔═╡ c34152c0-c12c-4a8e-838e-5f867647cd19
statsF20 = @timed Frank_Wolfe_algorithm!(
    M,
    weighted_mean_cost,
    grad_weighted_mean!,
    q1;
    subtask=special_oracle!,
    evaluation=MutatingEvaluation(),
    stopping_criterion=StopAfterIteration(20),
);

# ╔═╡ 1a14a0de-89a3-4099-9f7b-e438df5c47bc
c1 = weighted_mean_cost(M, q1)

# ╔═╡ c0ac2089-bd95-496f-85fc-3dee85958811
oG = gradient_descent(
    M,
    weighted_mean_cost,
    grad_weighted_mean!,
    data[1];
    record=[:Iteration, :Iterate, :Cost],
    debug=[
        :Iteration,
        :Cost,
        (:Change, " | Change: %1.5e | "),
        DebugGradientNorm(; format=" | grad F |: %1.5e |"),
        "\n",
        :Stop,
        1,
    ],
    evaluation=MutatingEvaluation(),
    stopping_criterion=StopAfterIteration(200) | StopWhenGradientNormLess(1e-12),
    return_options=true,
);

# ╔═╡ 63f5166e-4755-4892-a7f5-f7be48f2fc52
q2 = copy(M, data[1]);

# ╔═╡ 9d0f4fc6-3f54-404a-ab26-050a5e52c458
statsG = @timed gradient_descent!(
    M,
    weighted_mean_cost,
    grad_weighted_mean!,
    q2;
    evaluation=MutatingEvaluation(),
    stopping_criterion=StopAfterIteration(200) | StopWhenGradientNormLess(1e-12),
);

# ╔═╡ 460a7a9d-e652-44f0-b731-a4cd03d901ea
cG = weighted_mean_cost(M, q2)

# ╔═╡ 95f34a6e-4932-44bd-9caa-2fcded798d05
md"""
We get the following results in the cost

| Method | Cost | Computational time
| ------ | :---- | :--- |
| `mean` (`Manifolds.jl`) | $cT  | $(statsM.time) sec. |
| `FrankWolfe` | $cF | $(statsF.time) sec. |
| `gradient_descent` | $cG | $(statsG.time) sec. |

And since we recorded the values in the first runs, that were not timed, we can also plot how the cost evolves over time. Note that gradient descent already finishes after 7 iterations, while we shortened Frank Wolfe to only the first 8 iterations.
"""

# ╔═╡ b76baa92-1f47-4e69-a2f7-29f539300cbb
begin
    fig = plot(
        [0, get_record(oF, :Iteration, :Iteration)[1:8]...],
        [weighted_mean_cost(M, data[1]), get_record(oF, :Iteration, :Cost)[1:8]...];
        label="Frank Wolfe",
    )
    plot!(
        fig,
        [0, get_record(oG, :Iteration, :Iteration)...],
        [weighted_mean_cost(M, data[1]), get_record(oG, :Iteration, :Cost)...];
        label="Gradient Descent",
    )
end

# ╔═╡ 9aa04bf2-3265-4317-8bf8-db05feb338f9
md"""
A challenge seems to be to find a good stopping criterion for Frank Wolfe on manifolds, since after iteration 10 the cost only changes in order `1e-4`, and the iterates do as well, which is above the current threshold used for most other algorithms.

It also seems gradient descent outperforms Frank Wolfe in the speed it reaches the final cost when using `ArmijoLinesearch`, which was not used as an example in the paper [^WeberSra2022].
"""

# ╔═╡ a86b5add-3ec8-4fd9-ba62-e6ab89f0dbda
md"""
## Literature

[^WeberSra2022]:
    > M. Weber, S. Sra: _Riemannian Optimization via Frank-Wolfe Methods_,
    > Math. Prog., 2022, to appear.
    > doi: [10.1007/s10107-022-01840-5](https://doi.org/10.1007/s10107-022-01840-5).
"""

# ╔═╡ Cell order:
# ╟─285f33c1-e8ed-45d4-ba33-9e5eb17ce422
# ╠═e2dd0dd2-0444-11ed-1e55-5b3126a23f62
# ╟─c355af21-9912-4de2-9a3e-21c3dd7d2fdd
# ╠═397f4aa4-bd8b-4f98-9c5a-c3514e50ef37
# ╟─8e59bb69-e073-40cc-a304-d1256463dcbb
# ╠═09e304f8-b153-4076-81d7-a3a01644d65a
# ╠═1a96dee3-f2b2-42a4-ac5a-2af399e86463
# ╠═a56c69d9-768b-4ce5-99d7-986820b8b53c
# ╠═b2fdb060-31c6-4d2c-9e04-e5fa0caab5d0
# ╠═e5fc5216-5aab-4638-9444-02dd9b1cb4e3
# ╠═6c9c3984-2de8-4f4e-b8e9-e747059043cf
# ╠═58594aa5-1dc8-4193-914f-98ab4bfcdc03
# ╠═701ace87-ef6e-42a6-9e81-563c3abc55b4
# ╠═11130019-505c-4557-933a-ab034d6b5b7b
# ╠═41ad71e7-708f-42e9-a92b-902c6324215f
# ╠═d77613d4-ede8-44c7-bc3f-46ab4c828b90
# ╠═17ec6a97-d7af-4b35-b388-523537e88a0f
# ╠═43d14777-70dc-4e76-a481-4c5409f03115
# ╠═e6d19915-c495-4693-bdda-f83dc6d5ddfc
# ╠═3d369862-dc1c-4e06-acde-9eb1baa1a425
# ╠═cd22d1df-260b-44a8-9a0a-d349f3f588e1
# ╠═b0974627-ae5e-432e-99b9-10b3b7463615
# ╠═9c883e38-bc39-418c-8414-ba36656329f0
# ╠═c34152c0-c12c-4a8e-838e-5f867647cd19
# ╠═1a14a0de-89a3-4099-9f7b-e438df5c47bc
# ╠═c0ac2089-bd95-496f-85fc-3dee85958811
# ╠═63f5166e-4755-4892-a7f5-f7be48f2fc52
# ╠═9d0f4fc6-3f54-404a-ab26-050a5e52c458
# ╠═460a7a9d-e652-44f0-b731-a4cd03d901ea
# ╟─95f34a6e-4932-44bd-9caa-2fcded798d05
# ╠═b76baa92-1f47-4e69-a2f7-29f539300cbb
# ╠═9aa04bf2-3265-4317-8bf8-db05feb338f9
# ╟─a86b5add-3ec8-4fd9-ba62-e6ab89f0dbda
