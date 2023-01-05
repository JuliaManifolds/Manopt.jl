### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 34e43a09-0141-4ffe-9c44-0996507c0c47
using Pkg;

# ╔═╡ 70c1055f-6fd7-4dce-afca-32b7ab8237c0
using Manopt, Manifolds, Colors, Random, PlutoUI

# ╔═╡ 1c3a0022-43f1-4f2f-a4c5-9dfbe0d9437e
using LinearAlgebra: svd

# ╔═╡ 09eb0cd6-8c08-11ec-3c61-a1da0a6e4937
md"""
# Geodesic Regression

Geodesic regression generalizes [linear regression](https://en.wikipedia.org/wiki/Linear_regression)
to Riemannian manifolds. Let's first phrase it informally as follows:

> For given data points ``d_1,\ldots,d_n`` on a Riemannian manifold ``\mathcal M``, find
> the geodesic that “best explains” the data.

The meaning of “best explain” has still to be clarified. We distinguish two cases: time labelled data and unlabelled data
"""

# ╔═╡ 7df51bcb-020e-475a-8218-8fc811b6a75d
use_local= false

# ╔═╡ ea0baa25-b3e8-4359-a5f6-4b4c769aae49
use_local || Pkg.activate()

# ╔═╡ c867bd9c-da61-49a0-b6e6-c5fd12536b55
md"We define some colors from [Paul Tol](https://personal.sron.nl/~pault/)."

# ╔═╡ 39ac3269-1cc9-4eaa-9340-2e7e37648dc8
begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantTeal = RGBA{Float64}(colorant"#009988")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
    TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE")
    Random.seed!(42)
end;

# ╔═╡ c901b5ab-fdc7-43e9-b728-5dbd3b21104a
md"""
We use the following data, where you can choose one data point that you want to highlight.
"""

# ╔═╡ 501378b8-ea0b-467f-a67c-d804de9b8d01
begin
    n = 7
    highlighted = 4
    (highlighted > n - 1) && error(
        "Please choose a highlighted point from {1,...,$(n-1)} – you set it to 		highlighted.",
    )
    σ = π / 8
    S = Sphere(2)
    base = 1 / sqrt(2) * [1.0, 0.0, 1.0]
    dir = [-0.75, 0.5, 0.75]
    data_orig = [exp(S, base, dir, t) for t in range(-0.5, 0.5; length=n)]
    # add noise to the points on the geodesic
    data = map(p -> exp(S, p, rand(S; vector_at=p, σ=σ)), data_orig)
    localpath = join(splitpath(@__FILE__)[1:(end - 1)], "/") # files folder
    image_prefix = localpath * "/regression"
    @info image_prefix
    render_asy = false # on CI or when you do not have asymptote, this should be false
end;

# ╔═╡ 75b3aff8-842e-407d-84e4-6373129911f3
md"""
This notebook loads data if `render_asymptote` is set to false, and renders the images (using [Asymptote](https://asymptote.sourceforge.io)) otherwise.
"""

# ╔═╡ 115d380c-2729-46f1-82bf-32a318c4d8aa
render_asy && asymptote_export_S2_signals(
    image_prefix * "/regression_data.asy";
    points=[data],
    colors=Dict(:points => [TolVibrantBlue]),
    dot_size=3.5,
    camera_position=(1.0, 0.5, 0.5),
);

# ╔═╡ a8184bf9-133b-4952-b11d-abaaf0ee39d8
render_asy && render_asymptote(image_prefix * "/regression_data.asy"; render=2);

# ╔═╡ 24337e9c-b5c3-4aed-9043-2e7eaca8f984
PlutoUI.LocalResource(image_prefix * "/regression_data.png")

# ╔═╡ dd4a7653-edad-4400-9e29-59b9cca72679
md"""
## Time Labeled Data
If for each data item $d_i$ we are also given a time point $t_i\in\mathbb R$, which are pairwise different,
then we can use the least squares error to state the objetive function as [^Fletcher2013]

```math
F(p,X) = \frac{1}{2}\sum_{i=1}^n d_{\mathcal M}^2(γ_{p,X}(t_i), d_i),
```

where ``d_{\mathcal M}`` is the Riemannian distance and ``γ_{p,X}`` is the geodesic
with ``γ(0) = p`` and ``\dot\gamma(0) = X``.

For the real-valued case ``\mathcal M = \mathbb R^m`` the solution ``(p^*, X^*)`` is given in closed form
as follows: with ``d^* = \frac{1}{n}\displaystyle\sum_{i=1}^{n}d_i`` and ``t^* = \frac{1}{n}\displaystyle\sum_{i=1}^n t_i``
we get

```math
 X^* = \frac{\sum_{i=1}^n (d_i-d^*)(t-t^*)}{\sum_{i=1}^n (t_i-t^*)^2}
\quad\text{ and }\quad
p^* = d^* - t^*X^*
```

and hence the linear regression result is the line ``γ_{p^*,X^*}(t) = p^* + tX^*``.

On a Riemannian manifold we can phrase this as an optimization problem on the [tangent bundle](https://en.wikipedia.org/wiki/Tangent_bundle),
i.e. the disjoint union of all tangent spaces, as

```math
\operatorname*{arg\,min}_{(p,X) \in \mathrm{T}\mathcal M} F(p,X)
```

Due to linearity, the gradient of ``F(p,X)`` is the sum of the single gradients of

```math
 \frac{1}{2}d_{\mathcal M}^2\bigl(γ_{p,X}(t_i),d_i\bigr)
 = \frac{1}{2}d_{\mathcal M}^2\bigl(\exp_p(t_iX),d_i\bigr)
 ,\quad i∈\{1,\ldots,n\}
```

which can be computed using a chain rule of the squared distance and the exponential map,
see for example [^BergmannGousenbourger2018] for details or Equations (7) and (8) of [^Fletcher2013]:
"""

# ╔═╡ 3d4bd7d0-876b-4915-88a4-49b188df16b5
M = TangentBundle(S)

# ╔═╡ 92b18619-f806-412f-a125-52a7731a1389
begin
    struct RegressionCost{T,S}
        data::T
        times::S
    end
    RegressionCost(data::T, times::S) where {T,S} = RegressionCost{T,S}(data, times)
    function (a::RegressionCost)(M, x)
        pts = [geodesic(M.manifold, x[M, :point], x[M, :vector], ti) for ti in a.times]
        return 1 / 2 * sum(distance.(Ref(M.manifold), pts, a.data) .^ 2)
    end
end;

# ╔═╡ 5bd7dbf5-90cb-4e7b-8a12-eb1876cda2f2
begin
    struct RegressionGradient!{T,S}
        data::T
        times::S
    end
    function RegressionGradient!(data::T, times::S) where {T,S}
        return RegressionGradient!{T,S}(data, times)
    end
    function (a::RegressionGradient!)(M, Y, x)
        pts = [geodesic(M.manifold, x[M, :point], x[M, :vector], ti) for ti in a.times]
        gradients = grad_distance.(Ref(M.manifold), a.data, pts)
        Y[M, :point] .= sum(
            adjoint_differential_exp_basepoint.(
                Ref(M.manifold),
                Ref(x[M, :point]),
                [ti * x[M, :vector] for ti in a.times],
                gradients,
            ),
        )
        Y[M, :vector] .= sum(
            adjoint_differential_exp_argument.(
                Ref(M.manifold),
                Ref(x[M, :point]),
                [ti * x[M, :vector] for ti in a.times],
                gradients,
            ),
        )
        return Y
    end
end;

# ╔═╡ 876e011d-fc77-4fe4-a61b-1071b29d7d81
md"""
Now we just need a starting point.

For the Euclidean case, the result is given by the first principal component of a principal component analysis,
see [PCR](https://en.wikipedia.org/wiki/Principal_component_regression), i.e. with ``p^* = \frac{1}{n}\displaystyle\sum_{i=1}^n d_i``
the direction ``X^*`` is obtained by defining the zero mean data matrix

```math
D = \bigl(d_1-p^*, \ldots, d_n-p^*\bigr) \in \mathbb R^{m,n}
```

and taking ``X^*`` as an eigenvector to the largest eigenvalue of ``D^{\mathrm{T}}D``.

We can do something similar, when considering the tangent space at the (Riemannian) mean
of the data and then do a PCA on the coordinate coefficients with respect to a basis.
"""

# ╔═╡ 39015e8f-1fcd-4dc1-87d4-15be5ea9f8fa
begin
    m = mean(S, data)
    A = hcat(
        map(x -> get_coordinates(S, m, log(S, m, x), DefaultOrthonormalBasis()), data)...
    )
    pca1 = get_vector(S, m, svd(A).U[:, 1], DefaultOrthonormalBasis())
    x0 = ProductRepr(m, pca1)
end

# ╔═╡ 9302c3ff-f7ac-4104-b80e-9678a0d68ab8
md"""
The optimal “time labels” are then just the projections ``t_i = ⟨d_i,X^*⟩``, ``i=1,\ldots,n``.
"""

# ╔═╡ ff15718a-5edb-4ccc-8782-597b261be970
t = map(d -> inner(S, m, pca1, log(S, m, d)), data)

# ╔═╡ f6deecdd-31b9-493d-a5ed-d9e62cfea9f7
md"""
And we can call the gradient descent. Note that since `gradF!` works in place of `Y`, we have to set the
`evalutation` type accordingly.
"""

# ╔═╡ bada1b3c-6a62-4896-aa4a-7f1f52694682
with_terminal() do
    global y = gradient_descent(
        M,
        RegressionCost(data, t),
        RegressionGradient!(data, t),
        x0;
        evaluation=InplaceEvaluation(),
        stepsize=ArmijoLinesearch(
            M;
            initial_stepsize=1.0,
            contraction_factor=0.990,
            sufficient_decrease=0.05,
            linesearch_stopsize=1e-9,
        ),
        stopping_criterion=StopAfterIteration(200) |
                           StopWhenGradientNormLess(1e-8) |
                           StopWhenStepsizeLess(1e-9),
        debug=[:Iteration, " | ", :Cost, "\n", :Stop, 50],
    )
end

# ╔═╡ 72f5ffef-6133-440b-842a-566c0eb09377
md"The result looks like"

# ╔═╡ f58e540f-e83e-4a6c-9e21-74bae7b3d69f
begin
    dense_t = range(-0.5, 0.5; length=100)
    geo = geodesic(S, y[M, :point], y[M, :vector], dense_t)
    init_geo = geodesic(S, x0[M, :point], x0[M, :vector], dense_t)
    geo_pts = geodesic(S, y[M, :point], y[M, :vector], t)
    geo_conn_highlighted = shortest_geodesic(
        S, data[highlighted], geo_pts[highlighted], 0.5 .+ dense_t
    )
end

# ╔═╡ 06c520d1-720c-4eb4-97fb-96870aba73a0
render_asy && asymptote_export_S2_signals(
    image_prefix * "/regression_result1.asy";
    points=[data, [y[M, :point]], geo_pts],
    curves=[init_geo, geo, geo_conn_highlighted],
    tangent_vectors=[[Tuple([y[M, :point], y[M, :vector]])]],
    colors=Dict(
        :curves => [black, TolVibrantTeal, TolVibrantBlue],
        :points => [TolVibrantBlue, TolVibrantOrange, TolVibrantTeal],
        :tvectors => [TolVibrantOrange],
    ),
    dot_sizes=[3.5, 3.5, 2],
    line_widths=[0.33, 0.66, 0.33, 1.0],
    camera_position=(1.0, 0.5, 0.5),
);

# ╔═╡ ab26bdda-9c18-4df3-9a39-c81423a84c16
render_asy && render_asymptote(image_prefix * "/regression_result1.asy"; render=2);

# ╔═╡ 7c491e8b-44d6-420b-89a1-cbf947a98ffc
PlutoUI.LocalResource(image_prefix * "/regression_result1.png")

# ╔═╡ c1f18ab3-c804-43bd-8aba-b83446c355f0
md"""
In this image, together with the blue data points, you see the geodesic of the initialization in black
(evaluated on ``[-\frac{1}{2},\frac{1}{2}]``),
the final point on the tangent bundle in orange, as well as the resulting regression geodesic in teal,
(on the same interval as the start) as well as small teal points indicating the time points on the geodesic corresponding to the data.
Additionally, a thin blue line indicates the geodesic between a data point and its corresponding data point on the geodesic.
While this would be the closest point in Euclidean space and hence the two directions (along the geodesic vs. to the data point) orthogonal, here we have
"""

# ╔═╡ 1d408cc3-7fff-4b85-a264-0841372b3b2d
inner(
    S,
    geo_pts[highlighted],
    log(S, geo_pts[highlighted], geo_pts[highlighted + 1]),
    log(S, geo_pts[highlighted], data[highlighted]),
)

# ╔═╡ 8a371cb2-101f-448b-b219-61e298761026
md"""
But we also started with one of the best scenarios, i.e. equally spaced points on a geodesic obstructed by noise.

This gets worse if you start with less evenly distributed data
"""

# ╔═╡ 0bf7e944-1d9f-490f-8a86-d9dffc564256
begin
    data2 = [exp(S, base, dir, t) for t in [-0.5, -0.49, -0.48, 0.1, 0.48, 0.49, 0.5]]
    data2 = map(p -> exp(S, p, rand(S; vector_at=p, σ=σ / 2)), data2)
    m2 = mean(S, data2)
    A2 = hcat(
        map(x -> get_coordinates(S, m, log(S, m, x), DefaultOrthonormalBasis()), data2)...
    )
    pca2 = get_vector(S, m, svd(A2).U[:, 1], DefaultOrthonormalBasis())
    x1 = ProductRepr(m, pca2)
    t2 = map(d -> inner(S, m2, pca2, log(S, m2, d)), data2)
end

# ╔═╡ 3f584132-46c7-42dd-9e2c-0bf2bbf38eb4
with_terminal() do
    global y2 = gradient_descent(
        M,
        RegressionCost(data2, t2),
        RegressionGradient!(data2, t2),
        x1;
        evaluation=InplaceEvaluation(),
        stepsize=ArmijoLinesearch(
            M;
            initial_stepsize=1.0,
            contraction_factor=0.990,
            sufficient_decrease=0.05,
            linesearch_stopsize=1e-9,
        ),
        stopping_criterion=StopAfterIteration(200) |
                           StopWhenGradientNormLess(1e-8) |
                           StopWhenStepsizeLess(1e-9),
        debug=[:Iteration, " | ", :Cost, "\n", :Stop, 50],
    )
end

# ╔═╡ c8badbb1-b999-4398-9967-6966a81bcbc1
begin
    geo2 = geodesic(S, y2[M, :point], y2[M, :vector], dense_t)
    init_geo2 = geodesic(S, x1[M, :point], x1[M, :vector], dense_t)
    geo_pts2 = geodesic(S, y2[M, :point], y2[M, :vector], t2)
    geo_conn_highlighted2 = shortest_geodesic(
        S, data2[highlighted], geo_pts2[highlighted], 0.5 .+ dense_t
    )
end

# ╔═╡ e4d16b82-3d18-4c1a-bb85-db400c5c65f7
render_asy && asymptote_export_S2_signals(
    image_prefix * "/regression_result2.asy";
    points=[data2, [y2[M, :point]], geo_pts2],
    curves=[init_geo2, geo2, geo_conn_highlighted2],
    tangent_vectors=[[Tuple([y2[M, :point], y2[M, :vector]])]],
    colors=Dict(
        :curves => [black, TolVibrantTeal, TolVibrantCyan],
        :points => [TolVibrantBlue, TolVibrantOrange, TolVibrantTeal],
        :tvectors => [TolVibrantOrange],
    ),
    dot_sizes=[3.5, 3.5, 2.5],
    line_widths=[0.33, 0.66, 0.33, 1.0],
    camera_position=(1.0, 0.5, 0.5),
);

# ╔═╡ 837f2bc0-a1c1-4d39-adb3-8a7e262dda7d
render_asy && render_asymptote(image_prefix * "/regression_result2.asy"; render=2);

# ╔═╡ ea577515-04d1-4d80-af32-aa941d5a2efb
PlutoUI.LocalResource(image_prefix * "/regression_result2.png")

# ╔═╡ 4bb8db61-2f35-4c41-af1d-4b053536b291
md"""
## Unlabeled Data

If we are not given time points $t_i$, then the optimization problem extends – informally speaking –
to also finding the “best fitting” (in the sense of smallest error).
To formalize, the objective function here reads


```math
F(p, X, t) = \frac{1}{2}\sum_{i=1}^n d_{\mathcal M}^2(γ_{p,X}(t_i), d_i),
```

where ``t = (t_1,\ldots,t_n) \in \mathbb R^n`` is now an additional parameter of the objective function.
We write ``F_1(p, X)`` to refer to the function on the tangent bundle for fixed values of ``t`` (as the one in the last part)
and ``F_2(t)`` for the function ``F(p, X, t)`` as a function in ``t`` with fixed values ``(p, X)``.

For the Euclidean case, there is no neccessity to optimize with respect to ``t``, as we saw
above for the initialization of the fixed time points.

On a Riemannian manifold this can be stated as a problem on the product manifold ``\mathcal N = \mathrm{T}\mathcal M \times \mathbb R^n``, i.e.

"""

# ╔═╡ d762b0e0-d722-4fab-a46c-647a07853354
N = M × Euclidean(length(t2))

# ╔═╡ 03c99340-7927-4404-ba23-e0226e3f32a4
md"""
```math
  \operatorname*{arg\,min}_{\bigl((p,X),t\bigr)\in\mathcal N} F(p, X, t).
```

In this tutorial we present an approach to solve this using an alternating gradient descent scheme.
To be precise, we define the cost funcion now on the product manifold
"""

# ╔═╡ 10b85d9f-e1ec-46e2-99ce-7e25d1dc7ac0
begin
    struct RegressionCost2{T}
        data::T
    end
    RegressionCost2(data::T) where {T} = RegressionCost2{T}(data)
    function (a::RegressionCost2)(N, x)
        TM = N[1]
        pts = [
            geodesic(TM.manifold, x[N, 1][TM, :point], x[N, 1][TM, :vector], ti) for
            ti in x[N, 2]
        ]
        return 1 / 2 * sum(distance.(Ref(TM.manifold), pts, a.data) .^ 2)
    end
end

# ╔═╡ 498dde67-324c-431e-970d-b200f9835f83
md"""
The gradient in two parts, namely (a) the same gradient as before w.r.t. ``(p,X) ∈ T\mathcal M``,
just now with a fixed `t` in mind for the second component of the product manifold ``\mathcal N``
"""

# ╔═╡ 52ee5cdd-4a45-47c6-ba25-482d81cb1b6a
begin
    struct RegressionGradient2a!{T}
        data::T
    end
    RegressionGradient2a!(data::T) where {T} = RegressionGradient2a!{T}(data)
    function (a::RegressionGradient2a!)(N, Y, x)
        TM = N[1]
        p = x[N, 1]
        pts = [geodesic(TM.manifold, p[TM, :point], p[TM, :vector], ti) for ti in x[N, 2]]
        gradients = grad_distance.(Ref(TM.manifold), a.data, pts)
        Y[TM, :point] .= sum(
            adjoint_differential_exp_basepoint.(
                Ref(TM.manifold),
                Ref(p[TM, :point]),
                [ti * p[TM, :vector] for ti in x[N, 2]],
                gradients,
            ),
        )
        Y[TM, :vector] .= sum(
            adjoint_differential_exp_argument.(
                Ref(TM.manifold),
                Ref(p[TM, :point]),
                [ti * p[TM, :vector] for ti in x[N, 2]],
                gradients,
            ),
        )
        return Y
    end
end

# ╔═╡ 9da6b6b0-303e-4865-afdd-aad397495517
md"""
Finally, we addionally look for a fixed point ``x=(p,X) ∈ \mathrm{T}\mathcal M`` at
the gradient with respect to ``t∈\mathbb R^n``, i.e. the second component, which is given by

```math
  (\operatorname{grad}F_2(t))_i
  = - ⟨\dot γ_{p,X}(t_i), \log_{γ_{p,X}(t_i)}d_i⟩_{γ_{p,X}(t_i)}, i = 1, \ldots, n.
```
"""

# ╔═╡ 96907f2c-e7e4-4bbe-8a2e-91295fac2a8c
begin
    struct RegressionGradient2b!{T}
        data::T
    end
    RegressionGradient2b!(data::T) where {T} = RegressionGradient2b!{T}(data)
    function (a::RegressionGradient2b!)(N, Y, x)
        TM = N[1]
        p = x[N, 1]
        pts = [geodesic(TM.manifold, p[TM, :point], p[TM, :vector], ti) for ti in x[N, 2]]
        logs = log.(Ref(TM.manifold), pts, a.data)
        pt = map(
            d -> vector_transport_to(TM.manifold, p[TM, :point], p[TM, :vector], d), pts
        )
        Y .= -inner.(Ref(TM.manifold), pts, logs, pt)
        return Y
    end
end

# ╔═╡ 52138cac-404a-41e0-ad6e-a7cb51dbec3e
md"""
We can reuse the computed initial values from before, just that now we are on a product manifold
"""

# ╔═╡ a64b78bb-c226-4c68-af5a-02659a4a2dd9
begin
    x2 = ProductRepr(x1, t2)
    F3 = RegressionCost2(data2)
    gradF3_vector = [RegressionGradient2a!(data2), RegressionGradient2b!(data2)]
end;

# ╔═╡ 27548918-eb14-404c-97a6-d39a1fc87929
with_terminal() do
    global y3 = alternating_gradient_descent(
        N,
        F3,
        gradF3_vector,
        x2;
        evaluation=InplaceEvaluation(),
        debug=[:Iteration, " | ", :Cost, "\n", :Stop, 50],
        stepsize=ArmijoLinesearch(M; contraction_factor=0.999, sufficient_decrease=0.066, linesearch_stopsize=1e-11, retraction_method=ProductRetraction(SasakiRetraction(2), ExponentialRetraction())),
        inner_iterations=1,
    )
end

# ╔═╡ 1099f1b0-6304-491b-84c9-6f39e08968a9
begin
    geo3 = geodesic(S, y3[N, 1][M, :point], y3[N, 1][M, :vector], dense_t)
    init_geo3 = geodesic(S, x1[M, :point], x1[M, :vector], dense_t)
    geo_pts3 = geodesic(S, y3[N, 1][M, :point], y3[N, 1][M, :vector], y3[N, 2])
    t3 = y3[N, 2]
    geo_conns = shortest_geodesic.(Ref(S), data2, geo_pts3, Ref(0.5 .+ 4*dense_t))
end;

# ╔═╡ 2d8d4f2f-85cf-49fc-a204-02e28566b582
render_asy && asymptote_export_S2_signals(
    image_prefix * "/regression_result3.asy";
    points=[data2, [y3[N, 1][M, :point]], geo_pts3],
    curves=[init_geo3, geo3, geo_conns...],
    tangent_vectors=[[Tuple([y3[N, 1][M, :point], y3[N, 1][M, :vector]])]],
    colors=Dict(
        :curves =>
            [black, TolVibrantTeal, repeat([TolVibrantCyan], length(geo_conns))...],
        :points => [TolVibrantBlue, TolVibrantOrange, TolVibrantTeal],
        :tvectors => [TolVibrantOrange],
    ),
    dot_sizes=[3.5, 3.5, 2.5],
    line_widths=[0.33, 0.66, repeat([0.33], length(geo_conns))..., 1.0],
    camera_position=(1.0, 0.5, 0.5),
);

# ╔═╡ 695478c0-e8e5-4621-bce4-e76eb1f00835
render_asy && render_asymptote(image_prefix * "/regression_result3.asy"; render=2);

# ╔═╡ 6f72b449-e51d-4b6f-bc2a-cd3631343286
PlutoUI.LocalResource(image_prefix * "/regression_result3.png")

# ╔═╡ 426d965f-04e9-4d8e-b44b-bf8d1b85d84b
md"""
Note that the geodesics from the data to the regression geodesic meet at a nearly orthogonal angle.

**Acknowledgement.** Parts of this tutorial are based on the bachelor thesis of
[Jeremias Arf](https://orcid.org/0000-0003-3765-0130).
"""

# ╔═╡ c54d2401-023f-424d-a9a0-db5683de50de
md"""
## References

[^BergmannGousenbourger2018]:
    > Bergmann, R. and Gousenbourger, P.-Y.: _A variational model for data fitting on manifolds
    > by minimizing the acceleration of a Bézier curve_.
    > Frontiers in Applied Mathematics and Statistics, 2018.
    > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059),
    > arXiv: [1807.10090](https://arxiv.org/abs/1807.10090)
[^Fletcher2013]:
    > Fletcher, P. T., _Geodesic regression and the theory of least squares on Riemannian manifolds_,
    > International Journal of Computer Vision(105), 2, pp. 171–185, 2013.
    > doi: [10.1007/s11263-012-0591-y](https://doi.org/10.1007/s11263-012-0591-y)
"""

# ╔═╡ Cell order:
# ╟─09eb0cd6-8c08-11ec-3c61-a1da0a6e4937
# ╠═7df51bcb-020e-475a-8218-8fc811b6a75d
# ╠═34e43a09-0141-4ffe-9c44-0996507c0c47
# ╠═ea0baa25-b3e8-4359-a5f6-4b4c769aae49
# ╠═70c1055f-6fd7-4dce-afca-32b7ab8237c0
# ╠═1c3a0022-43f1-4f2f-a4c5-9dfbe0d9437e
# ╟─c867bd9c-da61-49a0-b6e6-c5fd12536b55
# ╠═39ac3269-1cc9-4eaa-9340-2e7e37648dc8
# ╟─c901b5ab-fdc7-43e9-b728-5dbd3b21104a
# ╠═501378b8-ea0b-467f-a67c-d804de9b8d01
# ╟─75b3aff8-842e-407d-84e4-6373129911f3
# ╠═115d380c-2729-46f1-82bf-32a318c4d8aa
# ╠═a8184bf9-133b-4952-b11d-abaaf0ee39d8
# ╠═24337e9c-b5c3-4aed-9043-2e7eaca8f984
# ╟─dd4a7653-edad-4400-9e29-59b9cca72679
# ╠═3d4bd7d0-876b-4915-88a4-49b188df16b5
# ╠═92b18619-f806-412f-a125-52a7731a1389
# ╠═5bd7dbf5-90cb-4e7b-8a12-eb1876cda2f2
# ╟─876e011d-fc77-4fe4-a61b-1071b29d7d81
# ╠═39015e8f-1fcd-4dc1-87d4-15be5ea9f8fa
# ╟─9302c3ff-f7ac-4104-b80e-9678a0d68ab8
# ╠═ff15718a-5edb-4ccc-8782-597b261be970
# ╟─f6deecdd-31b9-493d-a5ed-d9e62cfea9f7
# ╠═bada1b3c-6a62-4896-aa4a-7f1f52694682
# ╟─72f5ffef-6133-440b-842a-566c0eb09377
# ╠═f58e540f-e83e-4a6c-9e21-74bae7b3d69f
# ╠═06c520d1-720c-4eb4-97fb-96870aba73a0
# ╠═ab26bdda-9c18-4df3-9a39-c81423a84c16
# ╠═7c491e8b-44d6-420b-89a1-cbf947a98ffc
# ╟─c1f18ab3-c804-43bd-8aba-b83446c355f0
# ╠═1d408cc3-7fff-4b85-a264-0841372b3b2d
# ╟─8a371cb2-101f-448b-b219-61e298761026
# ╠═0bf7e944-1d9f-490f-8a86-d9dffc564256
# ╠═3f584132-46c7-42dd-9e2c-0bf2bbf38eb4
# ╠═c8badbb1-b999-4398-9967-6966a81bcbc1
# ╠═e4d16b82-3d18-4c1a-bb85-db400c5c65f7
# ╠═837f2bc0-a1c1-4d39-adb3-8a7e262dda7d
# ╠═ea577515-04d1-4d80-af32-aa941d5a2efb
# ╟─4bb8db61-2f35-4c41-af1d-4b053536b291
# ╠═d762b0e0-d722-4fab-a46c-647a07853354
# ╟─03c99340-7927-4404-ba23-e0226e3f32a4
# ╠═10b85d9f-e1ec-46e2-99ce-7e25d1dc7ac0
# ╟─498dde67-324c-431e-970d-b200f9835f83
# ╠═52ee5cdd-4a45-47c6-ba25-482d81cb1b6a
# ╟─9da6b6b0-303e-4865-afdd-aad397495517
# ╠═96907f2c-e7e4-4bbe-8a2e-91295fac2a8c
# ╟─52138cac-404a-41e0-ad6e-a7cb51dbec3e
# ╠═a64b78bb-c226-4c68-af5a-02659a4a2dd9
# ╠═27548918-eb14-404c-97a6-d39a1fc87929
# ╠═1099f1b0-6304-491b-84c9-6f39e08968a9
# ╠═2d8d4f2f-85cf-49fc-a204-02e28566b582
# ╠═695478c0-e8e5-4621-bce4-e76eb1f00835
# ╠═6f72b449-e51d-4b6f-bc2a-cd3631343286
# ╟─426d965f-04e9-4d8e-b44b-bf8d1b85d84b
# ╟─c54d2401-023f-424d-a9a0-db5683de50de
