### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

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
    data = map(x -> exp(S, x, random_tangent(S, x, :Gaussian, σ)), data_orig)
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
        evaluation=MutatingEvaluation(),
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
    data2 = map(x -> exp(S, x, random_tangent(S, x, :Gaussian, σ / 2)), data2)
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
        evaluation=MutatingEvaluation(),
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
        evaluation=MutatingEvaluation(),
        debug=[:Iteration, " | ", :Cost, "\n", :Stop, 50],
        stepsize=ArmijoLinesearch(1.0, ExponentialRetraction(), 0.999, 0.066, 1e-11),
        inner_iterations=1,
    )
end

# ╔═╡ 1099f1b0-6304-491b-84c9-6f39e08968a9
begin
    geo3 = geodesic(S, y3[N, 1][M, :point], y3[N, 1][M, :vector], dense_t)
    init_geo3 = geodesic(S, x1[M, :point], x1[M, :vector], dense_t)
    geo_pts3 = geodesic(S, y3[N, 1][M, :point], y3[N, 1][M, :vector], y3[N, 2])
    t3 = y3[N, 2]
    geo_conns = shortest_geodesic.(Ref(S), data2, geo_pts3, Ref(0.5 .+ dense_t))
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Colors = "~0.12.8"
Manifolds = "~0.8.31"
Manopt = "~0.3.40"
PlutoUI = "~0.7.43"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.1"
manifest_format = "2.0"
project_hash = "6d10d66c0d63e06254fcd8ce21841cf16ad5e16d"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5bb0f8292405a516880a3809954cb832ae7a31c5"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.20"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "3ca828fe1b75fa84b021a7860bd039eaea84d2f2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "3c8de95b4e932d76ec8960e12d681eba580e9674"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.8"

[[deps.DataAPI]]
git-tree-sha1 = "46d2680e618f8abd007bce0c3026cb0c4a8f2032"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.12.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0d7d213133d948c56e8c2d9f4eab0293491d8e4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.75"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "87519eb762f85534445f5cda35be12e32759ee14"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.4"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ba2d094a88b6b287bd25cfa86f301e7693ffae2f"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.4"

[[deps.HybridArrays]]
deps = ["LinearAlgebra", "Requires", "StaticArrays"]
git-tree-sha1 = "0de633a951f8b5bd32febc373588517aa9f2f482"
uuid = "1baab800-613f-4b0a-84e4-9cd3431bfbb9"
version = "0.4.13"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "78d9909daf659c901ae6c7b9de7861ba45a743f4"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "d1b46faefb7c2f48fdec69e6f3cc34857769bc15"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.8.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Manifolds]]
deps = ["Colors", "Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldsBase", "Markdown", "MatrixEquations", "Quaternions", "Random", "RecipesBase", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "92bb59b8baa0dac76294290270708b7a599d1aef"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.8.31"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown"]
git-tree-sha1 = "d39d5f8f117c9b370f4b8520182f48ecf9e32620"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.13.21"

[[deps.Manopt]]
deps = ["ColorSchemes", "ColorTypes", "Colors", "DataStructures", "Dates", "LinearAlgebra", "ManifoldsBase", "Markdown", "Printf", "Random", "Requires", "SparseArrays", "StaticArrays", "Statistics", "Test"]
git-tree-sha1 = "7f4beb9dfeb4329ead0d6fb11f5a1bc6539c7f13"
uuid = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
version = "0.3.40"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "3b284e9c98f645232f9cf07d4118093801729d43"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.2.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "cb8ebcee2b4e07b72befb9def593baef8aa12f07"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.50"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "595c0b811cf2bab8b0849a70d9bd6379cc1cfb52"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "2777a5c2c91b3145f5aa75b61bb4c2eb38797136"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.43"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "3c009334f45dfd546a16a57960a821a1a023d241"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.5.0"

[[deps.Quaternions]]
deps = ["DualNumbers", "LinearAlgebra", "Random"]
git-tree-sha1 = "4ab19353944c46d65a10a75289d426ef57b0a40c"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.5.7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "612a4d76ad98e9722c8ba387614539155a59e30c"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.0"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "3004608dc42101a944e44c1c68b599fa7c669080"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "f86b3a049e5d05227b10e15dbb315c5b90f14988"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.9"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "2d7164f7b8a066bcfa6224e67736ce0eb54aef5b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.9.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─09eb0cd6-8c08-11ec-3c61-a1da0a6e4937
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
