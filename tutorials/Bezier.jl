### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 6f9be298-5374-11ec-3b9d-fd74a32d6b53
using Colors, PlutoUI, Manopt, Manifolds

# ╔═╡ e701b513-4ad9-474c-9d79-60a24f545ebd
md"""
# Bezier curves and their acceleration

This tutorial illustrates how Bézier curves are generalized to manifolds and how to
minimize their acceleration, i.e. how to get a curve that is as straight or as geodesic
while fulfilling constraints

This example also illustrates how to apply the minimization on the corresponding [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html) manifold using a [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent.html) with [`ArmijoLinesearch`](https://manoptjl.org/stable/plans/index.html#Manopt.ArmijoLinesearch).
"""

# ╔═╡ 087fb979-d68d-4ee7-9be5-fad992fdb792
md"and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)"

# ╔═╡ 85d8985b-f5ee-4cba-bcc5-528049798330
begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
    TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE")
    TolVibrantTeal = RGBA{Float64}(colorant"#009988")
end;

# ╔═╡ e62e36e9-19ce-4a7d-8bda-5634813c1ee9
md"Let's also set up a few parameters"

# ╔═╡ 18d378dd-abc6-4fa6-a2c9-1cc09e4cec46
begin
    geo_pts = collect(range(0.0, 1.0; length=101))
    bezier_pts = collect(range(0.0, 3.0; length=201))
    camera_position = (-1.0, -0.7, 0.3)
    #render asy yes/no. If not, images included w/ markdown are assumed to be prerendered
    render_asy = false
    localpath = join(splitpath(@__FILE__)[1:(end - 1)], "/") # remove file to get files folder
    image_prefix = localpath * "/bezier"
    @info image_prefix
end;

# ╔═╡ dee4eb65-145c-4b2b-98a7-ae3a7045f24b
md"""
We finally load our data, see [`artificial_S2_composite_bezier_curve`](https://manoptjl.org/stable/helpers/data.html#Manopt.artificial_S2_composite_bezier_curve-Tuple{}), a composite Bezier curve consisting of 3 segments on the Sphere. The middle segment consists of the control points
"""

# ╔═╡ 2b485956-459d-40e7-9364-79ea77073804
begin
    B = artificial_S2_composite_bezier_curve()
    b = B[2].pts
end

# ╔═╡ 98c925ac-9ac8-42f3-9855-a01e5eaac350
M = Sphere(2)

# ╔═╡ d22e825e-d8ff-44fe-8b06-b14c09765180
md"""
On Euclidean spaces Bézier curves of these ``n=4`` so called control points like this segment yield polynomials of degree``3``.
The resulting curve ``γ: [0,1] → ℝ^m`` is called [Bezier curve](https://en.wikipedia.org/wiki/Bézier_curve) or Bézier spline and is named after [Piérre Bezier](https://en.wikipedia.org/wiki/Pierre_Bézier) (1910–1999).
They can be evaluated by the de Casteljau algorithm by evaluating line segments between points.
While it is not easy to evaluate polynomials on a manifold, evaluating line segments generalizes to the evaluation of [`shortest_geodesic`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.shortest_geodesic-Tuple{Manifold,Any,Any})s.
We will illustrate this using these points ``b=(b_1,b_2,b_3,b_4)`` on the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/sphere.html) ``\mathbb S^2``.
Let's evaliuate this at the point ``t=\frac{1}{4}∈[0,1]``. We first compute
"""

# ╔═╡ 157d13c8-7311-4832-a9ff-da22125b2bbf
begin
    t = 0.66
    pts1 = shortest_geodesic.(Ref(M), b[1:3], b[2:4], Ref(t))
end

# ╔═╡ 5af58241-50c7-464e-bfa3-53819e605793
md"""
We obtain 3 points on the geodesics connecting the control points. Repeating this again twice
"""

# ╔═╡ b372fcfc-cad0-48bc-b759-54d15bd2e28b
begin
    pts2 = shortest_geodesic.(Ref(M), pts1[1:2], pts1[2:3], Ref(t))
    p = shortest_geodesic(M, pts2[1], pts2[2], t)
end

# ╔═╡ 62115e3f-cdfe-464c-91a1-eb80e674a610
md"""we obtain the point on the Bézier curve ``c(t)``."""

# ╔═╡ 8653ca13-21e9-4049-a81e-02b6047d2c66
begin
    curves1 = [shortest_geodesic(M, b[i], b[i + 1], geo_pts) for i in 1:3]
    curves2 = [shortest_geodesic(M, pts1[i], pts1[i + 1], geo_pts) for i in 1:2]
    bezier_curve = [shortest_geodesic(M, pts2[1], pts2[2], geo_pts)]
end;

# ╔═╡ 05612a12-1080-4b79-b505-198e079a4b2a
render_asy && begin
    asymptote_export_S2_signals(
        image_prefix * "/Casteljau-illustr.asy";
        points=[b, pts1, pts2, [p]],
        curves=[curves1..., curves2..., bezier_curve..., de_casteljau(M, B[2], geo_pts)],
        colors=Dict(
            :points =>
                [TolVibrantBlue, TolVibrantCyan, TolVibrantTeal, TolVibrantOrange],
            :curves => [
                TolVibrantBlue,
                TolVibrantBlue,
                TolVibrantBlue,
                TolVibrantCyan,
                TolVibrantCyan,
                TolVibrantTeal,
                black,
            ],
        ),
        dot_size=3.5,
        line_widths=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5],
        camera_position=camera_position,
    )
    render_asymptote(image_prefix * "/Casteljau-illustr.asy"; render=2)
end;

# ╔═╡ bdb85fb8-0707-47d1-95ee-9bf2e56500f5
md"""This image summarizes Casteljaus algorithm:"""

# ╔═╡ 38b8f8af-ba81-4d3b-83b7-56b0d41ea708
PlutoUI.LocalResource(image_prefix * "/Casteljau-illustr.png")

# ╔═╡ d6759a8e-a705-4d95-a8da-e172ada644d8
md"""
From the control points (blue) and their geodesics, ont evaluation per geodesic yields three interims points (cyan), their two successive geodeics another two points (teal) and at its geodesic at ``t=0.66`` we obtain the point on the curve.

In Manopt.jl, to evaluate a Bézier curve knowing its [`BezierSegment`](https://manoptjl.org/stable/functions/bezier.html#Manopt.BezierSegment), use [`de_casteljau`](https://manoptjl.org/stable/functions/bezier.html#Manopt.de_casteljau-Tuple{AbstractManifold,%20Vararg{Any,%20N}%20where%20N}).

There are a few nice observations to make, that hold also for these Bézier curves on manifolds:
* The curve starts in the first controlpoint ``b_0`` and ends in the last controlpoint ``b_3``
* The tangent vector to the curve at the start ``\dot c(0)`` is equal to ``\frac{1}{3}\log_{b_0}b_1 = \dot γ_{b_0,b_1}(0)``, where ``γ_{a,b}`` denotes the shortest geodesic between ``a`` and ``b``.
* The tangent vector to the curve at the end ``\dot c(1)`` is equal to ``-\frac{1}{3}\log_{b_3}b_2 = -γ_{b_3,b_2}(0) = \dot γ_{b_2,b_3}(1)``.
* the curve is differentiable.

For more details on these properties, see for example [^PopielNoakes2007].
"""

# ╔═╡ 7c6947a0-7d0b-4459-b999-58ea4f2b0169
md"""
## Composite Bézier curves

With the properties of a single Bézier curve, also called Bézier segment, we can “stitch” curves together. Let ``a_0,…,a_n`` and ``b_0,…,b_m`` be two sets of controlpoints for the Bézier segments ``c(t)`` and ``d(t)``, respectively.
We define the composite Bézier curve by ``B(t) = \begin{cases} c(t) & \text{ if } 0\leq t < 1, \\ d(t-1) & \text{ if } 1\leq t \leq 2,\end{cases}`` where ``t∈[0,2]``.
This can of course be generalised straight forward to more than two cases.
With the properties from the previous section we can now state that

* the curve ``B(t)`` is continuous if ``c(1)=d(0)`` or in other words ``a_n=b_0``
* the curve ``B(t)`` is differentiable if additionally ``\dot c(1)=\dot d(0)`` or in
other words ``-\log_{a_n}a_{n-1} = \log_{b_0}b_1``. This is equivalent to ``a_n=b_0 = \gamma_{a_{n-1}b_1}(\tfrac{1}{2})``.

One nice interpretation of the last characterization is, that the tangents ``\log_{a_n}a_{n-1}`` and ``\log_{b_0}b_1`` point into opposite directions.
For a continuous curve, the first point of every segment (except for the first segment) can be ommitted, for a differentiable curve the first two points (except for the first segment) can be ommitted.
You can reduce storage by calling [`get_bezier_points`](https://manoptjl.org/stable/functions/bezier.html#Manopt.get_bezier_points), though for econstruciton with [`get_bezier_segments`](https://manoptjl.org/stable/functions/bezier.html#Manopt.get_bezier_segments-Union{Tuple{P},%20Tuple{AbstractManifold,%20Vector{P},%20Any},%20Tuple{AbstractManifold,%20Vector{P},%20Any,%20Symbol}}%20where%20P) you also need [`get_bezier_degrees`](https://manoptjl.org/stable/functions/bezier.html#Manopt.get_bezier_degrees-Tuple{AbstractManifold,%20AbstractVector{var%22#s56%22}%20where%20var%22#s56%22%3C:BezierSegment}).
The reduced storage is represented as an array of points, i.e. an element of the corresponding [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html).

For the three segment example from the beginning this looks as follows
"""

# ╔═╡ 3f1d0989-2f97-4945-82f8-22e9c7bda6d2
render_asy && begin
    asymptote_export_S2_signals(
        image_prefix * "/Bezier-composite-curve.asy";
        curves=[de_casteljau(M, B, bezier_pts)],
        points=[get_bezier_junctions(M, B), get_bezier_inner_points(M, B)],
        tangent_vectors=[[
            Tuple(a) for a in zip(
                get_bezier_junctions(M, B, true),
                get_bezier_junction_tangent_vectors(M, B),
            )
        ]],
        colors=Dict(
            :curves => [black],
            :points => [TolVibrantBlue, TolVibrantTeal],
            :tvectors => [TolVibrantCyan],
        ),
        camera_position=camera_position,
        arrow_head_size=10.0,
        line_widths=[1.5, 1.5],
        dot_size=4.0,
    )
    render_asymptote(image_prefix * "/Bezier-composite-curve.asy"; render=2)
end;

# ╔═╡ 41ba8ac2-5a81-46f5-9269-404cc6a1d287
PlutoUI.LocalResource(image_prefix * "/Bezier-composite-curve.png")

# ╔═╡ e7fc05b3-a198-4a64-b62a-43cdf8f56238
md"""
## Minimizing the acceleration of a composite Bézier curve
The motivation to minimize the acceleration of the composite Bézier curve is, that the curve should get “straighter” or more geodesic like.
If we discretize the curve ``B(t)`` with its control points denoted by ``b_{i,j}`` for the ``j``th note in the ``i``th segment, the discretized model for equispaced ``t_i``, ``i=0,…,N`` in the domain of ``B`` reads[^BergmannGousenbourger2018]

```math
A(b) :eqq\sum_{i=1}^{N-1}\frac{\mathrm{d}^2_2 \bigl[ B(t_{i-1}), B(t_{i}), B(t_{i+1}) \bigr]}{\Delta_t^3},
```

where $\mathrm{d}_2$ denotes the second order finite difference using the mid point approach, see [`costTV2`](@ref)[^BacakBergmannSteidlWeinmann2016],

```math
d_2(x,y,z) := \min_{c ∈ \mathcal C_{x,z}} d_{\mathcal M}(c,y),\qquad x,y,z∈\mathcal M.
```

Another model is based on logarithmic maps, see [^BoumalAbsil2011], but that is not considered here.
An advantage of the model considered here is, that it only consist of the evaluation of geodesics.
This yields a gradient of ``A(b)`` with respect to ``b`` [`adjoint_Jacobi_field`](https://manoptjl.org/stable/functions/Jacobi_fields.html#Manopt.adjoint_Jacobi_field)s. The following image shows the negative gradient (scaled)
"""

# ╔═╡ 7b545a1e-b9c3-4c57-a5ee-3a629407b67a
render_asy && begin
    gradFullB = Manopt._grad_acceleration_bezier(
        M,
        get_bezier_points(M, B, :differentiable),
        [3, 3, 3],
        collect(range(0.0, 3.0; length=151)),
    )
    asymptote_export_S2_signals(
        image_prefix * "/Bezier-composite-curve-gradient.asy";
        curves=[de_casteljau(M, B, bezier_pts)],
        points=[get_bezier_junctions(M, B), get_bezier_inner_points(M, B)],
        tangent_vectors=[[
            Tuple(a) for a in zip(
                get_bezier_points(M, B, :continuous),
                -0.05 .* get_bezier_points(M, gradFullB, :continuous),
            )
        ]],
        colors=Dict(
            :curves => [black],
            :points => [TolVibrantBlue, TolVibrantTeal],
            :tvectors => [TolVibrantOrange],
        ),
        camera_position=camera_position,
        arrow_head_size=10.0,
        line_widths=[1.5, 1.5],
        dot_size=4.0,
    )
    render_asymptote(image_prefix * "/Bezier-composite-curve-gradient.asy"; render=2)
end;

# ╔═╡ 5157ac76-d359-4f18-883d-8afbe78d3b4d
PlutoUI.LocalResource(image_prefix * "/Bezier-composite-curve-gradient.png")

# ╔═╡ 7640628b-e232-4ba3-94aa-d8a00218ff6a
md"""
In the following we consider two cases: Interpolation, which fixes the junction and end points of ``B(t)``
and approximation, where a weight and a dataterm are additionally introduced.
"""

# ╔═╡ da970042-8326-4d17-b440-ab1960f10283
md"""
### Interpolation

For interpolation, the junction points are fixed and their gradient entries are hence set to zero.
After transferring to the already mentioned [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html), we can then perform a  [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent.html)  as follows
"""

# ╔═╡ 7db3e138-6ec1-4a0c-8f3b-279f7251b4c3
begin
    #number of sampling points refers to exactness of approximating d^2
    curve_samples = collect(range(0.0, 3.0; length=151))
    pB = get_bezier_points(M, B, :differentiable)
    N = PowerManifold(M, NestedPowerRepresentation(), length(pB))
    function F(M, pB)
        return cost_acceleration_bezier(
            M.manifold, pB, get_bezier_degrees(M.manifold, B), curve_samples
        )
    end
    function gradF(M, pB)
        return grad_acceleration_bezier(
            M.manifold, pB, get_bezier_degrees(M.manifold, B), curve_samples
        )
    end
    x0 = get_bezier_points(M, B, :differentiable)
    pB_opt_ip = gradient_descent(
        N,
        F,
        gradF,
        x0;
        stepsize=ArmijoLinesearch(1.0, ExponentialRetraction(), 0.5, 0.0001),
        stopping_criterion=StopWhenChangeLess(5 * 10.0^(-7)),
    )
end;

# ╔═╡ 5e8e98f6-8896-462e-b610-4b379c727a72
md"and the resut looks like"

# ╔═╡ 50854504-af68-4d7f-999e-f3054c088fb6
render_asy && begin
    B_opt_ip = get_bezier_segments(M, pB_opt_ip, get_bezier_degrees(M, B), :differentiable)
    asymptote_export_S2_signals(
        image_prefix * "/Bezier-IP-Min.asy";
        curves=[de_casteljau(M, B_opt_ip, bezier_pts), de_casteljau(M, B, bezier_pts)],
        points=[get_bezier_junctions(M, B_opt_ip), get_bezier_inner_points(M, B_opt_ip)],
        tangent_vectors=[[
            Tuple(a) for a in zip(
                get_bezier_junctions(M, B_opt_ip, true),
                get_bezier_junction_tangent_vectors(M, B_opt_ip),
            )
        ]],
        colors=Dict(
            :curves => [TolVibrantBlue, black],
            :points => [TolVibrantBlue, TolVibrantTeal],
            :tvectors => [TolVibrantCyan],
        ),
        camera_position=camera_position,
        arrow_head_size=10.0,
        line_widths=[1.5, 0.75, 1.5],
        dot_size=4.0,
    )
    render_asymptote(image_prefix * "/Bezier-IP-Min.asy"; render=2)
end;

# ╔═╡ 4f35b3e8-90ca-4d21-a377-007f93eb0efd
PlutoUI.LocalResource(image_prefix * "/Bezier-IP-Min.png")

# ╔═╡ 203a0538-d608-4d3e-b57d-edec20756300
md"""
Where the original courve is shown in black and the interpolating curve with minimized (discretized) acceleration is shown in blue including its junction points (also blue), tangent vectors (light blue) and control points (teal).
"""

# ╔═╡ 3e43f3fe-5c9a-4fb5-9033-db4a34bac5a5
md"""
### Approximation
Similarly if we introduce the junction points as data fixed given ``d_i`` and set (for simplicity) ``p_i=b_{i,0}`` and ``p_{n+1}=b_{n,4}`` and ``λ=3`` in

```math
\frac{λ}{2}\sum_{k=0}^3 d_{\mathcal M}(d_i,p_i)^2 + A(b),
```

then ``λ`` models how important closeness to the data ``d_i`` is.

Then we obtain the folowing code to minimize the acceleration while approximating the original curve
"""

# ╔═╡ 2f9ef252-5046-4af7-aa69-449a04bab26e
begin
    λ = 3.0
    d = get_bezier_junctions(M, B)
    function F2(M, pB)
        return cost_L2_acceleration_bezier(
            M.manifold, pB, get_bezier_degrees(M.manifold, B), curve_samples, λ, d
        )
    end
    function gradF2(M, pB)
        return grad_L2_acceleration_bezier(
            M.manifold, pB, get_bezier_degrees(M.manifold, B), curve_samples, λ, d
        )
    end
    x1 = get_bezier_points(M, B, :differentiable)
    pB_opt_appr = gradient_descent(
        N,
        F2,
        gradF2,
        x1;
        stepsize=ArmijoLinesearch(1.0, ExponentialRetraction(), 0.5, 0.001),
        stopping_criterion=StopWhenChangeLess(10.0^(-5)),
    )
end

# ╔═╡ 6cfb38a8-c29f-42a6-b642-31c047dd6cb8
render_asy && begin
    B_opt_appr = get_bezier_segments(M, pB_opt_appr, get_bezier_degrees(M, B), :differentiable)
    asymptote_export_S2_signals(
        image_prefix * "/Bezier-Appr-Min.asy";
        curves=[de_casteljau(M, B_opt_appr, bezier_pts), de_casteljau(M, B, bezier_pts)],
        points=[
            d,
            get_bezier_junctions(M, B_opt_appr),
            get_bezier_inner_points(M, B_opt_appr),
        ],
        tangent_vectors=[[
            Tuple(a) for a in zip(
                get_bezier_junctions(M, B_opt_appr, true),
                get_bezier_junction_tangent_vectors(M, B_opt_appr),
            )
        ]],
        colors=Dict(
            :curves => [TolVibrantBlue, black],
            :points => [TolVibrantOrange, TolVibrantBlue, TolVibrantTeal],
            :tvectors => [TolVibrantCyan],
        ),
        camera_position=camera_position,
        arrow_head_size=10.0,
        line_widths=[1.5, 0.75, 1.5],
        dot_size=4.0,
    )
    render_asymptote(image_prefix * "/Bezier-Appr-Min.asy"; render=2)
end;

# ╔═╡ db8110e0-d271-424c-b920-817dc1aab0c9
PlutoUI.LocalResource(image_prefix * "/Bezier-Appr-Min.png")

# ╔═╡ 0855f66e-f9af-4c63-b041-424c9f3fd655
md"""Additionally to the last image, the data points ``d_i`` (junction points of the original curve) are shown in orange, the distance between these and the blue junction points is part of the cost function here."""

# ╔═╡ 67e8e794-a16a-4d39-804a-6bfb350f8247
md"""
The role of ``λ`` can be interpreted as follows: for large values of $λ$, the
minimizer, i.e. the resulting curve, is closer to the original Bézier junction points.
For small ``λ`` the resting curve is closer to a geodesic and the control points are closer to the curve.
For ``λ=0`` _any_ (not necessarily shortest) geodesic is a solution and the problem is ill-posed.
To illustrate the effect of ``λ``, the following image contains 1000 runs for ``λ=10`` in dark currant to ``λ=0.01`` in bright yellow.

![Approximation min Acc](https://manoptjl.org/stable/assets/images/tutorials/Bezier_Approximation_video-Summary-result.png)

The effect of the data term can also be seen in the following video, which starts a little slow and takes about 40 seconds.

![Video of the effect of lambda, the weight of the dataterm](https://manoptjl.org/stable/assets/videos/tutorials/Bezier_Approximation_video-movie.mp4)
"""

# ╔═╡ 446f7276-7ed8-4b5c-a3f1-fdde4fab1479
md"""
## Literature

[^BacakBergmannSteidlWeinmann2016]:
    > Bačák, M., Bergmann, R., Steidl, G. and Weinmann, A.: _A second order nonsmooth
    > variational model for restoring manifold-valued images_,
    > SIAM Journal on Scientific Computations, Volume 38, Number 1, pp. A567–597,
    > doi: [10.1137/15M101988X](https://doi.org/10.1137/15M101988X),
    > arXiv: [1506.02409](https://arxiv.org/abs/1506.02409)
[^BergmannGousenbourger2018]:
    > Bergmann, R. and Gousenbourger, P.-Y.: _A variational model for data fitting on manifolds
    > by minimizing the acceleration of a Bézier curve_.
    > Frontiers in Applied Mathematics and Statistics, 2018.
    > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059),
    > arXiv: [1807.10090](https://arxiv.org/abs/1807.10090)
[^BoumalAbsil2011]:
    > Boumal, N. and Absil, P.-A.: _A discrete regression method on manifolds and its application to data on SO(n)._
    > In: IFAC Proceedings Volumes (IFAC-PapersOnline). Vol. 18. Milano (2011). p. 2284–89.
    > doi: [10.3182/20110828-6-IT-1002.00542](https://dx.doi.org/10.3389/10.3182/20110828-6-IT-1002.00542),
    > web: [www](https://web.math.princeton.edu/~nboumal/papers/Boumal_Absil_A_discrete_regression_method_on_manifolds_and_its_application_to_data_on_SOn.pdf)
[^Casteljau1959]:
    > de Casteljau, P.: _Outillage methodes calcul_, Enveloppe Soleau 40.040 (1959),
    > Institute National de la Propriété Industrielle, Paris.
[^Casteljau1963]:
    > de Casteljau, P.: _Courbes et surfaces à pôles_, Microfiche P 4147-1,
    > André Citroën Automobile SA, Paris, (1963).
[^PopielNoakes2007]:
    > Popiel, T. and Noakes, L.: _Bézier curves and ``C^2`` interpolation in Riemannian
    > manifolds_. Journal of Approximation Theory (2007), 148(2), pp. 111–127.-
    > doi: [10.1016/j.jat.2007.03.002](https://doi.org/10.1016/j.jat.2007.03.002).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Colors = "~0.12.8"
Manifolds = "~0.7.1"
Manopt = "~0.3.14"
PlutoUI = "~0.7.22"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "7e37e10c8226d8b710193f5fde84acba0f25a824"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "265b06e2b1f6a216e0e8f183d28e4d354eab3220"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "bc3930158d2be029e90b7c40d1371c4f54fa04db"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.6"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

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
git-tree-sha1 = "7f3bec11f4bcd01bc1f507ebce5eadf1b0a78f47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.34"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "92243c07e786ea3458532e199eb3feee0e7e08eb"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.4.1"

[[deps.HybridArrays]]
deps = ["LinearAlgebra", "Requires", "StaticArrays"]
git-tree-sha1 = "c1d5b1dcdf2140644e1c6beb9ca09fbed601c241"
uuid = "1baab800-613f-4b0a-84e4-9cd3431bfbb9"
version = "0.4.9"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "a51f46415c844dee694cb8b20a3fcbe6dba342c2"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.1"

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

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Manifolds]]
deps = ["Colors", "Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldsBase", "Markdown", "Random", "RecipesBase", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3811935fd2549b0f5f9b365d6e7173fcdb98ec9a"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.7.1"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown"]
git-tree-sha1 = "77a5949567437d185ee929c405e3c6c0768118ea"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.12.9"

[[deps.Manopt]]
deps = ["ColorSchemes", "ColorTypes", "Colors", "DataStructures", "Dates", "LinearAlgebra", "ManifoldsBase", "Markdown", "Random", "Requires", "SparseArrays", "StaticArrays", "Statistics", "Test"]
git-tree-sha1 = "e87c818e19f79444fff1c77880cdb43b51974dfe"
uuid = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
version = "0.3.14"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

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

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "4f34e3ff2fa7f2a1c03fb2c4fdd637380d760bbc"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.42"

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
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "565564f615ba8c4e4f40f5d29784aa50a8f7bbaf"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.22"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecursiveArrayTools]]
deps = ["ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "c944fa4adbb47be43376359811c0a14757bdc8a8"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.20.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

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
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "e7bc80dc93f50857a5d1e3c8121495852f407e6a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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
# ╟─e701b513-4ad9-474c-9d79-60a24f545ebd
# ╠═6f9be298-5374-11ec-3b9d-fd74a32d6b53
# ╠═087fb979-d68d-4ee7-9be5-fad992fdb792
# ╠═85d8985b-f5ee-4cba-bcc5-528049798330
# ╟─e62e36e9-19ce-4a7d-8bda-5634813c1ee9
# ╠═18d378dd-abc6-4fa6-a2c9-1cc09e4cec46
# ╟─dee4eb65-145c-4b2b-98a7-ae3a7045f24b
# ╠═2b485956-459d-40e7-9364-79ea77073804
# ╟─98c925ac-9ac8-42f3-9855-a01e5eaac350
# ╟─d22e825e-d8ff-44fe-8b06-b14c09765180
# ╠═157d13c8-7311-4832-a9ff-da22125b2bbf
# ╟─5af58241-50c7-464e-bfa3-53819e605793
# ╠═b372fcfc-cad0-48bc-b759-54d15bd2e28b
# ╟─62115e3f-cdfe-464c-91a1-eb80e674a610
# ╠═8653ca13-21e9-4049-a81e-02b6047d2c66
# ╠═05612a12-1080-4b79-b505-198e079a4b2a
# ╟─bdb85fb8-0707-47d1-95ee-9bf2e56500f5
# ╠═38b8f8af-ba81-4d3b-83b7-56b0d41ea708
# ╟─d6759a8e-a705-4d95-a8da-e172ada644d8
# ╟─7c6947a0-7d0b-4459-b999-58ea4f2b0169
# ╠═3f1d0989-2f97-4945-82f8-22e9c7bda6d2
# ╟─41ba8ac2-5a81-46f5-9269-404cc6a1d287
# ╟─e7fc05b3-a198-4a64-b62a-43cdf8f56238
# ╠═7b545a1e-b9c3-4c57-a5ee-3a629407b67a
# ╟─5157ac76-d359-4f18-883d-8afbe78d3b4d
# ╟─7640628b-e232-4ba3-94aa-d8a00218ff6a
# ╟─da970042-8326-4d17-b440-ab1960f10283
# ╠═7db3e138-6ec1-4a0c-8f3b-279f7251b4c3
# ╟─5e8e98f6-8896-462e-b610-4b379c727a72
# ╠═50854504-af68-4d7f-999e-f3054c088fb6
# ╟─4f35b3e8-90ca-4d21-a377-007f93eb0efd
# ╟─203a0538-d608-4d3e-b57d-edec20756300
# ╟─3e43f3fe-5c9a-4fb5-9033-db4a34bac5a5
# ╠═2f9ef252-5046-4af7-aa69-449a04bab26e
# ╠═6cfb38a8-c29f-42a6-b642-31c047dd6cb8
# ╠═db8110e0-d271-424c-b920-817dc1aab0c9
# ╟─0855f66e-f9af-4c63-b041-424c9f3fd655
# ╟─67e8e794-a16a-4d39-804a-6bfb350f8247
# ╟─446f7276-7ed8-4b5c-a3f1-fdde4fab1479
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
