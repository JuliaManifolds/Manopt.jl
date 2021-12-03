### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 6f9be298-5374-11ec-3b9d-fd74a32d6b53
using Colors, Images, Manopt, Manifolds

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
    render_asymptote(image_prefix * "/Casteljau-illustr.asy"; render=2) #src
end

# ╔═╡ bdb85fb8-0707-47d1-95ee-9bf2e56500f5
md"""This image summarizes Casteljaus algorithm:"""

# ╔═╡ 38b8f8af-ba81-4d3b-83b7-56b0d41ea708
load(image_prefix * "/Casteljau-illustr.png")

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
end

# ╔═╡ 41ba8ac2-5a81-46f5-9269-404cc6a1d287
load(image_prefix * "/Bezier-composite-curve.png")

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
end

# ╔═╡ 5157ac76-d359-4f18-883d-8afbe78d3b4d
load(image_prefix * "/Bezier-composite-curve-gradient.png")

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
end

# ╔═╡ 4f35b3e8-90ca-4d21-a377-007f93eb0efd
load(image_prefix * "/Bezier-IP-Min.png")

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
end

# ╔═╡ db8110e0-d271-424c-b920-817dc1aab0c9
load(image_prefix * "/Bezier-Appr-Min.png")

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
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"

[compat]
Colors = "~0.12.8"
Images = "~0.25.0"
Manifolds = "~0.7.1"
Manopt = "~0.3.14"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

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

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

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

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

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

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

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

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "bc3930158d2be029e90b7c40d1371c4f54fa04db"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.6"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

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

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "837c83e5574582e07662bbbba733964ff7c26b9d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.6"

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
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "fe385ec95ac5533650fb9b1ba7869e9bc28cdd0a"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.5"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "3fe985505b4b667e1ae303c9ca64d181f09d5c05"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.3"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2db648b6712831ecb333eae76dbfd1c156ca13bb"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.2"

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

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "7a20463713d239a19cbad3f6991e404aca876bda"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.15"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "15bd05c1c0d5dbb32a9a3d7e0ad2d50dd6167189"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.1"

[[deps.ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "a2951c93684551467265e0e32b577914f69532be"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "ca8d917903e7a1126b6583a097c5cb7a0bedeac1"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.2"

[[deps.ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "5581e18a74a5838bd919294a7138c2663d065238"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.0"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1d2d73b14198d10f7f12bf7f8481fd4b3ff5cd61"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.0"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "36832067ea220818d105d718527d6ed02385bf22"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.7.0"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "d0ac64c9bee0aed6fdbb2bc0e5dfa9a3a78e3acc"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.3"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "b4b161abc8252d68b13c5cc4a5f2ba711b61fec5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.3"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "35dc1cd115c57ad705c7db9f6ef5cc14412e8f00"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.0"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IntegralArrays]]
deps = ["IntervalSets"]
git-tree-sha1 = "4fdfe55b432bbb97adbb0c85c39dd208a7b2bd36"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[deps.JLD2]]
deps = ["DataStructures", "FileIO", "MacroTools", "Mmap", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "46b7834ec8165c541b0b5d1c8ba63ec940723ffb"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.15"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[deps.Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "a51f46415c844dee694cb8b20a3fcbe6dba342c2"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

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

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

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

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "2af69ff3c024d13bde52b34a2a7d6887d4e7b438"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "4f34e3ff2fa7f2a1c03fb2c4fdd637380d760bbc"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.42"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

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

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "6d105d40e30b635cfed9d52ec29cf456e27d38f8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.12"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.Quaternions]]
deps = ["DualNumbers", "LinearAlgebra"]
git-tree-sha1 = "adf644ef95a5e26c8774890a509a55b7791a139f"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

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

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

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

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "dbf5f991130238f10abbf4f2d255fb2837943c43"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.1.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

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

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

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

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "c342ae2abf4902d65a0b0bf59b28506a6e17078a"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.2"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─e701b513-4ad9-474c-9d79-60a24f545ebd
# ╠═6f9be298-5374-11ec-3b9d-fd74a32d6b53
# ╟─087fb979-d68d-4ee7-9be5-fad992fdb792
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
# ╟─38b8f8af-ba81-4d3b-83b7-56b0d41ea708
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
