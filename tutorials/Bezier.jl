### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 2a6ed8b8-aa64-44a0-a1f6-f130cdff037b
using Pkg;

# ╔═╡ 6f9be298-5374-11ec-3b9d-fd74a32d6b53
using Colors, PlutoUI, Manopt, Manifolds

# ╔═╡ e701b513-4ad9-474c-9d79-60a24f545ebd
md"""
# Bézier Curves and Their Acceleration

This tutorial illustrates how Bézier curves are generalized to manifolds and how to
minimize their acceleration, i.e. how to get a curve that is as straight or as geodesic as possible
while fulfilling constraints.

This example also illustrates how to apply the minimization on the corresponding [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html) manifold using a [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent.html) with [`ArmijoLinesearch`](https://manoptjl.org/stable/plans/index.html#Manopt.ArmijoLinesearch).
"""

# ╔═╡ bb71134f-703e-48ac-b49a-36624dbdc272
md"Let's first again load the necessary packages and decide whether we want to use the local Pluto package management or the current (global) environment"

# ╔═╡ acc6f2d0-2937-4056-9c7f-43857c77f772
use_local = false

# ╔═╡ d2a19756-ae80-416b-96d0-ebfb25a3cc32
use_local || Pkg.activate()

# ╔═╡ 087fb979-d68d-4ee7-9be5-fad992fdb792
md"We define some colors from [Paul Tol](https://personal.sron.nl/~pault/)"

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
    #@info image_prefix
end;

# ╔═╡ dee4eb65-145c-4b2b-98a7-ae3a7045f24b
md"""
We finally load our data, see [`artificial_S2_composite_bezier_curve`](https://manoptjl.org/stable/helpers/data.html#Manopt.artificial_S2_composite_bezier_curve-Tuple{}), a composite Bézier curve consisting of 3 segments on the Sphere. The middle segment consists of the control points
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
On Euclidean spaces, Bézier curves of these ``n=4`` so-called control points like this segment yield polynomials of degree ``3``.
The resulting curve ``γ: [0,1] → ℝ^m`` is called [Bézier curve](https://en.wikipedia.org/wiki/Bézier_curve) or Bézier spline and is named after [Pierre Bézier](https://en.wikipedia.org/wiki/Pierre_Bézier) (1910–1999).
They can be evaluated by the de Casteljau algorithm by evaluating line segments between points.
While it is not easy to evaluate polynomials on a manifold, evaluating line segments generalizes to the evaluation of [`shortest_geodesic`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.shortest_geodesic-Tuple{Manifold,Any,Any})s.
We will illustrate this using these points ``b=(b_1,b_2,b_3,b_4)`` on the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/sphere.html) ``\mathbb S^2``.
Let's evaluate this at the point ``t=\frac{1}{4}∈[0,1]``. We first compute
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
From the control points (blue) and their geodesics, one evaluation per geodesic yields three interim points (cyan), their two successive geodesics yield two other points (teal), and on the geodesic that connects these two at ``t=0.66`` we obtain the point on the curve.

In Manopt.jl, to evaluate a Bézier curve knowing its [`BezierSegment`](https://manoptjl.org/stable/functions/bezier.html#Manopt.BezierSegment), use [`de_casteljau`](https://manoptjl.org/stable/functions/bezier.html#Manopt.de_casteljau-Tuple{AbstractManifold,%20Vararg{Any,%20N}%20where%20N}).

There are a few nice observations to make, that also hold for these Bézier curves on manifolds:
* The curve starts in the first controlpoint ``b_0`` and ends in the last controlpoint ``b_3``.
* The tangent vector to the curve at the start ``\dot c(0)`` is equal to ``\frac{1}{3}\log_{b_0}b_1 = \dot γ_{b_0,b_1}(0)``, where ``γ_{a,b}`` denotes the shortest geodesic between ``a`` and ``b``.
* The tangent vector to the curve at the end ``\dot c(1)`` is equal to ``-\frac{1}{3}\log_{b_3}b_2 = -γ_{b_3,b_2}(0) = \dot γ_{b_2,b_3}(1)``.
* The curve is differentiable.

For more details on these properties, see for example [^PopielNoakes2007].
"""

# ╔═╡ 7c6947a0-7d0b-4459-b999-58ea4f2b0169
md"""
## Composite Bézier Curves

With the properties of a single Bézier curve, also called Bézier segment, we can “stitch” curves together. Let ``a_0,…,a_n`` and ``b_0,…,b_m`` be two sets of controlpoints for the Bézier segments ``c(t)`` and ``d(t)``, respectively.
We define the composite Bézier curve by ``B(t) = \begin{cases} c(t) & \text{ if } 0\leq t < 1, \\ d(t-1) & \text{ if } 1\leq t \leq 2,\end{cases}`` where ``t∈[0,2]``.
This can of course be generalized straightforwardly to more than two cases.
With the properties from the previous section we can now state that

* the curve ``B(t)`` is continuous if ``c(1)=d(0)`` or in other words ``a_n=b_0``
* the curve ``B(t)`` is differentiable if additionally ``\dot c(1)=\dot d(0)``, or in other words, ``-\log_{a_n}a_{n-1} = \log_{b_0}b_1``. This is equivalent to ``a_n=b_0 = \gamma_{a_{n-1}b_1}(\tfrac{1}{2})``.

One nice interpretation of the last characterization is that the tangents ``\log_{a_n}a_{n-1}`` and ``\log_{b_0}b_1`` point towards opposite directions.
For a continuous curve, the first point of every segment (except for the first segment) can be ommitted. For a differentiable curve the first two points (except for the first segment) can be ommitted.
You can reduce storage by calling [`get_bezier_points`](https://manoptjl.org/stable/functions/bezier.html#Manopt.get_bezier_points), though for a construction with [`get_bezier_segments`](https://manoptjl.org/stable/functions/bezier.html#Manopt.get_bezier_segments-Union{Tuple{P},%20Tuple{AbstractManifold,%20Vector{P},%20Any},%20Tuple{AbstractManifold,%20Vector{P},%20Any,%20Symbol}}%20where%20P) you also need [`get_bezier_degrees`](https://manoptjl.org/stable/functions/bezier.html#Manopt.get_bezier_degrees-Tuple{AbstractManifold,%20AbstractVector{var%22#s56%22}%20where%20var%22#s56%22%3C:BezierSegment}).
The reduced storage is represented as an array of points, i.e. an element of the corresponding [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html).

For the three-segment example from the beginning this looks as follows
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
## Minimizing the Acceleration of a Composite Bézier Curve
The motivation to minimize the acceleration of a composite Bézier curve is that the curve should get “straighter” or more geodesic-like.
If we discretize the curve ``B(t)`` with its control points denoted by ``b_{i,j}`` for the ``j``th note in the ``i``th segment, the discretized model for equispaced ``t_i``, ``i=0,…,N`` in the domain of ``B`` reads[^BergmannGousenbourger2018]

```math
A(b) :eqq\sum_{i=1}^{N-1}\frac{\mathrm{d}^2_2 \bigl[ B(t_{i-1}), B(t_{i}), B(t_{i+1}) \bigr]}{\Delta_t^3},
```

where $\mathrm{d}_2$ denotes the second order finite difference using the mid point approach, see [`costTV2`](@ref)[^BacakBergmannSteidlWeinmann2016],

```math
d_2(x,y,z) := \min_{c ∈ \mathcal C_{x,z}} d_{\mathcal M}(c,y),\qquad x,y,z∈\mathcal M.
```

Another model is based on logarithmic maps, see [^BoumalAbsil2011], but that is not considered here.
An advantage of the model considered here is that it only consist of the evaluation of geodesics.
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
In the following we consider two cases: interpolation, which fixes the junction and end points of ``B(t)``,
and approximation, where a weight and a data term are additionally introduced.
"""

# ╔═╡ da970042-8326-4d17-b440-ab1960f10283
md"""
### Interpolation

For interpolation, the junction points are fixed and their gradient entries are hence set to zero.
After transferring to the aforementioned [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html), we can then perform a  [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent.html)  as follows
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
        stepsize=ArmijoLinesearch(M; contraction_factor=0.5, sufficient_decrease=0.0001),
        stopping_criterion=StopWhenChangeLess(5 * 10.0^(-7)),
    )
end;

# ╔═╡ 5e8e98f6-8896-462e-b610-4b379c727a72
md"and the result looks like"

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
Where the original curve is shown in black and the interpolating curve with minimized (discretized) acceleration is shown in blue including its junction points (also blue), tangent vectors (light blue) and control points (teal).
"""

# ╔═╡ 3e43f3fe-5c9a-4fb5-9033-db4a34bac5a5
md"""
### Approximation
Similarly, if we introduce the junction points as fixed data given ``d_i`` and set (for simplicity) ``p_i=b_{i,0}`` and ``p_{n+1}=b_{n,4}`` and ``λ=3`` in

```math
\frac{λ}{2}\sum_{k=0}^3 d_{\mathcal M}(d_i,p_i)^2 + A(b),
```

then ``λ`` models how important closeness to the data ``d_i`` is.

Then we obtain the following code to minimize the acceleration while approximating the original curve
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
        stepsize=ArmijoLinesearch(M; contraction_factor=0.5, sufficient_decrease=0.0001),
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
md"""Additionally to the last image, the data points ``d_i`` (junction points of the original curve) are shown in orange. The distance between these and the blue junction points is part of the cost function here."""

# ╔═╡ 67e8e794-a16a-4d39-804a-6bfb350f8247
md"""
The role of ``λ`` can be interpreted as follows: for large values of $λ$, the
minimizer, i.e. the resulting curve, is closer to the original Bézier junction points.
For small ``λ`` the resulting curve is closer to a geodesic and the control points are closer to the curve.
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

# ╔═╡ Cell order:
# ╟─e701b513-4ad9-474c-9d79-60a24f545ebd
# ╠═bb71134f-703e-48ac-b49a-36624dbdc272
# ╠═acc6f2d0-2937-4056-9c7f-43857c77f772
# ╠═2a6ed8b8-aa64-44a0-a1f6-f130cdff037b
# ╠═d2a19756-ae80-416b-96d0-ebfb25a3cc32
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
# ╠═38b8f8af-ba81-4d3b-83b7-56b0d41ea708
# ╟─d6759a8e-a705-4d95-a8da-e172ada644d8
# ╟─7c6947a0-7d0b-4459-b999-58ea4f2b0169
# ╠═3f1d0989-2f97-4945-82f8-22e9c7bda6d2
# ╟─41ba8ac2-5a81-46f5-9269-404cc6a1d287
# ╟─e7fc05b3-a198-4a64-b62a-43cdf8f56238
# ╠═7b545a1e-b9c3-4c57-a5ee-3a629407b67a
# ╠═5157ac76-d359-4f18-883d-8afbe78d3b4d
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
