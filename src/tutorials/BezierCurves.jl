# # [Bezier Curves and their Acceleration](@id BezierCurvesTutorial)
#
# This tutorial illustrates how Bézier curves are generalized to Manfiolds and how to
# minimizer their acceleration, i.e. how to get a curve that is as straight or as geodesic
# while fulfilling constraints
#
# This example also illustrates the `PowerManifold` manifold as well
# as [`ArmijoLinesearch`](@ref).
# ## Table of contents
# * [Setup](@ref SetupTB)
# * [de Casteljau algorithm on manifolds](@ref Casteljau)
# * [Composite Bézire Curves](@ref CompositeBezier)
# * [Minimizing the Acceleration of a Bézier curve](@ref MinAccBezier)
# * [Literature](@ref LiteratureBT)
#
# ## [Setup](@id SetupTB)
# We first initialize the necessary packages
exportFolder = joinpath(@__DIR__,"..","..","docs","src","assets","images","tutorials") #src
using Manopt, Manifolds
# and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)
using Colors
black = RGBA{Float64}(colorant"#000000")
TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE")
TolVibrantTeal = RGBA{Float64}(colorant"#009988")
geo_pts = collect(range(0.0,1.0,length=101)) #hide
bezier_pts = collect(range(0.0,3.0,length=201)) #hide
cameraPosition = (-1.0, -0.7, 0.3) #hide
nothing #hide
# Then we load our data, a composite Bezier curve consisting of 3segments on the Sphere
B = artificial_S2_composite_bezier_curve();
#
# ## [De Casteljau algorithm on manifolds](@id Casteljau)
# This curve can be evaluated using de [Casteljau's algorithm](https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm)[^Casteljau1959][^Casteljau1963] named after [Paul de Casteljau](https://en.wikipedia.org/wiki/Paul_de_Casteljau)(*1930).
# To simplify the idea and understand this algorithm, we will first only look at the points of the first segment
M = Sphere(2)
b = B[2].pts
# On Euclidean spaces Bézier curves of these $n=4$ so called control points like this segment yield polynomials of degree $3$.
# The resulting curve $\gamma: [0,1] \to \mathbb R^m$ is called [Bezier curve](https://en.wikipedia.org/wiki/Bézier_curve) or Bézier spline and is named after [Piérre Bezier](https://en.wikipedia.org/wiki/Pierre_Bézier) (1910–1999).
# They can be evaluated by the de Casteljau algorithm by evaluating line segments between points.
# While it is not easy to evaluate polynomials on a manifold, evaluating line segments generalizes to the evaluation of [`shortest_geodesic`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.shortest_geodesic-Tuple{Manifold,Any,Any})s
#
# We will illustrate this using these points $b=(b_1,b_2,b_3,b_4)$ on the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/sphere.html) $\mathbb S^2$.
# Let's evaliuate this at the point $t=\frac{1}{4}\in [0,1]$. We first compute
t=0.66
pts1 = shortest_geodesic.(Ref(M), b[1:3], b[2:4],Ref(t))
# We obtain 3 points on the geodesics connecting the control points. Repeating this again twice
pts2 = shortest_geodesic.(Ref(M), pts1[1:2], pts1[2:3],Ref(t))
p = shortest_geodesic(M, pts2[1], pts2[2],t)
# we obtain the point on the Bézier curve $c(t)$.
# This procedure is illustrated in the following image:
#
curves1 = [shortest_geodesic(M, b[i], b[i+1],geo_pts) for i=1:3] #src
curves2 = [shortest_geodesic(M, pts1[i], pts1[i+1],geo_pts) for i=1:2] #src
curves3 = [shortest_geodesic(M, pts2[1], pts2[2],geo_pts),] #src
asymptote_export_S2_signals(exportFolder*"/Casteljau-illustr.asy"; #src
    points = [b, pts1, pts2, [p,]], #src
    curves = [curves1..., curves2..., curves3..., de_casteljau(M,B[2],geo_pts)], #src
    colors=Dict(:points => [TolVibrantBlue, TolVibrantCyan, TolVibrantTeal, TolVibrantOrange], #src
                :curves => [TolVibrantBlue, TolVibrantBlue, TolVibrantBlue, TolVibrantCyan, TolVibrantCyan, TolVibrantTeal, black] #src
                ), #src
    dotSize = 3.5, lineWidths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5], cameraPosition = cameraPosition #src
) #src
render_asymptote(exportFolder*"/Casteljau-illustr.asy"; render=2) #src
#md # ![Illustration of de Casteljau's algorithm on the Sphere.](../assets/images/tutorials/Casteljau-illustr.png)
#
# To evaluate a Bézier curve knowing its [`BezierSegment`](@ref), use [`de_casteljau`](@ref)
#
# From the control points (blue) and their geodesics, ont evaluation per geodesic yields three interims points (cyan), their two successive geodeics another two points (teal) and at its geodesic at $t=0.66$ we obtain the point on the curve.
# There are a few nice observations to make, that hold also for these Bézier curves on manifolds:
# * The curve starts in the first controlpoint $b_0$ and ends in the last controlpoint $b_3$
# * The tangent vector to the curve at the start $\dot c(0)$ is equal to $\log_{b_0}b_1 = \dot\gamma_{b_0,b_0}(0)$, where $\gamma_{a,b}$ denotes the shortest geodesic.
# * The tangent vector to the curve at the end $\dot c(1)$ is equal to $-\log_{b_3}b_2 = -\dot\gamma_{b_3,b_2}(0) = \dot\gamma_{b_2,b_3}(1)$.
# * the curve is differentiable.
#
# For more details on these properties, see for example [^PopielNoakes2007].
# ## [Composite Bézier curves](@id CompositeBezier)
# With the properties of a single Bézier curve, also called Bézier segment, we can “stitch” curves together. Let $a_0,\ldots,a_n$ and $b_0,\ldots,b_m$ be two sets of controlpoints for the Bézier segments $c(t)$ and $d(t)$, respectively.
# We define the composite Bézier curve by $B(t) = \begin{cases} c(t) & \text{ if } 0\leq t < 1, \\ d(t-1) & \text{ if } 1\leq t \leq 2\end{cases}$ where $t\in[0,2]$.
# This can of course be generalised straight forward to more than two cases.
# With the properties from the previous section we can now state that
#
# * the curve $B(t)$ is continuous if $c(1)=d(0)$ or in other words $a_n=b_0$
# * the curve $B(t)$ is differentiable if additionally $\cdot c(1)=\cdot d(0)$ or in other words $-\log_{a_n}a_{n-1} = \log_{b_0}b_1$. This is equivalent to $a_n=b_0 = \gamma_{a_{n-1}b_1}(\tfrac{1}{2})$.
#
# One nice interpretation of the last characterization is, that the tangents $\log_{a_n}a_{n-1}$ and $\log_{b_0}b_1$ point into opposite directions.
# For a continuous curve, the first point of every segment (except for the first segment) can be ommitted, for a differentiable curve the first two points (except for the first segment) can be ommitted.
# You can reduce storage by calling [`get_bezier_points`](@ref), though for econstruciton with [`get_bezier_segments`](@ref) you also need [`get_bezier_degrees`](@ref).
# The reduced storage is represented as an array of points, i.e. an element of the corresponding [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html).
#
# For the three segment example from the beginning this looks as follows:
#
asymptote_export_S2_signals(exportFolder*"/Bezier-composite-curve.asy"; #src
    curves = [de_casteljau(M, B, bezier_pts),], #src
    points = [get_bezier_junctions(M, B), get_bezier_inner_points(M, B)], #src
    tVectors = [[ #src
        Tuple(a) #src
        for #src
        a in #src
        zip(get_bezier_junctions(M, B, true), get_bezier_junction_tangent_vectors(M, B)) #src
    ]], #src
    colors = Dict( #src
        :curves => [black], #src
        :points => [TolVibrantBlue, TolVibrantTeal], #src
        :tvectors => [TolVibrantCyan], #src
    ), #src
    cameraPosition = cameraPosition, #src
    arrowHeadSize = 10.0, #src
    lineWidths = [1.5, 1.5], #src
    dotSize = 4.0, #src
) #src
render_asymptote(exportFolder*"/Bezier-composite-curve.asy"; render = 2) #src
#md # ![Illustration of a differentiable composite Bézier curve with 3 segments.](../assets/images/tutorials/Bezier-composite-curve.png)
#
# ## [Minimizing the acceleration of a composite Bézier curve](@id MinAccBezier)
# See [^BergmannGousenbourger2018].
#
# ## [Literature](@id LiteratureBT)
# [^BergmannGousenbourger2018]:
#     > R. Bergmann, P.-Y. Gousenbourger: _A variational model for data fitting on manifolds
#     > by minimizing the acceleration of a Bézier curve_.
#     > Frontiers in Applied Mathematics and Statistics, 2018.
#     > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059)
# [^Casteljau1959]:
#     > de Casteljau, P.: Outillage methodes calcul, Enveloppe Soleau 40.040 (1959),
#     > Institute National de la Propriété Industrielle, Paris.
# [^Casteljau1963]:
#     > de Casteljau, P.: Courbes et surfaces à pôles, Microfiche P 4147-1,
#     > André Citroën Automobile SA, Paris, (1963).
# [^PopielNoakes2007]:
#     > Popiel, T. and Noakes, L.: Bézier curves and $C^2$ interpolation in Riemannian
#     > manifolds. Journal of Approximation Theory (2007), 148(2), pp. 111–127.-
#     > doi: [10.1016/j.jat.2007.03.002](https://doi.org/10.1016/j.jat.2007.03.002).
