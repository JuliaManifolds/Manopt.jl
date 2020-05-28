# # [Bezier Curves and their Acceleration](@id BezierCurvesTutorial)
#
# This tutorial illustrates how Bézier curves are generalized to Manfiolds and how to
# minimizer their acceleration, i.e. how to get a curve that is as straight or as geodesic
# while fulfilling constraints
#
# This example also illustrates how to apply the minimization on the corresponding [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html) manifold using a [`gradient_descent`](@ref) with [`ArmijoLinesearch`](@ref).
#
# ## Table of contents
# * [Setup](@ref SetupTB)
# * [de Casteljau algorithm on manifolds](@ref Casteljau)
# * [Composite Bézire Curves](@ref CompositeBezier)
# * [Minimizing the Acceleration of a Bézier curve](@ref MinAccBezier)
# * [Literature](@ref LiteratureBT)
#
# ## [Setup](@id SetupTB)
#
# We first initialize the necessary packages
exportFolder = joinpath(@__DIR__,"..","..","docs","src","assets","images","tutorials") #src
using Manopt, Manifolds
# and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)
using Colors
black = RGBA{Float64}(colorant"#000000")
TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
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
# To evaluate a Bézier curve knowing its [`BezierSegment`](@ref), use [`de_casteljau`](@ref).
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
# We define the composite Bézier curve by $B(t) = \begin{cases} c(t) & \text{ if } 0\leq t < 1, \\ d(t-1) & \text{ if } 1\leq t \leq 2,\end{cases}$ where $t\in[0,2]$.
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
# The motivation to minimize the acceleration of the composite Bézier curve is, that the curve should get “straighter” or more geodesic like.
# If we discretize the curve $B(t)$ with its control points denoted by $b_{i,j}$ for the $j$th note in the $i$th segment, the discretized model for equispaced $t_i$, $i=0,\ldots,N$ in the domain of $B$ reads[^BergmannGousenbourger2018]
#
# $A(b) \coloneqq\sum_{i=1}^{N-1}\frac{\mathrm{d}^2_2 \bigl[ B(t_{i-1}), B(t_{i}), B(t_{i+1}) \bigr]}{\Delta_t^3},$
#
# where $\mathrm{d}_2$ denotes the second order finite difference using the mid point approach, see [`costTV2`](@ref)[^BacakBergmannSteidlWeinmann2016],
#
# $d_2(x,y,z) := \min_{c ∈ \mathcal C_{x,z}} d_{\mathcal M}(c,y),\qquad x,y,z∈\mathcal M.$
#
# Another model is based on logarithmic maps, see [^BoumalAbsil2011], but that is not considered here.
# An advantage of the model considered here is, that it only consist of the evaluation of geodesics.
# This yields a gradient of $A(b)$ with respect to $b$ [`adjoint_Jacobi_field`](@ref)s. The following image shows the negative gradient (scaled)
#
gradFullB = Manopt._∇acceleration_bezier(M, get_bezier_points(M, B, :differentiable), [3,3,3], collect(range(0.0,3.0,length=151))) # src
asymptote_export_S2_signals(exportFolder*"/Bezier-composite-curve-gradient.asy"; #src
    curves = [de_casteljau(M, B, bezier_pts),], #src
    points = [get_bezier_junctions(M, B), get_bezier_inner_points(M, B)], #src
    tVectors = [[ #src
        Tuple(a) #src
        for #src
        a in #src
        zip(get_bezier_points(M, B, :continuous), -0.05 .* get_bezier_points(M,gradFullB, :continuous)) #src
    ]], #src
    colors = Dict( #src
        :curves => [black], #src
        :points => [TolVibrantBlue, TolVibrantTeal], #src
        :tvectors => [TolVibrantOrange], #src
    ), #src
    cameraPosition = cameraPosition, #src
    arrowHeadSize = 10.0, #src
    lineWidths = [1.5, 1.5], #src
    dotSize = 4.0, #src
) #src
render_asymptote(exportFolder*"/Bezier-composite-curve-gradient.asy"; render = 2) #src
#md # ![Illustration of the gradient of the acceleration with respect to the control points.](../assets/images/tutorials/Bezier-composite-curve-gradient.png)
#
# In the following we consider two cases: Interpolation, which fixes the junction and end points of $B(t)$
# and approximation, where a weight and a dataterm are additionally introduced.
#
# ### Interpolation
# For interpolation, the junction points are fixed and their gradient entries are hence set to zero.
# After transferring to the already mentioned [`PowerManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/power.html), we can then perform a [`gradient_descent`](@ref) as follows
curve_samples = collect(range(0.0,3.0,length=151)) #exactness of approximating d^2
pB = get_bezier_points(M, B, :differentiable)
N = PowerManifold(M, NestedPowerRepresentation(), length(pB))
F(pB) = cost_acceleration_bezier(M, pB, get_bezier_degrees(M, B), curve_samples)
∇F(pB) = ∇acceleration_bezier(M, pB, get_bezier_degrees(M, B), curve_samples)
x0 = pB
pB_opt_ip = gradient_descent(N,F, ∇F, x0;
    stepsize = ArmijoLinesearch(1.0, ExponentialRetraction(), 0.5, 0.0001),
    stopping_criterion = StopWhenChangeLess(5* 10.0^(-7)),
    debug = [:Iteration, " | ", :Cost, " | ", DebugGradientNorm()," | ", DebugStepsize(),
        " | ", :Change, "\n",:Stop,10],
)
#
# and the result looks like
#
B_opt_ip = get_bezier_segments(M, pB_opt_ip, get_bezier_degrees(M, B), :differentiable) #src
asymptote_export_S2_signals(exportFolder*"/Bezier-IP-Min.asy"; #src
    curves = [de_casteljau(M, B_opt_ip, bezier_pts), de_casteljau(M, B, bezier_pts)], #src
    points = [get_bezier_junctions(M, B_opt_ip), get_bezier_inner_points(M, B_opt_ip)], #src
    tVectors = [[Tuple(a) for a in zip( #src
            get_bezier_junctions(M, B_opt_ip, true), #src
            get_bezier_junction_tangent_vectors(M, B_opt_ip), #src
        ) #src
    ]], #src
    colors = Dict( #src
        :curves => [TolVibrantBlue, black], #src
        :points => [TolVibrantBlue, TolVibrantTeal], #src
        :tvectors => [TolVibrantCyan], #src
    ), #src
    cameraPosition = cameraPosition, #src
    arrowHeadSize = 10.0, #src
    lineWidths = [1.5, 0.75, 1.5], #src
    dotSize = 4.0, #src
) #src
render_asymptote(exportFolder*"/Bezier-IP-Min.asy"; render = 2) #src
#md # ![Interpolation Min Acc](../assets/images/tutorials/Bezier-IP-Min.png)
#
# ### Approximation
# Similarly if we introduce the junction points as data fixed given $d_i$ and set (for simplicity) $p_i=b_{i,0}$ and $p_{n+1}=b_{n,4}$
# and set $λ=3$ in
#
# $\frac{\lambda}{2}\sum_{k=0}^3 d_{\mathcal M}(d_i,p_i)^2 + A(b)$,
#
# then $λ$ models how important closeness to the data $d_i$ is.
#
λ = 3.0
d = get_bezier_junctions(M, B)
F(pB) = cost_L2_acceleration_bezier(M, pB, get_bezier_degrees(M, B), curve_samples, λ, d)
∇F(pB) = ∇L2_acceleration_bezier(M, pB, get_bezier_degrees(M, B), curve_samples, λ, d)
x0 = pB
pB_opt_appr = gradient_descent(N, F, ∇F, x0;
    stepsize = ArmijoLinesearch(1.0, ExponentialRetraction(), 0.5, 0.001),
    stopping_criterion = StopWhenChangeLess(10.0^(-5)),
    debug = [:Iteration, " | ", :Cost, " | ", DebugGradientNorm()," | ", DebugStepsize(),
        " | ", :Change, "\n",:Stop,50],
)
#
# and the result looks like
#
B_opt_appr = get_bezier_segments(M, pB_opt_appr, get_bezier_degrees(M, B), :differentiable) #src
asymptote_export_S2_signals(exportFolder*"/Bezier-Appr-Min.asy"; #src
    curves = [de_casteljau(M, B_opt_appr, bezier_pts), de_casteljau(M, B, bezier_pts)], #src
    points = [d,get_bezier_junctions(M, B_opt_appr), get_bezier_inner_points(M, B_opt_appr)], #src
    tVectors = [[Tuple(a) for a in zip( #src
            get_bezier_junctions(M, B_opt_appr, true), #src
            get_bezier_junction_tangent_vectors(M, B_opt_appr), #src
        ) #src
    ]], #src
    colors = Dict( #src
        :curves => [TolVibrantBlue, black], #src
        :points => [TolVibrantOrange, TolVibrantBlue, TolVibrantTeal], #src
        :tvectors => [TolVibrantCyan], #src
    ), #src
    cameraPosition = cameraPosition, #src
    arrowHeadSize = 10.0, #src
    lineWidths = [1.5, 0.75, 1.5], #src
    dotSize = 4.0, #src
) #src
render_asymptote(exportFolder*"/Bezier-Appr-Min.asy"; render = 2) #src
#md # ![Approximation min Acc](../assets/images/tutorials/Bezier-Appr-Min.png)

# ## [Literature](@id LiteratureBT)
#
# [^BacakBergmannSteidlWeinmann2016]:
#     > Bačák, M, Bergmann, R., Steidl, G. and Weinmann, A.: _A second order nonsmooth
#     > variational model for restoring manifold-valued images_,
#     > SIAM Journal on Scientific Computations, Volume 38, Number 1, pp. A567–597,
#     > doi: [10.1137/15M101988X](https://doi.org/10.1137/15M101988X),
#     > arXiv: [1506.02409](https://arxiv.org/abs/1506.02409)
# [^BergmannGousenbourger2018]:
#     > R. Bergmann, P.-Y. Gousenbourger: _A variational model for data fitting on manifolds
#     > by minimizing the acceleration of a Bézier curve_.
#     > Frontiers in Applied Mathematics and Statistics, 2018.
#     > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059)
# [^BoumalAbsil2011]:
#     > Boumal, N. and Absil, P.-A.: _A discrete regression method on manifolds and its application to data on SO(n)._
#     > In: IFAC Proceedings Volumes (IFAC-PapersOnline). Vol. 18. Milano (2011). p. 2284–89.
#     > doi: [10.3182/20110828-6-IT-1002.00542](https://dx.doi.org/10.3389/10.3182/20110828-6-IT-1002.00542)
#     > [www](https://web.math.princeton.edu/~nboumal/papers/Boumal_Absil_A_discrete_regression_method_on_manifolds_and_its_application_to_data_on_SOn.pdf)
# [^Casteljau1959]:
#     > de Casteljau, P.: _Outillage methodes calcul_, Enveloppe Soleau 40.040 (1959),
#     > Institute National de la Propriété Industrielle, Paris.
# [^Casteljau1963]:
#     > de Casteljau, P.: _Courbes et surfaces à pôles_, Microfiche P 4147-1,
#     > André Citroën Automobile SA, Paris, (1963).
# [^PopielNoakes2007]:
#     > Popiel, T. and Noakes, L.: _Bézier curves and $C^2$ interpolation in Riemannian
#     > manifolds_. Journal of Approximation Theory (2007), 148(2), pp. 111–127.-
#     > doi: [10.1016/j.jat.2007.03.002](https://doi.org/10.1016/j.jat.2007.03.002).
