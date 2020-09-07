# # Illustration of Jacobi Fields
#
# This tutorial illustrates the usage of Jacobi Fields within
# `Manopt.jl`.
# For this tutorial you should be familiar with the basic terminology on a
# manifold like the exponential and logarithmic map as well as
# [shortest geodesic](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.shortest_geodesic-Tuple{Manifold,Any,Any})s.
#
# We first initialize the manifold
exportFolder = #src
    joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "images", "tutorials") #src
using Manopt, Manifolds
# and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)
using Colors
black = RGBA{Float64}(colorant"#000000")
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE")
TolVibrantTeal = RGBA{Float64}(colorant"#009988")
nothing #hide
# Assume we have two points on the equator of the [Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) $\mathcal M = \mathbb S^2$
M = Sphere(2)
p, q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
# their connecting [shortest geodesic](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.shortest_geodesic-Tuple{Manifold,Any,Any}) (sampled at `100` points)
geodesicCurve = shortest_geodesic(M, p, q, [0:0.1:1.0...]);
nothing #hide
# looks as follows using the [`asymptote_export_S2_signals`](@ref) export
asymptote_export_S2_signals( #src
    exportFolder * "/jacobiGeodesic.asy"; #src
    curves = [geodesicCurve], #src
    points = [[p, q]], #src
    colors = Dict(:curves => [black], :points => [TolVibrantOrange]), #src
    dotSize = 3.5, #src
    lineWidth = 0.75, #src
    cameraPosition = (1.0, 1.0, 0.5), #src
)#src
render_asymptote(exportFolder * "/jacobiGeodesic.asy"; render = 2) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("jacobiGeodesic.asy";
#md #     render = asyResolution,
#md #     curves=[geodesicCurve], points = [ [x,y] ],
#md #     colors=Dict(:curves => [black], :points => [TolVibrantOrange]),
#md #     dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.,1.,.5)
#md # )
#md # render_asymptote("jacobiGeodesic.asy"; render = 2)
#md # ```
#md #
#md # ![A geodesic connecting two points on the equator](../assets/images/tutorials/jacobiGeodesic.png)
#
# where $x$ is on the left. Then this tutorial solves the following task:
#
# Given a direction $X_p∈ T_x\mathcal M$, for example
X = [0.0, 0.4, 0.5]
# we move the start point $x$ into, how does any point on the geodesic move?
#
# Or mathematically: Compute $D_p g(t; p,q)$ for some fixed $t∈[0,1]$
# and a given direction $X_p$.
# Of course two cases are quite easy: For $t=0$ we are in $x$ and how $x$ “moves”
# is already known, so $D_x g(0;p,q) = X$. On the other side, for $t=1$,
# $g(1; p,q) = q$ which is fixed, so $D_p g(1; p,q)$ is the zero tangent vector
# (in $T_q\mathcal M$).
#
# For all other cases we employ a [`jacobi_field`](@ref), which is a (tangent)
# vector field along the [shortest geodesic](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.shortest_geodesic-Tuple{Manifold,Any,Any}) given as follows: The _geodesic variation_
# $\Gamma_{g,X}(s,t)$ is defined for some $\varepsilon > 0$ as
# ````math
# \Gamma_{g,X}(s,t):=\exp{\gamma_{p,X}(s)}[t\log_{g(s;p,X)}p],\qquad s∈(-\varepsilon,\varepsilon),\ t∈[0,1].
# ````
# Intuitively we make a small step $s$ into direction $\xi$ using the geodesic
# $g(\cdot; p,X)$ and from $r=g(s; p,X)$ we follow (in $t$) the geodesic
# $g(\cdot; r,q)$. The corresponding Jacobi field~\(J_{g,X}\)
# along~\(g(\cdot; p,q)\) is given
#
# ````math
# J_{g,X}(t):=\frac{D}{\partial s}\Gamma_{g,X}(s,t)\Bigl\rvert_{s=0}$
# ````
#
# which is an ODE and we know the boundary conditions $J_{g,X}(0)=X$ and
# $J_{g,X}(t) = 0$. In symmetric spaces we can compute the solution, since the
# system of ODEs decouples, see for example [do Carmo](#doCarmo1992),
# Chapter 4.2. Within `Manopt.jl` this is implemented as
# [`jacobi_field`](@ref)`(M,p,q,t,X[,β])`, where the optional parameter (function)
# `β` specifies, which Jacobi field we want to evaluate and the one used here is
# the default.
#
# We can hence evaluate that on the points on the geodesic at
T = [0:0.1:1.0...]
nothing #hide
# namely
r = shortest_geodesic(M, p, q, T)
nothing #hide
# the geodesic moves as
W = jacobi_field.(Ref(M), Ref(p), Ref(q), T, Ref(X))
# which can also be called using [`differential_geodesic_startpoint`](@ref).
# We can add to the image above by creating extended tangent vectors
# the include their base points
V = [Tuple([a, b]) for (a, b) in zip(r, W)]
# and add that as one further set to the Asymptote export.
asymptote_export_S2_signals( #src
    exportFolder * "/jacobiGeodesicdifferential_geodesic_startpoint.asy"; #src
    curves = [geodesicCurve], #src
    points = [[p, q], r], #src
    tVectors = [V], #src
    colors = Dict( #src
        :curves => [black], #src
        :points => [TolVibrantOrange, TolVibrantCyan], #src
        :tvectors => [TolVibrantCyan], #src
    ), #src
    dotSizes = [3.5, 2.0], #src
    lineWidth = 0.75, #src
    cameraPosition = (1.0, 1.0, 0.5), #src
) #src
render_asymptote( #src
    exportFolder * "/jacobiGeodesicdifferential_geodesic_startpoint.asy"; #src
    render = 2, #src
) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("jacobiGeodesicdifferential_geodesic_startpoint.asy";
#md #     render = asyResolution,
#md #     curves=[geodesicCurve], points = [ [x,y], Z], tVectors = [Vx],
#md #     colors=Dict(
#md #         :curves => [black],
#md #         :points => [TolVibrantOrange,TolVibrantCyan],
#md #         :tvectors => [TolVibrantCyan]
#md #     ),
#md #     dotSizes = [3.5,2.], lineWidth = 0.75, cameraPosition = (1.,1.,.5)
#md # )
#md # render_asymptote("jacobiGeodesicdifferential_geodesic_startpoint.asy"; render = 2)
#md # ```
#
#md # ![A Jacobi field for $D_xg(t,x,y)[\eta]$](../assets/images/tutorials/jacobiGeodesicdifferential_geodesic_startpoint.png)
#
# If we further move the end point, too, we can derive that Differential in direction
Xq = [0.2, 0.0, -0.5]
W2 = differential_geodesic_endpoint.(Ref(M), Ref(p), Ref(q), T, Ref(Xq))
V2 = [Tuple([a, b]) for (a, b) in zip(r, W2)]
# and we can combine both keeping the base point
V3 = [Tuple([a, b]) for (a, b) in zip(r, W2 + W)]
asymptote_export_S2_signals( #src
    exportFolder * "/jacobiGeodesicResult.asy"; #src
    curves = [geodesicCurve], #src
    points = [[p, q], r], #src
    tVectors = [V, V2, V3], #src
    colors = Dict( #src
        :curves => [black], #src
        :points => [TolVibrantOrange, TolVibrantCyan], #src
        :tvectors => [TolVibrantCyan, TolVibrantCyan, TolVibrantTeal], #src
    ), #src
    dotSizes = [3.5, 2.0], #src
    lineWidth = 0.75, #src
    cameraPosition = (1.0, 1.0, 0.0), #src
) #src
render_asymptote(exportFolder * "/jacobiGeodesicResult.asy"; render = 2) #src
#md # ```julia
#md # asymptote_export_S2_signals("jacobiGeodesicResult.asy";
#md #    render = asyResolution,
#md #    curves=[geodesicCurve], points = [ [x,y], Z], tVectors = [Vx,Vy,Vb],
#md #    colors=Dict(
#md #        :curves => [black],
#md #        :points => [TolVibrantOrange,TolVibrantCyan],
#md #        :tvectors => [TolVibrantCyan,TolVibrantCyan,TolVibrantTeal]
#md #   ),
#md #   dotSizes = [3.5,2.], lineWidth = 0.75, cameraPosition = (1.,1.,0.)
#md # )
#md # render_asymptote("jacobiGeodesicResult.asy"; render = 2)
#md # ```
#md #
#md # ![A Jacobi field for the effect of two differentials (blue) in sum (teal)](../assets/images/tutorials/jacobiGeodesicResult.png)
#
# ## Literature
#
# ```@raw html
# <ul><li id="doCarmo1992">[<a>doCarmo1992</a>] do Carmo, M. P.:
#    <emph>Riemannian Geometry</emph>, Mathematics: Theory & Applications,
#    Birkhäuser Basel, 1992, ISBN: 0-8176-3490-8</li>
# <li id="BergmannGousenbourger2018">[<a>BergmannGousenbourger2018</a>]
#   Bergmann, R.; Gousenbourger, P.-Y.: <emph>A variational model for data
#   fitting on manifolds by minimizing the acceleration of a Bézier curve</emph>,
#   Frontiers in Applied Mathematics and Statistics, 2018.
#   doi: <a href="https://dx.doi.org/10.3389/fams.2018.00059">10.3389/fams.2018.00059</a></li>
# </ul>
# ```
