# # Illustration of Jacobi Fields
#
# This tutorial illustrates the usage of Jacobi Fields within
# `Manopt.jl`.
# For this tutorial you should be familiar with the basic terminology on a
# manifold like the exponential and logarithmic map as well as
# [`geodesic`](@ref)s.
#
# We first initialize the manifold
exportFolder = joinpath(@__DIR__,"..","..","docs","src","assets","images","tutorials") #src
using Manopt
# and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)
using Colors
black = RGBA{Float64}(colorant"#000000")
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE")
TolVibrantTeal = RGBA{Float64}(colorant"#009988")
nothing #hide
# Assume we have two [`SnPoint`](@ref)s on the equator of the [`Sphere`](@ref)`(2)` $\mathcal M = \mathbb S^2$
M = Sphere(2)
x,y = [ SnPoint([1.,0.,0.]), SnPoint([0.,1.,0.])]
# their connecting [`geodesic`](@ref) (sampled at `100` points)
geodesicCurve = geodesic(M,x,y,100);
asyResolution = 2
nothing #hide
# looks as follows using [`renderAsymptote`](@ref) with the [`asyExportS2Signals`](@ref) export
renderAsymptote(exportFolder*"/jacobiGeodesic.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    curves=[geodesicCurve], points = [ [x,y] ], #src
    colors=Dict(:curves => [black], :points => [TolVibrantOrange]), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.,1.,.5) #src
)#src
#md #
#md # ```julia
#md # renderAsymptote("jacobiGeodesic.asy",asyExportS2Signals;
#md #     render = asyResolution,
#md #     curves=[geodesicCurve], points = [ [x,y] ],
#md #     colors=Dict(:curves => [black], :points => [TolVibrantOrange]),
#md #     dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.,1.,.5)
#md # )
#md # ```
#md # 
#md # ![A geodesic connecting two points on the equator](../assets/images/tutorials/jacobiGeodesic.png)
#
# where $x$ is on the left. Then this tutorial solves the following task:
# 
# Given a direction $\xi_x\in T_x\mathcal M$, for example the [`SnTVector`](@ref)
ξx = SnTVector([0.,0.4,0.5])
# we move the start point $x$ into, how does any point on the geodesic move?
#
# Or mathematically: Compute $D_x g(t; x,y)$ for some fixed $t\in[0,1]$
# and a given direction $\xi_x$.
# Of course two cases are quite easy: For $t=0$ we are in $x$ and how $x$ “moves”
# is already known, so $D_x g(0;x,y) = \xi$. On the other side, for $t=1$,
# $g(1; x,y) = y$ which is fixed, so $D_x g(1; x,y)$ is the zero tangent vector
# (in $T_y\mathcal M$).
#
# For all other cases we employ a [`jacobiField`](@ref), which is a (tangent)
# vector field along the [`geodesic`](@ref) given as follows: The _geodesic variation_
# $\Gamma_{g,\xi}(s,t)$ is defined for some $\varepsilon > 0$ as
#
# $\Gamma_{g,\xi}(s,t):=\exp{\gamma_{x,\xi}(s)}[t\log_{g(s;x,\xi)}y],\qquad s\in(-\varepsilon,\varepsilon),\ t\in[0,1].$
#
# Intuitively we make a small step $s$ into direction $\xi$ using the geodesic
# $g(\cdot; x,\xi)$ and from $z=g(s; x,\xi)$ we follow (in $t$) the geodesic
# $g(\cdot; z,y)$. The corresponding Jacobi field~\(J_{g,\xi}\)
# along~\(g(\cdot; x,y\) is given
#
# $J_{g,\xi}(t):=\frac{D}{\partial s}\Gamma_{g,\xi}(s,t)\Bigl\rvert_{s=0}$
#
# which is an ODE and we know the boundary conditions $J_{g,\xi}(0)=\xi$ and
# $J_{g,\xi}(t) = 0$. In symmetric spaces we can compute the solution, since the
# system of ODEs decouples, see for example [do Carmo](#doCarmo1992),
# Chapter 4.2. Within `Manopt.jl` this is implemented as
# [`jacobiField`](@ref)`(M,x,y,t,ξ[,β])`, where the optional parameter (function)
# `β` specifies, which Jacobi field we want to evaluate and the one used here is
# the default.
#
# We can hence evaluate that on the points on the geodesic at
T = [0:0.1:1.0...]
nothing #hide
# namely
Z = geodesic(M,x,y,T)
nothing #hide
# the geodesic moves as
ηx = jacobiField.(Ref(M), Ref(x), Ref(y), T, Ref(ξx) )
# which can also be called using [`DxGeo`](@ref).
# We can add to the image above by creating extended tangent vectors
# [`TVectorE`](@ref) the include their base points
Vx = TVectorE.(ηx,Z)
# and add that as one further set to the Asymptote export.
renderAsymptote(exportFolder*"/jacobiGeodesicDxGeo.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    curves=[geodesicCurve], points = [ [x,y], Z], tVectors = [Vx], #src
    colors=Dict( #src
        :curves => [black], #src
        :points => [TolVibrantOrange,TolVibrantCyan], #src
        :tvectors => [TolVibrantCyan] #src
    ), #src
    dotSizes = [3.5,2.], lineWidth = 0.75, cameraPosition = (1.,1.,.5) #src
) #src
#md #
#md # ```julia
#md # renderAsymptote("jacobiGeodesicDxGeo.asy",asyExportS2Signals;
#md #     render = asyResolution,
#md #     curves=[geodesicCurve], points = [ [x,y], Z], tVectors = [Vx],
#md #     colors=Dict(
#md #         :curves => [black],
#md #         :points => [TolVibrantOrange,TolVibrantCyan],
#md #         :tvectors => [TolVibrantCyan]
#md #     ),
#md #     dotSizes = [3.5,2.], lineWidth = 0.75, cameraPosition = (1.,1.,.5)
#md # )
#md # ```
#
#md # ![A Jacobi field for $D_xg(t,x,y)[\eta]$](../assets/images/tutorials/jacobiGeodesicDxGeo.png)
#
# If we further move the end point, too, we can derive that Differential in direction
ξy = SnTVector([0.2,0.,-0.5])
ηy = DyGeo.(Ref(M),Ref(x),Ref(y),T,Ref(ξy))
Vy = TVectorE.(ηy,Z)
# and we can look at the total effect, where the [`TVectorE`](@ref)s even verify
# that only tangent vectors are added that have a common base point
Vb = Vx .+ Vy
renderAsymptote(exportFolder*"/jacobiGeodesicResult.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    curves=[geodesicCurve], points = [ [x,y], Z], tVectors = [Vx,Vy,Vb], #src
    colors=Dict( #src
        :curves => [black], #src
        :points => [TolVibrantOrange,TolVibrantCyan], #src
        :tvectors => [TolVibrantCyan,TolVibrantCyan,TolVibrantTeal] #src
    ), #src
    dotSizes = [3.5,2.], lineWidth = 0.75, cameraPosition = (1.,1.,0.) #src
) #src
#md # ```julia
#md # renderAsymptote("jacobiGeodesicResult.asy",asyExportS2Signals;
#md #    render = asyResolution,
#md #    curves=[geodesicCurve], points = [ [x,y], Z], tVectors = [Vx,Vy,Vb],
#md #    colors=Dict(
#md #        :curves => [black],
#md #        :points => [TolVibrantOrange,TolVibrantCyan],
#md #        :tvectors => [TolVibrantCyan,TolVibrantCyan,TolVibrantTeal]
#md #   ),
#md #   dotSizes = [3.5,2.], lineWidth = 0.75, cameraPosition = (1.,1.,0.)
#md # )
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