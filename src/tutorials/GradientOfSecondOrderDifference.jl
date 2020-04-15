# # [Illustration of the Gradient of a Second Order Difference](@id secondOrderDifferenceGrad)
#
# This example explains how to compute the gradient of the second order
# difference mid point model using [`adjoint_Jacobi_field`](@ref)s.
#
# This example also illustrates the `PowerManifold` manifold as well
# as [`ArmijoLinesearch`](@ref).

# We first initialize the manifold
exportFolder = joinpath(@__DIR__,"..","..","docs","src","assets","images","tutorials") #src
using Manopt
# and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)
using Colors
black = RGBA{Float64}(colorant"#000000")
TolVibrantBlue = RGBA{Float64}(colorant"#0077BB") # points
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733") # results
TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE") # vectors
TolVibrantTeal = RGBA{Float64}(colorant"#009988") # geo
asyResolution = 2
nothing #hide
# Assume we have two points $x,y$ on the equator of the
# [`Sphere`](@ref)`(2)` $\mathcal M = \mathbb S^2$
# and a point $y$ near the north pole
M = Sphere(2)
x = [1., 0., 0.]
z = [0., 1., 0.]
c = mid_point(M,x,z)
#src y is the north pole just bend a little bit towards
y = shortest_geodesic(M, [0., 0., 1.]), c, 0.1
[c,y]
# Now the second order absolute difference can be stated as (see [[Bačák, Bergmann, Steidl, Weinmann, 2016](#BacakBergmannSteidlWeinmann2016)])
#
# $d_2(x,y,z) := \min_{c ∈ \mathcal C_{x,z}} d_{\mathcal M}(c,y),\qquad x,y,z∈\mathcal M,$
#
# where $\mathcal C_{x,z}$ is the set of all mid points $g(\frac{1}{2};x,z)$, where $g$
# is a (not necessarily minimizing) geodesic connecting $x$ and $z$.
#
# For illustration we further define the point opposite of
c2 = opposite(M,c)
# and draw the geodesic connecting $y$ and the nearest mid point $c$, namely
T = [0:0.1:1.0...]
geoPts_yc = shortest_geodesic(M,y,c,T)
nothing #hide
# looks as follows using the [`asymptote_export_S2_signals`](@ref) export
asymptote_export_S2_signals(exportFolder*"/SecondOrderData.asy"; #src
    render = asyResolution, #src
    curves = [ geoPts_yc ], #src
    points = [ [x,y,z], [c,c2] ], #src
    colors=Dict(:curves => [TolVibrantTeal], :points => [black, TolVibrantBlue]), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("secondOrderData.asy";
#md #     render = asyResolution,
#md #     curves = [ geoPts_yc ],
#md #     points = [ [x,y,z], [c,c2] ],
#md #     colors=Dict(:curves => [TolVibrantTeal], :points => [black, TolVibrantBlue]),
#md #     dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5)
#md # )
#md # ```
#
#md # ![Three points $x,y,z$ and the midpoint $c=c(x,z)$ (blue)](../assets/images/tutorials/SecondOrderData.png)
#
# Since we moved $y$ 10% along the geodesic from the north pole to $c$, the distance
# to $c$ is $\frac{9\pi}{20}\approx 1.4137$, and this is also what
costTV2(M, (x,y,z) )
# returns, see [`costTV2`](@ref) for reference. But also its gradient can be
# easily computed since it is just a distance with respect to $y$ and a
# concatenation of a geodesic, where the start or end point is the argument,
# respectively, with a distance.
# Hence the [adjoint differentials](@ref adjointDifferentialFunctions)
# [`AdjDpGeo`](@ref) and [`AdjDqGeo`](@ref) can be employed,
# see [`∇TV2`](@ref) for details.
# we obtain
(ξx, ξy, ξz) = ∇TV2(M, (x,y,z) )
#
# When we aim to minimize this, we look at the negative gradient, i.e.
# we can draw this as
asymptote_export_S2_signals(exportFolder*"/SecondOrderGradient.asy"; #src
    render = asyResolution, #src
    points = [ [x,y,z], [c,c2] ], #src
    tVectors = [Tuple.( [-ξx, -ξy, -ξz], [x, y, z] )], #src
    colors=Dict(:tvectors => [TolVibrantCyan], :points => [black, TolVibrantBlue]), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("SecondOrderGradient.asy";
#md #    render = asyResolution,
#md #    points = [ [x,y,z], [c,c2] ],
#md #    tVectors = [Tuple.( [-ξx, -ξy, -ξz], [x, y, z] )],
#md #    colors=Dict(:tvectors => [TolVibrantCyan], :points => [black, TolVibrantBlue]),
#md #    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5)
#md # )
#md # ```
#md #
#md # ![Three points $x,y,z$ and the negative gradient of the second order absolute difference](../assets/images/tutorials/SecondOrderGradient.png)
#
# If we now perform a gradient step, we obtain the three points
xn, yn, zn = exp.(Ref(M), [x,y,z], [-ξx,-ξy,-ξz])
# as well we the new mid point
cn = mid_point(M,xn,zn)
geoPts_yncn = shortest_geodesic(M,yn,cn,T)
nothing #hide
# and obtain the new situation
asymptote_export_S2_signals(exportFolder*"/SecondOrderMin1.asy"; #src
    render = asyResolution, #src
    points = [ [x,y,z], [c,c2,cn], [xn,yn,zn] ], #src
    curves = [ geoPts_yncn ], #src
    tVectors = [Tuple.( [-ξx, -ξy, -ξz], [x, y, z] )], #src
    colors=Dict(:tvectors => [TolVibrantCyan], #src
                :points => [black, TolVibrantBlue, TolVibrantOrange], #src
                :curves => [TolVibrantTeal] #src
                ), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("SecondOrderMin1.asy";
#md #     render = asyResolution,
#md #     points = [ [x,y,z], [c,c2,cn], [xn,yn,zn] ],
#md #     curves = [ geoPts_yncn ] ,
#md #     tVectors = [Tuple.( [-ξx, -ξy, -ξz], [x, y, z] )],
#md #     colors=Dict(:tvectors => [TolVibrantCyan],
#md #         :points => [black, TolVibrantBlue, TolVibrantOrange],
#md #         :curves => [TolVibrantTeal]
#md #     ),
#md #     dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5)
#md # )
#md # ```
#md
#md # ![A gradient Step](../assets/images/tutorials/SecondOrderMin1.png)
#
# One can see, that this step slightly “overshoots”, i.e. $y$ is now even below $c$.
# and the cost function is still at
costTV2(M, (xn, yn, zn) )
#
# But we can also search for the best step size using [`ArmijoLinesearch`](@ref)
# on the `PowerManifold` manifold $\mathcal N = \mathcal M^3 = (\mathbb S^2)^3$
p = [x,y,z]
N = PowerManifold(M,3)
s = ArmijoLinesearch(1.0,exp,0.999,0.96)(N, p,
    x -> costTV2(M, Tuple(x)),
     [ ∇TV2(M, (x,y,z))... ]  # transform from tuple to PowTVector
)
# and for the new points
xm, ym, zm = exp.(Ref(M), [x,y,z], s*[-ξx,-ξy,-ξz])
cm = mid_point(M,xm,zm)
geoPts_xmzm = shortest_geodesic(M,xm,zm,T)
nothing #hide
# we obtain again with
asymptote_export_S2_signals(exportFolder*"/SecondOrderMin2.asy"; #src
    render = asyResolution, #src
    points = [ [x,y,z], [c,c2,cm], [xm,ym,zm] ], #src
    curves = [ geoPts_xmzm ], #src
    tVectors = [Tuple.( [-ξx, -ξy, -ξz], [x, y, z] )], #src
    colors=Dict(:tvectors => [TolVibrantCyan], #src
                :points => [black, TolVibrantBlue, TolVibrantOrange], #src
                :curves => [TolVibrantTeal] #src
                ), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("SecondOrderMin2.asy";
#md #     render = asyResolution,
#md #     points = [ [x,y,z], [c,c2,cm], [xm,ym,zm] ],
#md #     curves = [ geoPts_xmzm ] ,
#md #     tVectors = [Tuple.( [-ξx, -ξy, -ξz], [x, y, z] )],
#md #     colors=Dict(:tvectors => [TolVibrantCyan],
#md #                 :points => [black, TolVibrantBlue, TolVibrantOrange],
#md #                 :curves => [TolVibrantTeal]
#md #                 ),
#md #     dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5)
#md # )
#md # ```
#md #
#md # ![A gradient Step](../assets/images/tutorials/SecondOrderMin2.png)
#
# Here, the cost function yields
costTV2( M, (xm, ym, zm) )
# which is nearly zero, as one can also see, since the new center $c$ and $y$
# are quite close.
#
# ## Literature
#
# ```@raw html
# <ul>
# <li id="BačákBergmannSteidlWeinmann2016">[<a>Bačák, Bergmann, Steidl, Weinmann, 2016</a>]
#   Bačák, M; Bergmann, R.; Steidl, G; Weinmann, A.: <emph>A second order nonsmooth
#   variational model for restoring manifold-valued images.</emph>,
#   SIAM Journal on Scientific Computations, Volume 38, Number 1, pp. A567–597,
#   doi: <a href="https://doi.org/10.1137/15M101988X">10.1137/15M101988X</a></li>
# </ul>
# ```
