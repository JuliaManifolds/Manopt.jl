# # [Illustration of the Gradient of a Second Order Difference](@id secondOrderDifferenceGrad)
#
# This example explains how to compute the gradient of the second order
# difference mid point model using [`adjointJacobiField`](@ref)s.
#
# This example also illustrates the [`Power`](@ref) manifold as well
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
# Assume we have two [`SnPoint`](@ref)s $x,y$ on the equator of the
# [`Sphere`](@ref)`(2)` $\mathcal M = \mathbb S^2$
# and a point $y$ near the north pole
M = Sphere(2)
x = SnPoint([1., 0., 0.])
z = SnPoint([0., 1., 0.])
c = midPoint(M,x,z)
#src y is the north pole just bend a little bit towards 
y = geodesic(M, SnPoint([0., 0., 1.]), c, 0.1)
[c,y]
# Now the second order absolute difference can be stated as (see [[Bačák, Bergmann, Steidl, Weinmann, 2016](#BacakBergmannSteidlWeinmann2016)])
#
# $d_2(x,y,z) := \min_{c\in\mathcal C_{x,z}} d_{\mathcal M}(c,y),\qquad x,y,z\in\mathcal M,$
#
# where $\mathcal C_{x,z}$ is the set of all mid points $g(\frac{1}{2};x,z)$, where $g$
# is a (not necessarily minimizing) geodesic connecting $x$ and $z$.
# 
# For illustration we further define the point opposite of 
c2 = opposite(M,c)
# and draw the geodesic connecting $y$ and the nearest mid point $c$, namely
T = [0:0.1:1.0...]
geoPts_yc = geodesic(M,y,c,T)
nothing #hide
# looks as follows using [`renderAsymptote`](@ref) with the [`asyExportS2Signals`](@ref) export
renderAsymptote(exportFolder*"/SecondOrderData.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    curves = [ geoPts_yc ], #src
    points = [ [x,y,z], [c,c2] ], #src
    colors=Dict(:curves => [TolVibrantTeal], :points => [black, TolVibrantBlue]), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
#md # 
#md # ```julia
#md # renderAsymptote("secondOrderData.asy",asyExportS2Signals;
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
# [`AdjDxGeo`](@ref) and [`AdjDyGeo`](@ref) can be employed,
# see [`gradTV2`](@ref) for details.
# we obtain
(ξx, ξy, ξz) = gradTV2(M, (x,y,z) )
#
# When we aim to minimize this, we look at the negative gradient, i.e.
# we can draw this as
renderAsymptote(exportFolder*"/SecondOrderGradient.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    points = [ [x,y,z], [c,c2] ], #src
    tVectors = [TVectorE.( [-ξx, -ξy, -ξz], [x, y, z] )], #src
    colors=Dict(:tvectors => [TolVibrantCyan], :points => [black, TolVibrantBlue]), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
#md #
#md # ```julia
#md # renderAsymptote("SecondOrderGradient.asy",asyExportS2Signals;
#md #    render = asyResolution,
#md #    points = [ [x,y,z], [c,c2] ],
#md #    tVectors = [TVectorE.( [-ξx, -ξy, -ξz], [x, y, z] )],
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
cn = midPoint(M,xn,zn)
geoPts_yncn = geodesic(M,yn,cn,T)
nothing #hide
# and obtain the new situation
renderAsymptote(exportFolder*"/SecondOrderMin1.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    points = [ [x,y,z], [c,c2,cn], [xn,yn,zn] ], #src
    curves = [ geoPts_yncn ], #src
    tVectors = [TVectorE.( [-ξx, -ξy, -ξz], [x, y, z] )], #src
    colors=Dict(:tvectors => [TolVibrantCyan], #src
                :points => [black, TolVibrantBlue, TolVibrantOrange], #src
                :curves => [TolVibrantTeal] #src
                ), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
#md #
#md # ```julia
#md # renderAsymptote("SecondOrderMin1.asy",asyExportS2Signals;
#md #     render = asyResolution,
#md #     points = [ [x,y,z], [c,c2,cn], [xn,yn,zn] ],
#md #     curves = [ geoPts_yncn ] ,
#md #     tVectors = [TVectorE.( [-ξx, -ξy, -ξz], [x, y, z] )],
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
# on the [`Power`](@ref) manifold $\mathcal N = \mathcal M^3 = (\mathbb S^2)^3$
p = PowPoint([x,y,z])
N = Power(M,3)
s = ArmijoLinesearch(1.0,exp,0.999,0.96)(N, p,
    x -> costTV2(M, Tuple(getValue(x))),
    PowTVector( [ gradTV2(M, (x,y,z))... ] ) # transform from tuple to PowTVector
)
# and for the new points
xm, ym, zm = exp.(Ref(M), [x,y,z], s*[-ξx,-ξy,-ξz])
cm = midPoint(M,xm,zm)
geoPts_xmzm = geodesic(M,xm,zm,T)
nothing #hide
# we obtain again with
renderAsymptote(exportFolder*"/SecondOrderMin2.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    points = [ [x,y,z], [c,c2,cm], [xm,ym,zm] ], #src
    curves = [ geoPts_xmzm ], #src
    tVectors = [TVectorE.( [-ξx, -ξy, -ξz], [x, y, z] )], #src
    colors=Dict(:tvectors => [TolVibrantCyan], #src
                :points => [black, TolVibrantBlue, TolVibrantOrange], #src
                :curves => [TolVibrantTeal] #src
                ), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
#md #
#md # ```julia
#md # renderAsymptote("SecondOrderMin2.asy",asyExportS2Signals;
#md #     render = asyResolution,
#md #     points = [ [x,y,z], [c,c2,cm], [xm,ym,zm] ],
#md #     curves = [ geoPts_xmzm ] ,
#md #     tVectors = [TVectorE.( [-ξx, -ξy, -ξz], [x, y, z] )],
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
