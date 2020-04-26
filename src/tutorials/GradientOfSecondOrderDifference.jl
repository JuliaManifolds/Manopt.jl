# # [Illustration of the Gradient of a Second Order Difference](@id secondOrderDifferenceGrad)
#
# This example explains how to compute the gradient of the second order
# difference mid point model using [`adjoint_Jacobi_field`](@ref)s.
#
# This example also illustrates the `PowerManifold` manifold as well
# as [`ArmijoLinesearch`](@ref).

# We first initialize the manifold
exportFolder = joinpath(@__DIR__,"..","..","docs","src","assets","images","tutorials") #src
using Manopt, Manifolds
# and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)
using Colors
black = RGBA{Float64}(colorant"#000000")
TolVibrantBlue = RGBA{Float64}(colorant"#0077BB") # points
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733") # results
TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE") # vectors
TolVibrantTeal = RGBA{Float64}(colorant"#009988") # geo
nothing #hide
# Assume we have two points $x,y$ on the equator of the
# [Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) $\mathcal M = \mathbb S^2$
# and a point $y$ near the north pole
M = Sphere(2)
p = [1., 0., 0.]
q = [0., 1., 0.]
c = mid_point(M,p,q)
#src y is the north pole just bend a little bit towards
r = shortest_geodesic(M, [0., 0., 1.], c, 0.1)
[c,r]
# Now the second order absolute difference can be stated as (see [[Bačák, Bergmann, Steidl, Weinmann, 2016](#BacakBergmannSteidlWeinmann2016)])
#
# $d_2(x,y,z) := \min_{c ∈ \mathcal C_{x,z}} d_{\mathcal M}(c,y),\qquad x,y,z∈\mathcal M,$
#
# where $\mathcal C_{x,z}$ is the set of all mid points $g(\frac{1}{2};x,z)$, where $g$
# is a (not necessarily minimizing) geodesic connecting $x$ and $z$.
#
# For illustration we further define the point opposite of
c2 = -c
# and draw the geodesic connecting $y$ and the nearest mid point $c$, namely
T = [0:0.1:1.0...]
geoPts_yc = shortest_geodesic(M,r,c,T)
nothing #hide
# looks as follows using the [`asymptote_export_S2_signals`](@ref) export
asymptote_export_S2_signals(exportFolder*"/SecondOrderData.asy"; #src
    curves = [ geoPts_yc ], #src
    points = [ [p,r,q], [c,c2] ], #src
    colors=Dict(:curves => [TolVibrantTeal], :points => [black, TolVibrantBlue]), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
render_asymptote(exportFolder*"/SecondOrderData.asy"; render=2) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("secondOrderData.asy";
#md #     render = asyResolution,
#md #     curves = [ geoPts_yc ],
#md #     points = [ [x,y,z], [c,c2] ],
#md #     colors=Dict(:curves => [TolVibrantTeal], :points => [black, TolVibrantBlue]),
#md #     dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5)
#md # )
#md # render_asymptote("SecondOrderData.asy"; render=2)
#md # ```
#
#md # ![Three points $p,r,q$ and the midpoint $c=c(p,q)$ (blue)](../assets/images/tutorials/SecondOrderData.png)
#
# Since we moved $r$ 10% along the geodesic from the north pole to $c$, the distance
# to $c$ is $\frac{9\pi}{20}\approx 1.4137$, and this is also what
costTV2(M, (p,r,q) )
# returns, see [`costTV2`](@ref) for reference. But also its gradient can be
# easily computed since it is just a distance with respect to $y$ and a
# concatenation of a geodesic, where the start or end point is the argument,
# respectively, with a distance.
# Hence the [adjoint differentials](@ref adjointDifferentialFunctions)
# [`adjoint_differential_geodesic_startpoint`](@ref) and [`adjoint_differential_geodesic_endpoint`](@ref) can be employed,
# see [`∇TV2`](@ref) for details.
# we obtain
(Xp, Xr, Xq) = ∇TV2(M, (p,r,q) )
#
# When we aim to minimize this, we look at the negative gradient, i.e.
# we can draw this as
asymptote_export_S2_signals(exportFolder*"/SecondOrderGradient.asy"; #src
    points = [ [p,r,q], [c,c2] ], #src
    tVectors = [Tuple.([ [p, -Xp], [r, -Xr], [q, Xq]])], #src
    colors=Dict(:tvectors => [TolVibrantCyan], :points => [black, TolVibrantBlue]), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
render_asymptote(exportFolder*"/SecondOrderGradient.asy"; render=2) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("SecondOrderGradient.asy";
#md #    points = [ [x,y,z], [c,c2] ],
#md #    tVectors = [Tuple.([ [p, -Xp], [r, -Xr], [q, Xq]])], #src
#md #    colors=Dict(:tvectors => [TolVibrantCyan], :points => [black, TolVibrantBlue]),
#md #    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5)
#md # )
#md # render_asymptote("SecondOrderGradient.asy"; render=2)
#md # ```
#md #
#md # ![Three points $x,y,z$ and the negative gradient of the second order absolute difference](../assets/images/tutorials/SecondOrderGradient.png)
#
# If we now perform a gradient step, we obtain the three points
pn, rn, qn = exp.(Ref(M), [p,r,q], [-Xp,-Xr,-Xq])
# as well we the new mid point
cn = mid_point(M,pn,qn)
geoPts_yncn = shortest_geodesic(M,rn,cn,T)
nothing #hide
# and obtain the new situation
asymptote_export_S2_signals(exportFolder*"/SecondOrderMin1.asy"; #src
    points = [ [p,r,q], [c,c2,cn], [pn,rn,qn] ], #src
    curves = [ geoPts_yncn ], #src
    tVectors = [Tuple.([ [p, -Xp], [r, Xr], [q, Xq] ])], #src
    colors=Dict(:tvectors => [TolVibrantCyan], #src
                :points => [black, TolVibrantBlue, TolVibrantOrange], #src
                :curves => [TolVibrantTeal] #src
                ), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
render_asymptote(exportFolder*"/SecondOrderMin1.asy"; render=2) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("SecondOrderMin1.asy";
#md #     points = [ [x,y,z], [c,c2,cn], [xn,yn,zn] ],
#md #     curves = [ geoPts_yncn ] ,
#md #     tVectors = [Tuple.([ [p, -Xp], [r, Xr], [q, Xq] ])],
#md #     colors=Dict(:tvectors => [TolVibrantCyan],
#md #         :points => [black, TolVibrantBlue, TolVibrantOrange],
#md #         :curves => [TolVibrantTeal]
#md #     ),
#md #     dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5)
#md # )
#md # render_asymptote("SecondOrderMin1.asy"; render=2)
#md # ```
#md
#md # ![A gradient Step](../assets/images/tutorials/SecondOrderMin1.png)
#
# One can see, that this step slightly “overshoots”, i.e. $r$ is now even below $c$.
# and the cost function is still at
costTV2(M, (pn, rn, qn) )
#
# But we can also search for the best step size using [`ArmijoLinesearch`](@ref)
# on the `PowerManifold` manifold $\mathcal N = \mathcal M^3 = (\mathbb S^2)^3$
x = [p,r,q]
N = PowerManifold(M, NestedPowerRepresentation(),3)
s = ArmijoLinesearch(1.0,ExponentialRetraction(),0.999,0.96)(N, x,
    x -> costTV2(M, Tuple(x)),
     [ ∇TV2(M, (p,r,q))... ]  # transform from tuple to PowTVector
)
# and for the new points
pm, rm, qm = exp.(Ref(M), [p,r,q], s*[-Xp,-Xr,-Xq])
cm = mid_point(M,pm,qm)
geoPts_xmzm = shortest_geodesic(M,pm,qm,T)
nothing #hide
# we obtain again with
asymptote_export_S2_signals(exportFolder*"/SecondOrderMin2.asy"; #src
    points = [ [p,r,q], [c,c2,cm], [pm,rm,qm] ], #src
    curves = [ geoPts_xmzm ], #src
    tVectors = [Tuple.([ [p, -Xp], [r, Xr], [q, Xq] ])], #src
    colors=Dict(:tvectors => [TolVibrantCyan], #src
                :points => [black, TolVibrantBlue, TolVibrantOrange], #src
                :curves => [TolVibrantTeal] #src
                ), #src
    dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5) #src
) #src
render_asymptote(exportFolder*"/SecondOrderMin2.asy"; render=2) #src
#md #
#md # ```julia
#md # asymptote_export_S2_signals("SecondOrderMin2.asy";
#md #     points = [ [x,y,z], [c,c2,cm], [xm,ym,zm] ],
#md #     curves = [ geoPts_xmzm ] ,
#md #     tVectors = [Tuple.( [-ξx, -ξy, -ξz], [x, y, z] )],
#md #     colors=Dict(:tvectors => [TolVibrantCyan],
#md #                 :points => [black, TolVibrantBlue, TolVibrantOrange],
#md #                 :curves => [TolVibrantTeal]
#md #                 ),
#md #     dotSize = 3.5, lineWidth = 0.75, cameraPosition = (1.2,1.,.5)
#md # )
#md # render_asymptote("SecondOrderMin2.asy"; render=2) #src
#md # ```
#md #
#md # ![A gradient Step](../assets/images/tutorials/SecondOrderMin2.png)
#
# Here, the cost function yields
costTV2( M, (pm, rm, qm) )
# which is nearly zero, as one can also see, since the new center $c$ and $r$
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
