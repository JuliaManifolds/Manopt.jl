using Manopt, Colors
M = Sphere(2)
jBlue = RGBA{Float64}(colorant"#4063D8")
jGreen = RGBA{Float64}(colorant"#389826")
jPurple = RGBA{Float64}(colorant"#9558B2")
jRed = RGBA{Float64}(colorant"#CB3C33")
black = RGBA{Float64}(colorant"#000000")
white = RGBA{Float64}(colorant"#FFFFFF")
jPurpleAlpha = RGBA{Float64}(red(jPurple), green(jPurple), blue(jPurple),0.7)
scaleColor(c::RGBA{Float64},t::Float64) = RGBA{Float64}(t*red(c),t*green(c),t*blue(c),alpha(c))
 

x0 = SnPoint([1., 0., 0.])
ξ1 = SnTVector([0., 1., 0.])
ξ2 = SnTVector([0., 0., 1.])
ell = sqrt(2.)
η = SnTVector([0.,0.65,-0.55])
xHat = exp(M,x0,η)
# local function around xhat (roughly up to x0) where we know contour lines
f(x) = (dot(M,x0,log(M,x0,x),ξ1) - getValue(η)[2])^2 + (1/ell*(dot(M,x0,log(M,x0,x),ξ2) - getValue(η)[3]))^2
contour(pts=91,r=0.3) = [ SnTVector([0., r*cos(α) + getValue(η)[2], r*ell*sin(α)  + getValue(η)[3]]) for α in range(0,2*π,length=pts) ]
c1 = exp.(Ref(M),Ref(x0), contour(91,0.1)); p1 = f.(c1)
c2 = exp.(Ref(M),Ref(x0), contour(91,0.2)); p2 = f.(c2)
c3 = exp.(Ref(M),Ref(x0), contour(91,0.3)); p3 = f.(c3)
c4 = exp.(Ref(M),Ref(x0), contour(91,0.4)); p4 = f.(c4)
c5 = exp.(Ref(M),Ref(x0), contour(91,0.5)); p5 = f.(c5)
c6 = exp.(Ref(M),Ref(x0), contour(91,0.6)); p6 = f.(c6)
c7 = exp.(Ref(M),Ref(x0), contour(91,0.7)); p7 = f.(c7)

stepVec = TVectorE(3/4 * η,x0)
stepCurve = geodesic(M,x0,xHat,[range(0,3/4,length=50)...])
#
# two steps back - we don't know the function globally so this is merely random
ηP1 = SnTVector([0.,-0.2,-0.45])
xP1 = exp(M,x0,ηP1)
ηP2 = parallelTransport(M,x0,xP1,SnTVector([0.,-0.7,0.55]))
xP2 = exp(M,xP1,ηP2)
stepCurveP1 = geodesic(M,xP2,xP1,[range(0,1,length=50)...])
stepCurveP2 = geodesic(M,xP1,x0,[range(0,1,length=50)...])

renderAsymptote("Logo.png", asyExportS2Signals;
    curves=[c1,c2,c3,c4,c5,c6,c7,stepCurve,stepCurveP1,stepCurveP2],
    points = [[xP1,xP2,x0,stepCurve[end]],[xHat]],
    tVectors = [ [stepVec] ],
    colors=Dict(
        :curves => [scaleColor(white,.1), scaleColor(white,.2),
                    scaleColor(white,.3), scaleColor(white,.4),
                    scaleColor(white, .5), scaleColor(white,.6),
                    scaleColor(white,.7), jRed, jRed, jRed],
        :points => [jGreen,jPurple],
        :tvectors => [jBlue]
    ),
    dotSize = 10.,
    lineWidths = [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 4., 4., 4., 4.],
    arrowHeadSize = 22.,
    cameraPosition = (1., 0.2, -.75),
    sphereColor = RGBA{Float64}(0.85, 0.85, 0.85, 1.),
    sphereLineColor = RGBA{Float64}(0.75, 0.75, 0.75, 1.),
    # sphereColor = jPurpleAlpha, sphereLineColor = jPurpleAlpha,
    sphereLineWidth=1.5,
    format="png", render=6
)