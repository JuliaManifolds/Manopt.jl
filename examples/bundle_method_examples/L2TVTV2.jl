using Random, QuadraticModels, RipQP, Manifolds, Manopt

Random.seed!(1)
include("level_set_diameter.jl")
α = 0.035
β = 0.02
M = SymmetricPositiveDefinite(3)
img = artificial_SPD_image(10)
N = PowerManifold(M, NestedPowerRepresentation(), size(img)[1], size(img)[2])
p0 = rand(N)
img2 = map(p -> exp(M, p, rand(M; vector_at=p, tangent_distr=:Rician, σ=0.03)), img) #add (exp) noise to image
f(N, q) = costL2TVTV2(N, img, α, β, q)
gradf(N, q) =  grad_distance(N, img, q) + α * grad_TV(N, q) + β * grad_TV2(N, q)

# diam = level_set_diameter(N, f, gradf, p0; debug_var=true)
println("Bundle method:")
b1 = bundle_method(N, f, gradf, img2; 
    diam=1., 
    stopping_criterion=StopWhenSubgradientNormLess(1e-8),
    # debug=[:Stop],
    debug=["    ", :Iteration, (:Cost,"F(p): %1.20e"), "\n", :Stop, 1]
)
b2 = bundle_method(N, f, gradf, img2; 
    diam=1., 
    stopping_criterion=StopWhenBundleLess(1e-10),
    # debug=[:Stop],
    debug=["    ", :Iteration, (:Cost,"F(p): %1.20e"), "\n", :Stop, 1]
)
println("Subgradient method:")
s = subgradient_method(N, f, gradf, img2; 
    stopping_criterion=StopWhenSubgradientNormLess(1e-8),
    # debug=[:Stop],
    debug=["    ", :Iteration, (:Cost,"F(p): %1.20e"), "\n", :Stop, 1]
)

# prox_f = prox_TV2(M, 0.5, (data[1,2],data[1,3],data[1,4]))
#cyclic_proximal_point(N, f, proxf, p0; debug=["    ", :Iteration, (:Cost,"F(p): %1.20e"), "\n", :Stop, 1])

asymptote_export_SPD("initial.fig"; data=img, scale_axes=(4.,4.,4.), color_scheme=ColorSchemes.hsv)
render_asymptote("initial.fig")
asymptote_export_SPD("noisy.fig"; data=img2, scale_axes=(12.,12.,12.), color_scheme=ColorSchemes.hsv)
render_asymptote("noisy.fig")
asymptote_export_SPD("bundle_subnorm.fig"; data=b1, scale_axes=(12.,12.,12.), color_scheme=ColorSchemes.hsv)
render_asymptote("bundle_subnorm.fig")
asymptote_export_SPD("bundle_bundless.fig"; data=b2, scale_axes=(12.,12.,12.), color_scheme=ColorSchemes.hsv)
render_asymptote("bundle_bundless.fig")
asymptote_export_SPD("subgradient.fig"; data=s, scale_axes=(12.,12.,12.), color_scheme=ColorSchemes.hsv)
render_asymptote("subgradient.fig")