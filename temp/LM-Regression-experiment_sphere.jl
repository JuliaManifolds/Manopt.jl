#
#
# --- Regession on the sphere with outliers
using Colors, Distributions, GLMakie, Makie, ManifoldDiff, Manifolds, Manopt, NamedColors, Random, RecursiveArrayTools
ptc = NamedColors.load_paul_tol()

# Parameters
export_asy = false
show_plots = false
add_gaussian_noise = true
name = "S2-Robust-Regression"
σ = 1 * π / 32
# For outliers we use a fixed size and a random angle to disturb them into
r = π / 4
N = 100 # on the range these are 0.05 apart for 39
oN = 7
outlier_indices = [8:(8 + oN - 1)..., [(N - 7):-1:(N - 7 - oN + 1)...]...]


S = Manifolds.Sphere(2)
M = TangentBundle(S)
R(α) = [cos(α) -sin(α); sin(α) cos(α)]
T = 1.0

# True data
p_true = [0.0, 1.0, 0.0]
X_true = π / 2 .* [1.0, 0.0, 1.0]
ts_true = collect(range(; start = -T, stop = T, length = N))
qs_true = geodesic(S, p_true, X_true, ts_true)
geo_line = geodesic(S, p_true, X_true, range(-T, T; length = 1000))

orig_color = ptc["mutedcyan"]
noisy_color = ptc["mutedwine"]

lsq_color = ptc["mutedsand"]
robust_color = ptc["mutedgreen"]


if show_plots
    n = 30; u = range(0, stop = 2 * π, length = n); v = range(0, stop = π, length = n)
    sx = [cos(ui) * sin(vj) for ui in u, vj in v]
    sy = [sin(ui) * sin(vj) for ui in u, vj in v]
    sz = [cos(vj) for ui in u, vj in v]

    fig1 = Figure(size = (1400, 900), padding = 0)
    ax1 = Axis3(fig1[1, 1]; aspect = :data)
    hidedecorations!(ax1)
    hidespines!(ax1)
    wireframe!(ax1, sx, sy, sz, color = ptc["paleblue"]; transparency = true, alpha = 0.2)
    scatterlines!(
        ax1, Point3d.(geo_line); markersize = 0, color = orig_color, linewidth = 2,
    )
    scatter!(ax1, Point3d.([p_true]); markersize = 12, color = orig_color)
    scatter!(ax1, Point3d.(qs_true); markersize = 8, color = orig_color)
    arrows3d!(
        ax1, Point3d.([p_true]), Point3d.([X_true]);
        color = orig_color, transparency = true, shaftradius = 0.0025, tiplength = 0.075, tipradius = 0.0125,
    )
end
Random.seed!(42)
data = [
    if i ∈ outlier_indices
            # sample random angle
            # α = rand(Uniform(0,2π))
            # all are outliers to the left or right
            α = π / 2
            c = get_coordinates(S, q, parallel_transport_to(S, p_true, X_true, q), DefaultOrthonormalBasis())
            X_noise = get_vector(S, q, r / norm(c) .* R(α) * c, DefaultOrthonormalBasis())
            exp(S, q, X_noise)
    else
            exp(S, q, get_vector(S, q, add_gaussian_noise ? σ * randn(2) : zeros(2), DefaultOrthonormalBasis()))
    end for (i, q) in enumerate(qs_true)
]
show_plots && scatter!(ax1, Point3d.(data); markersize = 12, color = noisy_color)

# Cost (Vectorial) and its gradients (Jacobian) on the tangent bundle
# maybe as a struct / functor ?
function F(M, P; t = ts_true, d = data)
    S = base_manifold(M)
    p = P[M, :point]
    X = P[M, :vector]
    return [distance(S, geodesic(S, p, X, ti), di) for (ti, di) in zip(t, d)]
end
function JF(M, P; t = ts_true, d = data)
    S = base_manifold(M)
    p = P[M, :point]
    X = P[M, :vector]
    return [
        ArrayPartition(
                cost1_grad_p(S, p, X, ti, di),
                cost1_grad_X(S, p, X, ti, di),
            )
            for (ti, di) in zip(t, d)
    ]
end

function cost1(M::AbstractManifold, p, X, ti::Real, di)
    return 1 / 2 * distance(M, exp(M, p, ti * X), di)^2
end

function cost1_grad_p(M::AbstractManifold, p, X, ti::Real, di)
    z = exp(M, p, ti * X)
    gz = ManifoldDiff.grad_distance(M, di, z, 1)
    return ManifoldDiff.adjoint_differential_exp_basepoint(M, p, ti * X, gz)
end

function cost1_grad_X(M::AbstractManifold, p, X, ti::Real, di)
    z = exp(M, p, ti * X)
    gz = ManifoldDiff.grad_distance(M, di, z, 1)
    return ti * ManifoldDiff.adjoint_differential_exp_argument(M, p, ti * X, gz)
end

# This check requires plots which does not play so well with Makie
# Maybe also write a Makie extension?
#
# p0 = rand(S)
# X0 = rand(S; vector_at=p0)
# p1 = rand(S)
# Manopt.check_gradient(S,
#     (M, p) -> cost1(M, p, parallel_transport_to(M, p0, X0, p), 0.5, p1),
#     (M, p) -> cost1_grad_p(M, p, parallel_transport_to(M, p0, X0, p), 0.5, p1),
#     p0; plot=true
#     )

# Manopt.check_gradient(TangentSpace(S, p0),
#     (M, X) -> cost1(S, p0, X, 0.5, p1),
#     (M, X) -> cost1_grad_X(S, p0, X, 0.5, p1),
#     X0; plot=true
#     )

f = VectorGradientFunction(
    F, JF, N;
    evaluation = AllocatingEvaluation(),
    function_type = FunctionVectorialType(),
    jacobian_type = FunctionVectorialType(),
)

m = mean(S, data)
X0 = log(S, m, data[end])
p0 = ArrayPartition(m, X0)

@show p0

# Least Squares
P_star = LevenbergMarquardt(
    M, f, p0;
    damping_increase_factor = 2.0, candidate_acceptance_threshold = 0.2, damping_term_min = 1.0e-5,
    damping_increase_threshold = 0.2,
    damping_reduction_threshold = 0.5,
    scaling_threshold = 1.0e-1, scaling_mode = :Strict,
    retraction_method = StabilizedRetraction(default_retraction_method(M)),
    debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop],
)
p_star = P_star[M, :point]
X_star = P_star[M, :vector]

@show p0

qs_star = geodesic(S, p_star, X_star, ts_true)

# Robust
P_ast = LevenbergMarquardt(
    M, f, p0;
    damping_increase_factor = 8.0, candidate_acceptance_threshold = 0.2, damping_term_min = 1.0e-5,
    scaling_threshold = 1.0e-1, scaling_mode = :Strict,
    damping_increase_threshold = 0.2,
    damping_reduction_threshold = 0.5,
    robustifier = 1.0e-4 ∘ HuberRobustifier(),
    retraction_method = StabilizedRetraction(default_retraction_method(M)),
    debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop],
)
@show P_ast
p_ast = P_ast[M, :point]
X_ast = P_ast[M, :vector]
qs_ast = geodesic(S, p_ast, X_ast, ts_true)

geo_line_mean = geodesic(S, p_star, X_star, range(-T, T; length = 1000))
geo_line_robust = geodesic(S, p_ast, X_ast, range(-T, T; length = 1000))
if show_plots
    scatterlines!(
        ax1, Point3d.(geo_line_mean); markersize = 0, color = lsq_color, linewidth = 2,
    )
    scatter!(ax1, Point3d.([p_star]); markersize = 12, color = lsq_color)
    scatter!(ax1, Point3d.(qs_star); markersize = 8, color = lsq_color)
    arrows3d!(
        ax1, Point3d.([p_star]), Point3d.([X_star]);
        color = lsq_color, transparency = true, shaftradius = 0.0025, tiplength = 0.075, tipradius = 0.0125,
    )

    scatterlines!(
        ax1, Point3d.(geo_line_robust); markersize = 0, color = robust_color, linewidth = 2,
    )
    scatter!(ax1, Point3d.([p_ast]); markersize = 12, color = robust_color)
    scatter!(ax1, Point3d.(qs_ast); markersize = 8, color = robust_color)
    arrows3d!(
        ax1, Point3d.([p_ast]), Point3d.([X_ast]);
        color = robust_color, transparency = true, shaftradius = 0.0025, tiplength = 0.075, tipradius = 0.0125,
    )
end

if export_asy
    kwargs = (;
        camera_position = (0.75, 0.5, 0.125),
        arrow_head_size = 18.0,
        dot_sizes = 4 .* [2.5, 2.5, 2.5, 2.5],
        line_width = 4.0,
        sphere_line_width = 2.0,
        size = (1024, 1024),
    )
    tvec_scale = 0.5
    asymptote_export_S2_signals(
        name * "-orig.asy";
        curves = [geo_line],
        points = [data, qs_true],
        tangent_vectors = [[(p_true, tvec_scale .* X_true)]],
        colors = Dict(
            :curves => Colors.RGBA{Float64}.([orig_color]),
            :tvectors => Colors.RGBA{Float64}.([orig_color]),
            :points => Colors.RGBA{Float64}.([noisy_color, orig_color]),
        ),
        kwargs...
    )
    render_asymptote(name * "-orig.asy"; render = 4)
    asymptote_export_S2_signals(
        name * ".asy";
        curves = [geo_line, geo_line_mean, geo_line_robust],
        points = [data, qs_star, qs_ast, qs_true],
        tangent_vectors = [[(p_star, tvec_scale .* X_star)], [(p_ast, tvec_scale .* X_ast)], [(p_true, tvec_scale .* X_true)]],
        colors = Dict(
            :curves => Colors.RGBA{Float64}.([orig_color, lsq_color, robust_color]),
            :tvectors => Colors.RGBA{Float64}.([lsq_color, robust_color, orig_color]),
            :points => Colors.RGBA{Float64}.([noisy_color, lsq_color, robust_color, orig_color]),
        ),
        kwargs...
    )
    render_asymptote(name * ".asy"; render = 4)
end

@info "Mean error on sample points least squares: $(1 / N * norm([distance(S, qi, qmi) for (qi, qmi) in zip(qs_true, qs_star)]))"
@info "Mean error on sample points robust: $(1 / N * norm([distance(S, qi, qri) for (qi, qri) in zip(qs_true, qs_ast)]))"
show_plots && fig1
