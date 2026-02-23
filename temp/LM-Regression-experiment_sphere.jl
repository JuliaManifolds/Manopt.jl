#
#
# --- Regession on the sphere with outliers
using Distributions, GLMakie, Makie, ManifoldDiff, Manifolds, Manopt, NamedColors, Random, RecursiveArrayTools
ptc = NamedColors.load_paul_tol()

# Parameters
export_asy = true
show_plots = true
add_gaussian_noise = true
σ = 0.1
# For outliers we use a fixed size and a random angle to disturb them into
r = 0.4
N = 39 # on the range these are 0.05 apart for 39
outlier_indices = [2,10,23,29]

S = Manifolds.Sphere(2)
M = TangentBundle(S)

# True data
p_true = [0.0, 1.0, 0.0]
X_true = 1.5 .* [1.0, 0.0, 1.0]
ts_true = collect(range(; start = -0.95, stop = 0.95, length=N))
qs_true = geodesic(S, p_true, X_true, ts_true)

if show_plots
    n = 30; u = range(0,stop=2*π,length=n); v = range(0,stop=π,length=n);
    sx = [cos(ui) * sin(vj) for ui in u, vj in v]
    sy = [sin(ui) * sin(vj) for ui in u, vj in v]
    sz = [cos(vj) for ui in u, vj in v]

    fig1 = Figure(size = (1400, 900), padding=0)
    ax1 = Axis3(fig1[1,1]; aspect =:data)
    hidedecorations!(ax1)
    hidespines!(ax1)
 	wireframe!(ax1, sx, sy, sz, color = ptc["paleblue"]; transparency=true, alpha = 0.2)
    geo_line = geodesic(S, p_true, X_true, range(-1.0, 1.0; length=1000))
    scatterlines!(
       ax1, Point3d.(geo_line); markersize=0, color=ptc["mutedteal"], linewidth=2,
    )
    scatter!(ax1, Point3d.([p_true,]); markersize=12, color=ptc["mutedteal"])
    scatter!(ax1, Point3d.(qs_true); markersize=8, color=ptc["mutedteal"])
    arrows3d!(
        ax1, Point3d.([p_true,]), Point3d.([X_true,]);
        color=ptc["mutedteal"], transparency=true, shaftradius=0.005, tiplength = 0.15, tipradius=0.025,
    )
end
Random.seed!(42)
data = [
    if i ∈ outlier_indices
        # sample random angle
        α = rand(Uniform(0,2π))
        X_noise = get_vector(S, q, r .* [sin(α), cos(α)], DefaultOrthonormalBasis())
        exp(S, q, X_noise)
    else
        exp(S, q, get_vector(S, q, add_gaussian_noise ? σ*randn(2) : zeros(2), DefaultOrthonormalBasis()))
    end for (i,q) in enumerate(qs_true)
]
show_plots && scatter!(ax1, Point3d.(data); markersize=12, color=ptc["mutedrose"])

# Cost (Vectorial) and its gradients (Jacobian) on the tangent bundle
# maybe as a struct / functor ?
function F(M, P; t=ts_true, d=data)
    S = base_manifold(M)
    p = P[M,:point]
    X = P[M, :vector]
    return [distance(S, geodesic(S, p, X, ti), di) for (ti,di) in zip(t,d)]
end
function JF(M, P; t=ts_true, d=data)
    S = base_manifold(M)
    p = P[M,:point]
    X = P[M, :vector]
    return [
        ArrayPartition(
            [ ... ] ?
        )
        for (ti,di) in zip(t,d)
    ]
end

fig1