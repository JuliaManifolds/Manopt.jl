using Manopt, Manifolds, Makie, Colors, LinearAlgebra, ColorSchemes
#
# Settings
n_lon = 201
n_lat = 101
lat = range(-π / 2, π / 2; length = n_lat)
lon = range(-π, π; length = n_lon)
#
# Colors
sphere_color = colorant"rgb(223 186 105)"
plane_color = colorant"rgb(255 254 223)"
edge_color = colorant"rgb(85 85 85)"
dot_color = edge_color
vec_color = dot_color
geod_color = dot_color

A = [8.0 1.0 6.0; 3.0 5.0 7.0; 4.0 9.0 2.0]
A = A + A'
A = A ./ opnorm(A)
F(p) = p' * A * p
spherical_to_euclidean(θ, φ, r = 1) = [r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ)]
M = Manifolds.Sphere(2)

mesh_x = [spherical_to_euclidean(θ, φ)[1] for θ in lat, φ in lon]
mesh_y = [spherical_to_euclidean(θ, φ)[2] for θ in lat, φ in lon]
mesh_z = [spherical_to_euclidean(θ, φ)[3] for θ in lat, φ in lon]

mesh_pts = [spherical_to_euclidean(θ, φ) for θ in lat, φ in lon]
data = [F(spherical_to_euclidean(θ, φ)) for θ in lat, φ in lon]

manopt_pre_scheme = ColorScheme(
    [sphere_color, plane_color],
    "Manopt",
    "The Sphere color Scheme for the Manopt Logo",
)
manopt_scheme = ColorScheme(
    [get(manopt_pre_scheme, i) for i in 0.0:0.01:1.0],
    "Manopt",
    "The Sphere color Scheme for the Manopt Logo",
)

# mesh(Makie.Sphere(Point3f0(0), 1f0), color=data, colormap = manopt_pre_scheme, levels=5, interpolate=false)
contour3d(lat, lon, data, levels = 11, color = :black, linewidth = 3)
