using Manopt, Manifolds, LinearAlgebra, Test, Chairmarks
using CodecBzip2
using StaticArrays, RecursiveArrayTools

struct BALObservation{T<:Real, I<:Integer}
	camera_index::I
	point_index::I
	xy::SVector{2, T}
end

struct BALCamera{T<:Real}
	R::SMatrix{3, 3, T, 9}
	t::SVector{3, T}
	f::T
	k1::T
	k2::T
end

const BALPoint{T} = SVector{3, T}

struct BALDataset{T<:Real, I<:Integer}
	num_cameras::Int
	num_points::Int
	num_observations::Int
	observations::Vector{BALObservation{T, I}}
	cameras::Vector{BALCamera{T}}
	points::Vector{BALPoint{T}}
end

function _skew(v::NTuple{3, T}) where {T<:Real}
	vx, vy, vz = v
	return @SMatrix T[
		zero(T) -vz vy
		vz zero(T) -vx
		-vy vx zero(T)
	]
end

"""
	rodrigues_to_rotation_matrix(r)

Convert a Rodrigues vector `r = (r1, r2, r3)` to a 3×3 rotation matrix.
"""
function rodrigues_to_rotation_matrix(r::NTuple{3, T}) where {T<:Real}
	θ2 = r[1]^2 + r[2]^2 + r[3]^2
	θ = sqrt(θ2)

	A, B = if θ < sqrt(eps(T))
		(
			one(T) - θ2 / 6 + θ2^2 / 120,
			inv(T(2)) - θ2 / 24 + θ2^2 / 720,
		)
	else
		(sin(θ) / θ, (one(T) - cos(θ)) / θ2)
	end

	K = _skew(r)
	I3 = one(SMatrix{3, 3, T, 9})
	return I3 + A * K + B * (K * K)
end

"""
	project_point(camera, point)

Project a world-space `BALPoint` into image coordinates using the BAL camera model.
The model uses Rodrigues rotation `R`, translation `t`, focal length `f`, and radial distortion `(k1, k2)`.
Returns `SVector{2, T}`.
"""
function project_point(camera::BALCamera{T}, point::BALPoint{T}) where {T<:Real}
	R = camera.R
	xc, yc, zc = R * point + camera.t

    @show point
    @show camera
    @show xc yc zc

	abs(zc) > eps(T) || throw(DomainError(zc, "Point projects to infinity (z≈0 in camera frame)."))

	xn = -xc / zc
	yn = -yc / zc
	r2 = xn^2 + yn^2
	radial = one(T) + camera.k1 * r2 + camera.k2 * r2^2

	return SVector{2, T}(camera.f * radial * xn, camera.f * radial * yn)
end

"""
	reprojection_error(camera, point, observation)

Compute the reprojection residual as `SVector{2, T}`:
`project_point(camera, point) - observation.xy`.
"""
function reprojection_error(camera::BALCamera{T}, point::BALPoint{T}, obs_xy::AbstractVector) where {T<:Real}
	return project_point(camera, point) - obs_xy
end

function _next_nonempty_line!(state::Base.Iterators.Stateful)
	while !isempty(state)
		line = strip(popfirst!(state))
		if !isempty(line)
			return line
		end
	end
	throw(EOFError())
end

"""
	read_bal_bz2(path; one_based_indices=true, T=Float64, I=Int)

Read a bzip2-compressed BAL dataset from the given `path`.
Each camera is parsed as 9 parameters `(r, t, f, k1, k2)` from BAL, where `r` is a Rodrigues vector,
and stored as a 3×3 rotation matrix in `BALCamera.R`.
Each point is parsed as 3D coordinates `(x, y, z)`.
Returns a `BALDataset`.
"""
function read_bal_bz2(path::AbstractString; one_based_indices::Bool=true, T::Type=Float64, I::Type=Int)
	open(path, "r") do raw_io
		io = Bzip2DecompressorStream(raw_io)
		try
			lines = Base.Iterators.Stateful(eachline(io))

			header = split(_next_nonempty_line!(lines))
			length(header) == 3 || throw(ArgumentError("Header must contain exactly 3 fields"))

			num_cameras = parse(Int, header[1])
			num_points = parse(Int, header[2])
			num_observations = parse(Int, header[3])

			observations = Vector{BALObservation{T, I}}(undef, num_observations)
			idx_shift = one_based_indices ? 1 : 0

			for k in 1:num_observations
				fields = split(_next_nonempty_line!(lines))
				length(fields) == 4 || throw(ArgumentError("Observation line $k must have 4 fields"))

				cam_idx = parse(I, fields[1]) + idx_shift
				pt_idx = parse(I, fields[2]) + idx_shift
				x = parse(T, fields[3])
				y = parse(T, fields[4])

				observations[k] = BALObservation{T, I}(cam_idx, pt_idx, SVector{2, T}(x, y))
			end

			remaining_tokens = String[]
			while !isempty(lines)
				line = strip(popfirst!(lines))
				isempty(line) && continue
				append!(remaining_tokens, split(line))
			end

			expected_values = 9 * num_cameras + 3 * num_points
			length(remaining_tokens) == expected_values || throw(
				ArgumentError(
					"Expected $expected_values camera/point values, got $(length(remaining_tokens))",
				),
			)

			values = parse.(T, remaining_tokens)
			cursor = 1

			cameras = Vector{BALCamera{T}}(undef, num_cameras)
			for k in 1:num_cameras
				r = (values[cursor], values[cursor + 1], values[cursor + 2])
				t = SVector{3, T}(values[cursor + 3], values[cursor + 4], values[cursor + 5])
				f = values[cursor + 6]
				k1 = values[cursor + 7]
				k2 = values[cursor + 8]
				R = rodrigues_to_rotation_matrix(r)
				cameras[k] = BALCamera{T}(R, t, f, k1, k2)
				cursor += 9
			end

			points = Vector{BALPoint{T}}(undef, num_points)
			for k in 1:num_points
				points[k] = BALPoint{T}(values[cursor], values[cursor + 1], values[cursor + 2])
				cursor += 3
			end

			return BALDataset{T, I}(
				num_cameras,
				num_points,
				num_observations,
				observations,
				cameras,
				points,
			)
        catch e
            rethrow(e)
        finally
            close(io)
        end
	end
end

# We can dowload data from here: https://grail.cs.washington.edu/projects/bal/
data1 = read_bal_bz2("/home/mateusz/data/bal/ladybug/problem-49-7776-pre.txt.bz2")


struct Fi_block{TD<:BALDataset}
    dataset::TD
    obs_idx::Int
end


function (f::Fi_block)(M::AbstractManifold, r, p)
    M_cam, M_t, M_pt = M.manifolds
    p_cam, p_t, p_pt = p.x
    obs = f.dataset.observations[f.obs_idx]

    cam = BALCamera(
        SMatrix{3,3}(p_cam[M_cam, obs.camera_index]),
        SVector{3}(p_t[M_t, obs.camera_index]),
        400.0, # focal length is not optimized in this example
        0.0, # radial distortion k1 is not optimized
        0.0, # radial distortion k2 is not optimized
    )
    pt_idx = f.dataset.observations[f.obs_idx].point_index
    r .= reprojection_error(cam, SVector{3}(p_pt[M_pt, pt_idx]), obs.xy)
end

struct jacFi_block{TD<:BALDataset}
    dataset::TD
    obs_idx::Int
end

function (f::jacFi_block)(M::AbstractManifold, J, p)
    # TODO
end

function run_bundle_adjustment(data::BALDataset)
    M = ProductManifold(
        PowerManifold(Rotations(3), ArrayPowerRepresentation(), data.num_cameras), # camera rotations
        PowerManifold(Euclidean(3), ArrayPowerRepresentation(), data.num_cameras), # camera translations
        PowerManifold(Euclidean(3), ArrayPowerRepresentation(), data.num_points), # 3D point positions
    )

    F = [Fi_block(data, i) for i in 1:data.num_observations]
    JF = [jacFi_block(data, i) for i in 1:data.num_observations]

    f = [VectorGradientFunction(
        F[i], JF[i], 2;
        evaluation = InplaceEvaluation(),
        function_type = FunctionVectorialType(),
        jacobian_type = CoefficientVectorialType(DefaultOrthonormalBasis())
    ) for i in 1:data.num_observations]

    p0 = ArrayPartition(
        stack([Matrix{Float64}(I, 3, 3) for _ in 1:data.num_cameras]), # camera rotations
        ones(3, data.num_cameras), # camera translations
        zeros(3, data.num_points), # 3D point positions
    )

    hr = fill((1 / 30) ∘ HuberRobustifier(), length(F))

    q = LevenbergMarquardt(
        M, f, p0;
        β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = 1.0e-1, α_mode = :Strict,
        robustifier = hr,
        debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop, 5],
        # sub_state = CoordinatesNormalSystemState(M),
    )

    return q
end

run_bundle_adjustment(data1)
