### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 85e76846-912a-11ef-294a-c717389928e4
using Pkg; Pkg.activate();

# ╔═╡ 48950604-c2c2-4310-8de5-f89db905668b
begin
	using LinearAlgebra
	using SparseArrays
	using Manopt
	using Manifolds
	using ManoptExamples
	using OffsetArrays
	using Random
    using WGLMakie, Makie, GeometryTypes, Colors, NamedColors, ColorSchemes
	using FileIO, VideoIO, ProgressLogging
end;

# ╔═╡ fe0c1524-b5f4-4afc-aee7-bd2de9d482b6
begin
	# Hack fix.
	using ManifoldsBase
	using ManifoldsBase: PowerManifoldNested, get_iterator, _access_nested, _read, _write
	import ManifoldsBase: _get_vectors
	function _get_vectors(
    M::PowerManifoldNested,
    p,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:PowerBasisData},
) where {𝔽}
    zero_tv = zero_vector(M, p)
    rep_size = representation_size(M.manifold)
    vs = typeof(zero_tv)[]
    for i in get_iterator(M)
        b_i = _access_nested(M, B.data.bases, i)
        p_i = _read(M, rep_size, p, i)
        # println(get_vectors(M.manifold, p_i, b_i))
        for v in get_vectors(M.manifold, p_i, b_i) #b_i.data
            new_v = copy(M, p, zero_tv)
            copyto!(M.manifold, _write(M, rep_size, new_v, i), p_i, v)
            push!(vs, new_v)
        end
    end
    return vs
end
end

# ╔═╡ f96fcfed-cfa4-44b1-9034-5c3aedc75aa1
md"""
# Parameters
"""

# ╔═╡ bb8b3dfe-cc36-4949-b33a-4c77436a8416
begin
	N=500 # Number of sample values for the discretisation
	ex = 2 # exponent used in the energy functional
	scale = 12.0 # Integrand scaling
	scale_range = range(0.0, scale; length=500)
	file_name_prefix = "flugplanung_N$(N)_p$(ex)"
	temp_folder = "frames"
	!isdir(temp_folder) && mkdir(temp_folder)
end;

# ╔═╡ 6b1bca0b-f209-445e-8f41-b19372c9fffc
begin
	st = 0.5
	halt = pi-0.5
	h = (halt-st)/(N+1)
	#halt = pi - st
	Omega = range(; start=st, stop = halt, length=N+2)[2:end-1]
	y0 = [sin(st),0,cos(st)] # startpoint of geodesic
	yT = [sin(halt),0,cos(halt)] # endpoint of geodesic
end;

# ╔═╡ bd50d1db-7c55-4279-abe0-89f2fe986cc6
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
end

# ╔═╡ 449e9782-5bed-4642-a068-f8c9106bbe86
y(t) = [sin(t), 0, cos(t)]

# ╔═╡ 4ea7c8f7-80f9-482a-b535-229d2671fb4d
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ f08571fe-2e5d-417c-8330-9251233af25d
"""
Such a structure has to be filled for two purposes:
* Definition of an integrand and its derivative
* Definition of a vector transport and its derivative
"""
mutable struct DifferentiableMapping{M<:AbstractManifold,F1<:Function,F2<:Function,T}
	domain::M
	value::F1
	derivative::F2
	scaling::T
end


# ╔═╡ 1918a11e-88b9-43b9-b52d-026163580b5f
"""
 The following two routines define the vector transport and its derivative. The second is needed to obtain covariant derivative from the ordinary derivative.

I know: the first is already implemented, but this is just for demonstration purpose
"""
function transport_by_proj(S, p, X, q)
	return X - q*(q'*X)
end

# ╔═╡ d47accca-ebf3-4085-9d20-2e0ee6e88ef5
function transport_by_proj_prime(S, p, X, dq)
	return (- dq*p' - p*dq')*X
end

# ╔═╡ 28143183-eccb-4166-9f79-6baaa8f3f5c2
"""
Definition of a vector transport and its derivative given by the orthogonal projection
"""
transport=DifferentiableMapping(S,transport_by_proj,transport_by_proj_prime,nothing)

# ╔═╡ a789119e-04b9-4456-acba-ec8e8702c231
"""
	Evaluates the wind field at a point p on the sphere (here: winding field scaled by the third component)
"""
	function w(M, p, c)
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0]
	end

# ╔═╡ a5811f04-59f8-4ea2-8616-efad7872830a
"""
	Returns the first derivative of the wind field at point p as a matrix
"""
function w_prime(M, p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end

# ╔═╡ 0e5275a1-52d0-45b2-8d1c-5940b50afde8
"""
The following two routines define the integrand and its Euclidean derivative. They use a wind field w, its derivative and its second derivative, defined below. A scaling parameter is also employed.
"""
function F_at(Integrand, y, ydot, B, Bdot)
	return ex*(((ydot - w(Integrand.domain, y, Integrand.scaling))'*(ydot - w(Integrand.domain, y, Integrand.scaling)))^(ex/2.0 - 1))*(Bdot - w_prime(Integrand.domain,y,Integrand.scaling)*B)'*(ydot - w(Integrand.domain, y, Integrand.scaling))
end

# ╔═╡ 709daed1-7ef3-4fa5-b390-e18d27a7cefd
"""
	Returns the second derivative of the wind field at point p in direction v as a matrix
"""
function w_doubleprime(M, p, v, c)
	nenner = (p[1]^2+p[2]^2)
	w1 = 1/(nenner^2)*[(2*p[2]*p[3]*nenner-8*p[1]^2*p[2]*p[3])/nenner -p[3]*(2*p[1]*nenner^2-4*p[1]*(p[1]^4-p[2]^4))/nenner^2 2*p[1]*p[2]; (-2*p[1]*p[3]*nenner^2-4*p[1]*p[3]*(p[2]^4-p[1]^4))/nenner^2 (-2*p[2]*p[3]*nenner+8*p[1]^2*p[2]*p[3])/nenner (p[2]^2-p[1]^2); 0.0 0.0 0.0]

	w2 = 1/(nenner^2)*[(2*p[1]*p[3]*nenner-8*p[1]*p[2]^2*p[3])/nenner -p[3]*(-2*p[2]*nenner^2-4*p[2]*(p[1]^4-p[2]^4))/nenner^2 p[2]^2-p[1]^2; (2*p[2]*p[3]*nenner^2-4*p[2]*p[3]*(p[2]^4-p[1]^4))/nenner^2 (-2*p[1]*p[3]*nenner+8*p[1]*p[2]^2*p[3])/nenner -2*p[1]*p[2]; 0.0 0.0 0.0]

	w3 = 1/(nenner^2)*[2*p[1]*p[2] -p[1]^2+p[2]^2 0.0; p[2]^2-p[1]^2 -2*p[1]*p[2] 0.0; 0.0 0.0 0.0]
	return c*(v[1]*w1 + v[2]*w2 + v[3]*w3)
end

# ╔═╡ 8c726811-c87e-4924-9ebe-bc56e26855a9
function F_prime_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	S = Integrand.domain
	constant = Integrand.scaling
	return ex*(ex-2)*(((ydot - w(S, y, constant))'*(ydot - w(S, y, constant)))^(ex/2.0 - 2.0))*((B2dot - w_prime(S, y, constant)*B2)'*(ydot - w(S, y, constant)))*((B1dot - w_prime(S, y, constant)*B1)'*(ydot - w(S, y, constant))) + ex * (((ydot - w(S, y, constant))'*(ydot - w(S, y, constant)))^(ex/2.0 - 1.0)) * ((-1.0*w_doubleprime(S, y, B2, constant)*B1)'*(ydot - w(S, y, constant)) + (B1dot - w_prime(S, y, constant)*B1)'*(B2dot - w_prime(S, y, constant)
	*B2))
end

# ╔═╡ 7f9448c8-0f0c-40de-861f-16427fd335ad
"""
	Definition of an integrand and its derivative for the simplified flight planning problem
"""
integrand=DifferentiableMapping(S,F_at,F_prime_at,scale)

# ╔═╡ 5c13ce80-209c-4901-913a-3283339a11cc
"""
	Dummy, necessary for calling vectorbundle_newton
"""
function bundlemap(M, y)
		return y
end

# ╔═╡ 86a649c8-43f8-43a0-a652-24735d28c05e
"""
	Dummy, necessary for calling vectorbundle_newton
"""
function connection_map(E, q)
    return q
end

# ╔═╡ fbf47501-4a41-4f89-940c-9c68aca26b4b
"""
	Method for solving the Newton equation
		* assembling the Newton matrix and the right hand side (using ManoptExamples.jl)
		* using a direct solver for computing the solution in base representation
	Returns the Newton direction in vector representation
"""
function solve_linear_system(M, p, state, prob)
	obj = get_objective(prob)
	n = manifold_dimension(M)

	Ac::SparseMatrixCSC{Float64,Int32} =spzeros(n,n)
	bc = zeros(n)
	bcsys=zeros(n)
	bctrial=zeros(n)
	Oy = OffsetArray([y0, p..., yT], 0:(length(Omega)+1))
	Oytrial = OffsetArray([y0, state.p_trial..., yT], 0:(length(Omega)+1))
	S = M.manifold

	# println("Assemble:")
	#@time
	ManoptExamples.get_rhs_Jac!(bc,Ac,h,Oy,integrand,transport)
	if state.is_same == true
		bcsys=bc
	else
		#@time
		ManoptExamples.get_rhs_simplified!(bctrial,h,Oy,Oytrial,integrand,transport)
    	bcsys=bctrial-(1.0 - state.stepsize.alpha)*bc
	end
	#println("Solve:")
	#@time
	Xc = (Ac) \ (-bcsys)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	res_c = get_vector(M, p, Xc, B)
	return res_c
end

# ╔═╡ b2d6b477-28d1-421c-958e-ebcfe845a6bb
"""
	Set the solve method for solving the Newton equation in each step
"""
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, newtonstate.p, newtonstate, problem)

# ╔═╡ 25cdf23f-c2ff-44be-9a34-bb565c36775e
"""
	Initial geodesic
"""
y_0 = copy(power, discretized_y)

# ╔═╡ 7a56880f-7ecc-4cc7-a407-97d0f842de3e
solutions = [copy(power, y_0) for s in scale_range];

# ╔═╡ dcf6174d-0b6f-48da-b0d8-f057b2992e0b
@progress "Increasing wind" for (i,s) in enumerate(scale_range)
	# that integrand is taken from global scope is not yet optimal here.
	integrand.scaling = s
	# Take last result as start
	if i > 1
		copyto!(power, solutions[i], solutions[i-1])
	end
	# compute in-place
	vectorbundle_newton!(
		power, TangentBundle(power), bundlemap, bundlemap, connection_map,
		solutions[i];
		sub_problem=solve,
		sub_state=AllocatingEvaluation(),
		stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
		retraction_method=ProjectionRetraction(),
	)
end

# ╔═╡ 48e0d10c-23c4-48c4-91fa-b2a5ae443005
p_res = solutions[17]

# ╔═╡ 88c316c6-4463-4e42-abc8-d83fc117e56f
begin
n = 45
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);
sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)

ws = [1.0*w(Manifolds.Sphere(2), p, scale) for p in p_res]
for i in 1:n
    for j in 1:n
        sx[i,j] = cos.(u[i]) * sin(v[j]);
        sy[i,j] = sin.(u[i]) * sin(v[j]);
        sz[i,j] = cos(v[j]);
    end
end
fig, ax, plt = meshscatter(
  sx,sy,sz,
  color = fill(RGBA(1.,1.,1.,0.75), n, n),
  shading = Makie.automatic,
  transparency=true
)
ax.show_axis = false
wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.45); transparency=true)
    π1(x) = 1.02*x[1]
    π2(x) = 1.02*x[2]
    π3(x) = 1.02*x[3]
	scatter!(ax, π1.(p_res), π2.(p_res), π3.(p_res); markersize =8, color=:orange)
	scatter!(ax, π1.(y_0), π2.(y_0), π3.(y_0); markersize =8, color=:blue)
	scatter!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =8, color=:red)
	arrows!(ax, π1.(p_res), π2.(p_res), π3.(p_res), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)
	fig
end

# ╔═╡ 76ecbb61-431d-45b6-a114-134d20d71a55
begin
	paul_tol = load_paul_tol()
	# We have to trick with the colors a bit because the export is a bit too restrictive.
	indigo = RGBA(paul_tol["mutedindigo"])
	green = RGBA(paul_tol["mutedgreen"])
	sand = RGBA(paul_tol["mutedsand"])
	teal = RGBA{Float64}(paul_tol["mutedteal"])
	grey = RGBA(paul_tol["mutedgrey"])
	curve_colors = get(ColorSchemes.viridis, range(0.0, 1.0, length=length(scale_range)))
end

# ╔═╡ 812157bc-6de5-4817-a770-9d86cce8d59b
begin
	render = true
	render && @progress "Rendering images" for (i,s) in enumerate(scale_range)
		file_name = joinpath(temp_folder, file_name_prefix*"-$(lpad(string(i), 6,"0"))")
		#local force field
		ws_local = [1.0*w(Manifolds.Sphere(2), p, s) for p in solutions[i]]		
		asymptote_export_S2_signals(file_name*".asy";
		points= [ [y0,yT] ],
		curves = [y_0, solutions[i]],
		tangent_vectors = [
			[ Tuple(a) for a in zip(solutions[i], 1/50 .* ws_local)]
		],
		dot_sizes = [3.0,],
		line_widths = [0.8, 1., .25], #2 curves, tangent vectors
		arrow_head_size = 1.5,
		colors = Dict{Symbol, Vector{ColorTypes.RGBA{Float64}}}(
			:points => [teal],
			:curves => [teal, curve_colors[i]],
			:tvectors => [green],
		),
		camera_position=(2.0, 1.0, 1.5),
		);
		render_asymptote(file_name*".asy")
	end
end

# ╔═╡ 5a69cd05-f687-4201-a861-43df6fc96a29
render && begin
	imgnames = filter(
		x->occursin(file_name_prefix,x)&&occursin(".png",x),
		readdir(temp_folder)
	) # Populate list of all .pngs
	intstrings = map(x->split(split(x,".")[1],"-")[end], imgnames) # Extract index from filenames
	p = sortperm(parse.(Int, intstrings)) #sort files numerically
	imgnames = imgnames[p]
	
	encoder_options = (crf=23, preset="medium")
	
	firstimg = load(joinpath(temp_folder, imgnames[1]))
	open_video_out(file_name_prefix*"video.mp4", firstimg, framerate=24, encoder_options=encoder_options) do writer
	    @progress "Encoding video frames.." for i in eachindex(imgnames)
	        img = load(joinpath(temp_folder, imgnames[i]))
	        write(writer, img)
	    end
	end
end

# ╔═╡ 1a9409fe-9e0a-4856-9365-b2d444a2fcfb
begin
	file_name_s = file_name_prefix*"-summary"
	println(file_name_s)
	asymptote_export_S2_signals(file_name_s*".asy";
		points= [ [y0,yT] ],
		curves = [solutions...],
		dot_sizes = [3.0,],
		line_width = 1.0,
		colors = Dict{Symbol, Vector{ColorTypes.RGBA{Float64}}}(
			:points => [teal],
			:curves => [curve_colors...],
		),
		camera_position=(2.0, 1.0, 1.5),
	);
	render_asymptote(file_name_s*".asy")
end

# ╔═╡ Cell order:
# ╠═85e76846-912a-11ef-294a-c717389928e4
# ╠═48950604-c2c2-4310-8de5-f89db905668b
# ╠═fe0c1524-b5f4-4afc-aee7-bd2de9d482b6
# ╟─f96fcfed-cfa4-44b1-9034-5c3aedc75aa1
# ╠═bb8b3dfe-cc36-4949-b33a-4c77436a8416
# ╠═6b1bca0b-f209-445e-8f41-b19372c9fffc
# ╠═bd50d1db-7c55-4279-abe0-89f2fe986cc6
# ╠═449e9782-5bed-4642-a068-f8c9106bbe86
# ╠═4ea7c8f7-80f9-482a-b535-229d2671fb4d
# ╠═f08571fe-2e5d-417c-8330-9251233af25d
# ╠═1918a11e-88b9-43b9-b52d-026163580b5f
# ╠═d47accca-ebf3-4085-9d20-2e0ee6e88ef5
# ╠═28143183-eccb-4166-9f79-6baaa8f3f5c2
# ╠═0e5275a1-52d0-45b2-8d1c-5940b50afde8
# ╠═8c726811-c87e-4924-9ebe-bc56e26855a9
# ╠═7f9448c8-0f0c-40de-861f-16427fd335ad
# ╠═a789119e-04b9-4456-acba-ec8e8702c231
# ╠═a5811f04-59f8-4ea2-8616-efad7872830a
# ╠═709daed1-7ef3-4fa5-b390-e18d27a7cefd
# ╠═5c13ce80-209c-4901-913a-3283339a11cc
# ╠═86a649c8-43f8-43a0-a652-24735d28c05e
# ╠═b2d6b477-28d1-421c-958e-ebcfe845a6bb
# ╠═fbf47501-4a41-4f89-940c-9c68aca26b4b
# ╠═25cdf23f-c2ff-44be-9a34-bb565c36775e
# ╠═7a56880f-7ecc-4cc7-a407-97d0f842de3e
# ╠═dcf6174d-0b6f-48da-b0d8-f057b2992e0b
# ╠═48e0d10c-23c4-48c4-91fa-b2a5ae443005
# ╠═88c316c6-4463-4e42-abc8-d83fc117e56f
# ╠═76ecbb61-431d-45b6-a114-134d20d71a55
# ╠═812157bc-6de5-4817-a770-9d86cce8d59b
# ╠═5a69cd05-f687-4201-a861-43df6fc96a29
# ╠═1a9409fe-9e0a-4856-9365-b2d444a2fcfb
