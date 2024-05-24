### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ a8b5e20a-c347-4908-ae99-188e2e871b25
using Pkg; Pkg.activate()

# ╔═╡ d9d975b7-6fdd-4310-b596-5094d7ab8846
using Manopt, Manifolds, ManoptExamples, ManifoldDiff, Images, CSV, DataFrames, LinearAlgebra, JLD2, Dates, NamedColors, Plots

# ╔═╡ 6f918d7c-c756-11ee-0a18-1ff4b597f1ac
md"""
## Accelerate L2TV for manifold-valued data
"""

# ╔═╡ 73dfbf50-071f-403f-b508-f769ce5a6d2e
begin
paul_tol = load_paul_tol()
indigo = paul_tol["mutedindigo"]
green = paul_tol["mutedgreen"]
sand = paul_tol["mutedsand"]
olive = paul_tol["mutedolive"]
teal = paul_tol["mutedteal"]
wine = paul_tol["mutedwine"]
grey = paul_tol["mutedgrey"]
end;

# ╔═╡ 27195162-542f-4e49-b6d6-84cb9865ca80
begin # Settings and Data
	experiment_name = "SPD_Image_DR"
	export_orig = true
	export_result = true
	export_table = true
	asy_render_detail = 2
	results_folder = joinpath(@__DIR__)

	λ = 0.58 # prox parameter
	α = 0.99 # relaxation
	w = 6.0 # TV weight

	f = ManoptExamples.artificial_SPD_image2(32)
	M = SymmetricPositiveDefinite(3)
	N = PowerManifold(M, NestedPowerRepresentation(), 32,32)
end;

# ╔═╡ dab95acd-efcc-4f9c-aa05-a02e21d035fb
begin # Cost & Proxes
d = length(size(f))
rep(d) = (d > 1) ? [ones(Int, d)..., d] : d
fidelity(M, p) = 1 / 2 * distance(M, p, f)^2
Λ(M, p) = ManoptExamples.forward_logs(M, p) # on T_xN
prior(M, p) = norm(norm.(Ref(M.manifold), repeat(p, rep(d)...), Λ(M, p)), 1)

N2 = PowerManifold(N, NestedPowerRepresentation(), 5)
cost(N2, p) = fidelity(N2.manifold, p[1]) + w * prior(N2.manifold, p[1])
prox1 = (N2, λ, p) -> [
	ManifoldDiff.prox_distance(N2.manifold, λ, f, p[1]),
	ManoptExamples.prox_parallel_TV(N2.manifold, w * λ, p[2:5])...
]
prox2 = (N2, λ, p) -> fill(mean(N2.manifold, p, GradientDescentEstimation(); stop_iter=4), 5)
p0 = fill(f, 5)
end;

# ╔═╡ 25edd57a-2c23-46fc-a1ee-fd199cafd36c
begin # mutating versions
	function prox1!(N2, q, λ, p)
		ManifoldDiff.prox_distance!(N2.manifold, q[1], λ, f, p[1])
		ManoptExamples.prox_parallel_TV!(N2.manifold, q[2:5], w * λ, p[2:5])
		return q
	end
	function prox2!(N2, q, λ, p)
		q_ = copy(N2.manifold, q[1])
		mean!(N2.manifold, q_, p, GradientDescentEstimation())
		for i = 1:length(q)
			copyto!(N2.manifold, q[i], q_)
		end
	end
end

# ╔═╡ f882b4c6-b124-4a5e-9e6c-5857a9e17045
sc = StopWhenChangeLess(1e-6) | StopAfterIteration(1500);

# ╔═╡ b166715e-00ee-4661-8ff4-80e3d281b9ec
begin # Fix prox! for now
	import ManoptExamples: prox_parallel_TV!
function prox_parallel_TV!(
    M::PowerManifold, y::AbstractVector, λ, x::AbstractVector, p::Int=1
)
    R = CartesianIndices(x[1])
    d = ndims(x[1])
    if length(x) != 2 * d
        throw(
            ErrorException(
                "The number of inputs from the array ($(length(x))) has to be twice the data dimensions ($(d)).",
            ),
        )
    end
    maxInd = Tuple(last(R))
    # init y
    for i in 1:length(x)
        copyto!(M.manifold, y[i], x[i])
    end
    yV = reshape(y, d, 2)
    xV = reshape(x, d, 2)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1 # even odd
            for i in R # iterate over all pixel
                if (i[k] % 2) == l
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neighbour index as Cartesian Index
                        # parallel means we apply each (direction even/odd) to a separate copy of the data.
                        ManoptExamples.prox_Total_Variation!(
                            M.manifold,
                            [yV[k, l + 1][i], yV[k, l + 1][j]],
                            λ,
                            (xV[k, l + 1][i], xV[k, l + 1][j]),
                            p,
                        ) # Compute TV on these in place of y
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end
end

# ╔═╡ 0fa3ccff-d866-4922-a5b0-18fa55676536
begin
	Nt = PowerManifold(M, NestedPowerRepresentation(), 2,2)
	N2t = PowerManifold(Nt, 5)
	xT = f[1:2,1:2]
	yT = [deepcopy(xT), deepcopy(xT),deepcopy(xT),deepcopy(xT),deepcopy(xT)]
	zT = deepcopy(yT)
	zT2 = ManoptExamples.prox_parallel_TV(N2t, w * λ, yT[2:5])
	prox_parallel_TV!(N2t, zT[2:5], w * λ, yT[2:5])
end

# ╔═╡ a604aaa8-b929-4e50-a0e0-4b90dd97a7f2
@time s1 = DouglasRachford(
    N2,
    cost,
    [prox1, prox2],
    p0;
    λ=i -> λ,
    α=i -> α,
	debug =  [:Iteration, " | ", :Cost, :Change, "\n", 100, :Stop],
    record = [:Iteration, :Cost],
    stopping_criterion=sc,
    return_state=true,
)

# ╔═╡ c686565c-4639-4fb8-b80c-2e51fb17e9f8
q1 = get_solver_result(s1)[1];

# ╔═╡ 1f2366dd-1965-4138-bbb3-1da7fed6547d
r1 = get_record(s1)

# ╔═╡ 730b20f7-5f44-4ed7-91a2-737aa7d4647a
@time s2 = DouglasRachford(
    N2,
    cost,
    [prox1!, prox2!],
    p0;
    λ=i -> λ,
    α=i -> α,
	debug =  [:Iteration, " | ", :Cost, :Change, "\n", 100, :Stop],
	evaluation= InplaceEvaluation(),
    record = [:Iteration, :Cost],
    return_state=true,
    stopping_criterion=sc,
)

# ╔═╡ 979746e1-9e60-4a19-bd99-c315c2726221
q2 = get_solver_result(s2)[1];

# ╔═╡ c2ff7e0c-d624-4d6b-8e84-ac2f3fbaff20
r2 = get_record(s2)

# ╔═╡ 767a92b1-0a5c-4360-a976-876905710888
# acceleration
@time s3 = DouglasRachford(
    N2,
    cost,
    [prox1, prox2],
    p0;
	n=2,
    λ=i -> λ,
    α=i -> α,
	debug =  [:Iteration, " | ", :Cost, :Change, "\n", 100, :Stop],
    record = [:Iteration, :Cost],
    stopping_criterion=sc,
    return_state=true,
)

# ╔═╡ 3217cd53-fddb-4fd4-bf7f-1a0586812de5
q3 = get_solver_result(s3)[1];

# ╔═╡ 96045d19-96ad-4ec6-a292-9bd908c53b7e
r3 = get_record(s3)

# ╔═╡ cee27e3d-0e12-4cd5-967e-1997e8bac89a
# ineratia
@time s4 = DouglasRachford(
    N2,
    cost,
    [prox1, prox2],
    p0;
	θ=i -> 0.0,
    λ=i -> λ,
    α=i -> α,
	debug =  [:Iteration, " | ", :Cost, :Change, "\n", 100, :Stop],
    record = [:Iteration, :Cost],
    stopping_criterion=sc,
    return_state=true,
)

# ╔═╡ cbf627bc-7433-4ae8-bd32-33a8492eb16f
q4 = get_solver_result(s4)[1];

# ╔═╡ 8ae6501b-960f-4620-b63c-7299369259a6
r4 = get_record(s4)

# ╔═╡ 23102383-7643-4113-af35-d3a08d868d6c
iterates = [ [e[1] for e in get_record(s, :Iteration)] for s in [s1, s2, s3, s4]];

# ╔═╡ 2b24563f-d1b7-4908-b06d-de836a5126c4
costs = [ [e[2] for e in get_record(s, :Iteration)] for s in [s1, s2, s3, s4]];

# ╔═╡ 2276a077-0e85-4b03-afcc-a8b185803c62
begin
    fig = plot(
	    xlabel=raw"Iterations $k$", ylabel=raw"Cost $f(x)$ (log. scale)",
        yaxis=:log,
		#ylim = (1e-26,1e6),
		#xlim = (0,34),
    );
	plot!(fig, iterates[1], costs[1], color=indigo, label="Douglas-Rachford");
	#plot!(fig, iterates[2], costs[2], color=green, label=raw"DR with inplace test");
	plot!(fig, iterates[3], costs[3], color=sand, label=raw"DR with 3-acceleration");
	plot!(fig, iterates[4], costs[4], color=teal, label=raw"DR with intertia");
	fig
end

# ╔═╡ Cell order:
# ╟─6f918d7c-c756-11ee-0a18-1ff4b597f1ac
# ╠═a8b5e20a-c347-4908-ae99-188e2e871b25
# ╠═d9d975b7-6fdd-4310-b596-5094d7ab8846
# ╠═73dfbf50-071f-403f-b508-f769ce5a6d2e
# ╠═27195162-542f-4e49-b6d6-84cb9865ca80
# ╠═dab95acd-efcc-4f9c-aa05-a02e21d035fb
# ╠═25edd57a-2c23-46fc-a1ee-fd199cafd36c
# ╠═f882b4c6-b124-4a5e-9e6c-5857a9e17045
# ╠═b166715e-00ee-4661-8ff4-80e3d281b9ec
# ╠═0fa3ccff-d866-4922-a5b0-18fa55676536
# ╠═a604aaa8-b929-4e50-a0e0-4b90dd97a7f2
# ╠═c686565c-4639-4fb8-b80c-2e51fb17e9f8
# ╠═1f2366dd-1965-4138-bbb3-1da7fed6547d
# ╠═730b20f7-5f44-4ed7-91a2-737aa7d4647a
# ╠═979746e1-9e60-4a19-bd99-c315c2726221
# ╠═c2ff7e0c-d624-4d6b-8e84-ac2f3fbaff20
# ╠═767a92b1-0a5c-4360-a976-876905710888
# ╠═3217cd53-fddb-4fd4-bf7f-1a0586812de5
# ╠═96045d19-96ad-4ec6-a292-9bd908c53b7e
# ╠═cee27e3d-0e12-4cd5-967e-1997e8bac89a
# ╠═cbf627bc-7433-4ae8-bd32-33a8492eb16f
# ╠═8ae6501b-960f-4620-b63c-7299369259a6
# ╠═23102383-7643-4113-af35-d3a08d868d6c
# ╠═2b24563f-d1b7-4908-b06d-de836a5126c4
# ╠═2276a077-0e85-4b03-afcc-a8b185803c62
