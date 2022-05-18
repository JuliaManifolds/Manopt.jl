#
# Prepare cost, proximal maps and differentials
M = PowerManifold(pixelM, NestedPowerRepresentation(), size(f)...)
N = base_manifold(TangentBundle(M)) # TODO does this actually work in nD?
fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
Λ(M, x) = forward_logs(M, x)
# Λ(M, x) = ProductRepr(x, forward_logs(M, x))
prior(M, x) = norm(norm.(Ref(pixelM), x, Λ(M, x)), 1) # TODO does this actually work in 2D?
# prior(M, x) = norm(norm.(Ref(pixelM), x, submanifold_component(N, Λ(M, x), 2)), 1)
cost(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)

prox_F(M, λ, x) = prox_distance(M, λ / α, f, x, 2)
prox_G_dual(N, n, λ, ξ) = project_collaborative_TV(N, λ, n, ξ, Inf, Inf, 1.0) # non-isotropic

# TODO Dprox_F
Dprox_F(M,λ,x,η) = differential_geodesic_startpoint(M,x,f,λ/(α+λ),η)

function differential_project_collaborative_TV(N::PowerManifold, λ, x, Ξ, Η, p=2.0, q=1.0, α=1.0)
	pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        iRep = (1,)
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($(d)) has to be equal to the number of the previous ones ($(s)) but its not.",
                ),
            )
        end
        iRep = (Integer.(ones(d))..., d)
    end
    if q == Inf
        if p == Inf
            norms = norm.(Ref(N.manifold), x, Ξ)
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
        return convert(typeof(norms),(norms .<= 1)) .* Η + convert(typeof(norms),(norms .<= 1)) .* (Η - 1 ./(norms .+ 1e-16) .* inner.(Ref(N.manifold), n, Ξ, Η) .* Η)
    end # end q
    throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end

# TODO Dprox_G_dual
Dprox_G_dual(N, n, λ, ξ, η) = differential_project_collaborative_TV(N, λ, n, ξ, η, Inf, Inf, 1.0)
# Dprox_G_dual(N, n, λ, ξ, η; γ=0,isotropic=false) = differential_project_collaborative_TV(N, λ, n, ξ, Inf, Inf, 1.0)

# function
	# M2 = base_manifold(N)
    # m2 = n
    # power_size = power_dimensions(M2)
    # R = CartesianIndices(Tuple(power_size))
    # d = length(power_size)
    # maxInd = last(R).I
	#
    # J = zero_vector(N, n)
	#
	# if !isotropic || d==1
	# 	for j in R
	# 		for k in 1:d
	# 			mⱼ = m2[M2, j, k]
	# 			ηⱼ = η[N, j, k]
	# 			g = norm(M2.manifold,mⱼ,ηⱼ)
	# 			if !(j[k]==maxInd[k])
	# 				ξⱼ = ξ[N, j, k]
	# 				if g/(1+λ*γ) <=1
	# 					J[N, j..., k] +=  ξⱼ/(1+λ*γ)
	# 				else
	# 					J[N, j..., k] +=  1/g * (ξⱼ - 1/g^2 * inner(M2.manifold,mⱼ,ξⱼ,ηⱼ)*ηⱼ)
	# 				end
	# 			else
	# 				# Taking care of the boundary equations
	# 				# J[N, j... ,k] = zero_vector(M2.manifold,mⱼ)
	# 			end
	# 		end
	# 	end
	# else
	# 	for j in R
	# 		g = norm(M2.manifold, m2[M2, j..., 1], η[N, j..., :])
	# 		for k in 1:d
	# 			mⱼ = m2[M2, j..., k]
	# 			ηⱼ = η[N, j..., k]
	# 			if !(j[k]==maxInd[k])
	#
	# 				ξⱼ = ξ[N, j..., k]
	# 				if g/(1+λ* γ) <=1
	# 					J[N, j..., k] +=  ξⱼ/(1+λ* γ)
	# 				else
	# 					for κ in 1:d
	# 						if κ != k
	# 							J[N, j..., κ] += - 1/g^3 * inner(M2.manifold,mⱼ,ξⱼ,ηⱼ)*η[j,κ]
	# 						else
	# 							J[N, j..., k] += 1/g * (ξⱼ - 1/g^2 * inner(M2.manifold,mⱼ,ξⱼ,ηⱼ)*ηⱼ)
	# 						end
	# 					end
	# 				end
	# 			else
	# 				# Taking care of the boundary equations
	# 				# J[N, j... ,k] = zero_vector(M2.manifold,mⱼ)
	# 			end
	# 		end
	# 	end
	# end
	#
    # return J
# end

DΛ(M, m, X) = differential_forward_logs(M, m, X)
# DΛ(M, m, X) = ProductRepr(zero_vector(M, m), differential_forward_logs(M, m, X))
adjoint_DΛ(N, m, n, ξ) = adjoint_differential_forward_logs(N, m, ξ)
# adjoint_DΛ(N, m, n, ξ) = adjoint_differential_forward_logs(N.manifold, m, ξ[N, :vector])
