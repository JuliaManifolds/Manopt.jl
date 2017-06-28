#
#
# CPPAlgorithm based minimizations
#
#
export TV_Regularization_CPPA

"""
    TV_Regularization_CPPA(f,α, λ) - compute the TV regularization model of
 given data array f and paramater α and internal operator start λ.

 # Arguments
 * `f` an d-dimensional array of `ManifoldPoint`s
 * `α` parameter of the model (may be given as a vector to provide different
      weights to different directions)
 * `λ` internal parameter of the cyclic proximal point algorithm
 # Output
 * `x` the regulraized array
 # Optional Parameters
 * `MinimalChange` (`10.0^(-5)`) minimal change for the algorithm to stop
 * `MaxIterations` (`500`) maximal number of iterations
 * `FixedMask` : (`[]`) a binary mask of `size(f)` to fix certain input data, e.g. when
 impainting you might set the known ones to be FixedMask. The standard, an empty
 array sets none to be FixedMask.
 * `UnknownMask` : a binary mask indicating UnknownMask pixels that are inpainted

 This implementation is based on the article
> A. Weinmann, L. Demaret, M. Storath:
> Total Variation Regularization for Manifold-valued Data,
> SIAM J. Imaging Science, Vol. 7,
>

 ~ ManifoldValuedImageProcessing.jl ~ R. Bergmann ~ 2016-11-25
"""
function TV_Regularization_CPPA{T <: ManifoldPoint, N}(
      f::Array{T,N}, α::Array{Float64,1}, λ::Number; kwargs...)::Array{T,N}
      kwargs_dict = Dict(kwargs);
      # load optionals
      MinimalChange::Float64=get(kwargs_dict, "MinimalChange", 10.0^-9)
      MaxIterations::Int64=get(kwargs_dict, "MaxIterations", 400)
      FixedMask::Array{Bool,N} = get(kwargs_dict, "FixedMask", falses(f))
      UnknownMask::Array{Bool,N} = get(kwargs_dict, "UnknownMask", falses(f))
  if length(α) == 1
    αV::Array{Float64,1} = α*ones(length(size(f)))
  else
    if length(α) ≠ length(size(f))
      sig1 = length(α);
      throw( ErrorException(string(" Length of α vector (",length(α),
      ") as to be the same as the number of dimensions of f (",length(size(f)),").")) )
    end
    αV = α;
  end
#  @code_warntype _TVRegCPPA(f,αV,λ,MinimalChange,7,FixedMask,UnknownMask)
  @code_warntype _TV1CPPA(f,αV,λ,MinimalChange,MaxIterations,FixedMask,UnknownMask)
  return _TV1CPPA(f,αV,λ,MinimalChange,MaxIterations,FixedMask,UnknownMask)
end
function _TV1CPPA{T <: ManifoldPoint,N}(
      f::Array{T,N}, α::Array{Float64,1}, λ::Float64,
      MinimalChange::Float64,MaxIterations::Int64,FixedMask::Array{Bool,N},UnknownMask::Array{Bool,N}
    )::Array{T,N}
  iter::Int64 = 1
  x::Array{T,N} = f
  xold::Array{T,N} = x
  stillUnknownMask::BitArray{N} = copy(UnknownMask);
  R = CartesianRange(size(f))
  while (  ( (1.0/length(f)*sum( [ distance(ξ,xoldi) for (ξ,xoldi) in zip(x[~stillUnknownMask],xold[~stillUnknownMask]) ] ) > MinimalChange)
            && (iter < MaxIterations) ) || (iter==1)  )
    iter = iter+1
    xold = x;
    x = similar(xold)
    # First term: d(f,x)^2
    @fastmath @inbounds for k in eachindex(x)
      x[k] = proxDistanceSquared(f[k],λ/k,xold[k])
    end
    @fastmath @inbounds for d = 1:N
      unitOffset::CartesianIndex{N} = CartesianIndex{N}(ntuple(i::Integer->i::Integer==d ? 1: 0, N))
      for i in R
        i2 = i+unitOffset
        if (i2 in R)
          if stillUnknownMask[i]
            x[i] = x[i2]; stillUnknownMask[i] = false;
          elseif stillUnknownMask[i2]
            x[i2] = x[i]; stillUnknownMask[i] = false;
          else # both known
            (a,b) = proxTV((x[i], x[i2]),α[d]*λ/iter)
            if ~FixedMask[i]
              x[i] = a
            end
            if ~FixedMask[i2]
              x[i2] = b
            end
          end #endif known
        end #endif in Range
      end #endfor i
    end #endfor d
  end #endwhile
  return x
end
