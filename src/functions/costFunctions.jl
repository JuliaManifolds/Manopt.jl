export costL2TV, costTV

costL2TV(M::Power,f::PowPoint,x::PowPoint,α::Number) = 1/2*distance(M,f,x).^2 + α*costTV(M,x)

function costTV(M::mT,λ::Number,pointTuple::Tuple{P,P},p::Int=1) where {mT <: Manifold, P <: MPoint}
  return distance(M,pointTuple[1],pointTuple[2])^p
end
function costTV(M::Power, λ::Number, x::PowPoint, p::Int=1)
  R = CartesianIndices(M.dims)
  d = length(M.dims)
  maxInd = last(R)
  cost = 0.;
  for k in 1:d # for all directions
    ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
    for i in R # iterate over all pixel
      j = i+ek # compute neighbor
      if all( map(<=, j.I, maxInd.I)) # is this neighbor in range?
        cost += costTV( M.manifold,λ,(x[i],x[j]),p) # Compute TV on these
      end
    end
  end
  return cost
end
