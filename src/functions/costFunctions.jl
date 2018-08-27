export costL2TV, costTV

costL2TV(M::Power,f::PowPoint,x::PowPoint,α::Number) =
  1/2*distance(M,f,x)^2 + α*costTV(M,x)

costL2TV1plusTV2(M::Power,f::PowPoint,x::PowPoint,α::Number,β::Number) =
  1/2*distance(M,f,x)^2 + α*costTV(M,x) + β*costTV2(M,x)

costL2TV2(M::Power,f::PowPoint,x::PowPoint,β::Number) =
    1/2*distance(M,f,x)^2 + β*costTV2(M,x)

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
function costTV2(M::mT,λ::Number,pointTuple::Tuple{P,P,P},p::Int=1) where {mT <: Manifold, P <: MPoint}
  # TODO: sometimes necessary: mid point nearest to [2]
  return distance(M,midPoint(pointTuple[1],pointTuple[3]),pointTuple[2])^p
end
function costTV2(M::Power, λ::Number, x::PowPoint, p::Int=1)
  R = CartesianIndices(M.dims)
  d = length(M.dims)
  minInd, maxInd = first(R), last(R)
  cost = 0.
  for k in 1:d # for all directions
    ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
    for i in R # iterate over all pixel
      jF = i+ek # compute forward neighbor
      jB = i-ek # compute backward neighbor
      if all( map(<=, jF.I, maxInd.I) ) && all( map(>=, jB.I, minInd.I)) # are neighbors in range?
        cost += costTV2( M.manifold, λ, (y[jB], y[i], y[jF]) ) # Compute TV on these
      end
    end # i in R
  end # directions
  return cost
end
