import Base: convert, promote_rule, strip
export MPointE, TVectorE
export getValue, getBasePoint, checkBasePoint
export addNoise, distance, dot, exp, getValue, log, norm
export manifoldDimension, parallelTransport
export randomMPoint, randomTVector
export tangentONB, typicalDistance, zeroTVector
export validateMPoint, validateTVector
export convert, promote_rule, strip
#
#
#
# The extended types for more information/security on base points of tangent vectors
# ---
@doc doc"""
    MPointE <: MPoint

A decorator pattern based extension of MPoint to identify when to switch
to the extended `TVectorE` for functions just working on points, e.g. `log`.
The constructor avoids multiple encapsualtions of extensions.

# Constructors
    MPointE(x [,v=true])

the point can constructed by extending an existing [`MPoint`](@ref).
optionally, the `validation` can be turned off (but is `true`by default).
If `x` is a [`MPointE`](@ref) the default of `v` is taken from `x`.
"""
struct MPointE{P <: MPoint} <: MPoint
  base::P
  validate::Bool
  MPointE{P}(b,v=true) where {P <: MPoint} = new(b,v)
end
MPointE(x::P,v::Bool=true) where {P <: MPoint} = MPointE{P}(x,v)
MPointE(x::MPointE{P},v::Bool=x.validate) where {P <: MPoint} = MPointE{P}(strip(x),v)

@doc doc"""
    TVectorE <: MPoint

A decorator pattern based extension of TVector to additionally store the base
point. The decorator is then used to verify, that exp and dot are only called
with correct base points.

# Constructors
    TVectorE(ξ,x [,v=true])

constructs an extended tangential vector based on the [`TVector`](@ref) `ξ`
with base [`MPoint`](@ref) `x` with optional `validation v`. If none of the
first two arguments is an extended element, `v` defaults to `true`, otherwise,
the one which is an extended is inherited or the `&&` of both validations.
"""
struct TVectorE{T <: TVector, P <: MPoint} <: TVector
  vector::T
  base::P
  validate::Bool
  TVectorE{T,P}(vec,base,val=true) where {T <: TVector, P <: MPoint} = new(vec,base,val)
end
TVectorE(ξ::T, x::P,v::Bool=true) where {T <: TVector, P <: MPoint} = TVectorE{T,P}(ξ,x,v)
TVectorE(ξ::T, x::P,v::Bool=x.validate) where {T <: TVector, P <: MPointE} = TVectorE(ξ,strip(x),v)
TVectorE(ξ::T, x::P,v::Bool=ξ.validate) where {T <: TVectorE, P <: MPoint} = checkBasePoint(ξ,x) ? TVectorE(strip(ξ),getBasePoint(ξ),v) : 0
TVectorE(ξ::T, x::P,v::Bool=x.validate && ξ.validate ) where {T <: TVectorE, P <: MPointE} = checkBasePoint(ξ,x) ? TVectorE(strip(ξ),getBasePoint(ξ),v) : 0
#
getValue(ξ::T) where {T <: TVectorE}= getValue(strip(ξ))
"""
    getBasePoint(ξ)

returns the base point of an extended tangent vector. To continue promotion of
the extended type, the result is always a [`MPointE`](@ref). To eliminate the
decorator, use [`strip`](@ref).
"""
getBasePoint(ξ::T) where {T <: TVectorE{Tl,Pl} where {Tl <: TVector, Pl <: MPoint }} = MPointE(strip(ξ.base))
"""
    strip(ξ)

returns the internal [`TVector`](@ref) of an extended tangent vector
[`TVectorE`](@ref).
"""
strip(ξ::T) where {T <: TVector} = ξ
strip(ξ::T) where {T <: TVectorE} = ξ.vector
#
# Add promotions and implicit conversions
promote_rule(::Type{MPointE{P}}, ::Type{P}) where {P <: MPoint} = MPointE{P}
convert(::Type{MPointE{P}}, x::P) where {P <: MPoint} = MPointE(x)
convert(::Type{MPointE{Q}}, x::P) where {P <: MPoint,  Q <: MPoint} = throw( ErrorException("Cannot `convert` an $(P) to MPointE{$(Q)} since the base type is not of type $(P)."))
convert(::Type{MPointE{P}}, x::MPointE{P}) where {P <: MPoint} = x
convert(::Type{P}, x::MPointE{P}) where {P <: MPoint} = strip(x)
convert(::Type{P}, x::MPointE{Q}) where {P <: MPoint, Q <: MPoint} = throw( ErrorException( "Cannot `convert` an MPointE{$(Q)} to $(P) since the base type is not of type $(P)."))
convert(::Type{P}, x::P) where {P <: MPoint} = x
#
convert(::Type{TVectorE{T,P}}, ξ::T) where {P <: MPoint, T <: TVector} =throw( ErrorException("Cannot `convert` an $(T) to TVectorE{$(T)}; no base MPoint known."))
convert(::Type{T}, ξ::TVectorE{T,P}) where {P <: MPoint, T <: TVector} = ξ.vector
convert(::Type{T}, ξ::TVectorE{S,P}) where {P <: MPoint, T <: TVector, S <: TVector} = throw( ErrorException("Cannot `convert` from TVectorE{$S,$P} to a $T since the TVector types don't match."))
convert(::Type{TVectorE{T,P}}, ξ::TVectorE{T,P}) where {P <: MPoint, T <: TVector} = ξ
convert(::Type{T}, ξ::T) where {T <: TVector} = ξ

getValue(x::P) where {P <: MPointE} = getValue( strip(x) );
show(io::IO, x::MPointE) = print(io, "$(strip(x))E")
"""
    strip(x)

returns the [`MPoint`](@ref) the [`MPointE`](@ref) `x` stores internally.
If applied to an already non-extended [`MPoint`](@ref), nothing happens.
"""
strip(x::P) where {P <: MPointE} = x.base
strip(x::P) where {P <: MPoint} = x

show(io::IO, ξ::TVectorE) = print(io, "$(ξ.vector)E_$(ξ.base)")
function +(ξ::T,ν::T) where {T <: TVectorE}
  checkBasePoint(ξ,ν)
  return T(strip(ξ)+ν.vector,getBasePoint(ξ))
end
function -(ξ::T,ν::T) where {T <: TVectorE}
  checkBasePoint(ξ,ν)
  return T(strip(ξ)-ν.vector,getBasePoint(ξ))
end
"""
    checkBasePoint(ξ,ν)

checks, whether the base of two tangent vectors is identical, if both tangent
vectors are of type `TVectorE`. If one of them is not an extended vector, the
function returns true, expecting the tangent vector implicitly to be correct.
"""
function checkBasePoint(ξ::T,ν::T) where {T <: TVectorE}
  if getValue( getBasePoint(ξ) ) != getValue( getBasePoint(ν) )
    throw(
    DomainError("The two tangent vectors $ξ and $ν do not have the same base.")
    )
  else
    return true
  end
end
checkBasePoint(ξ::T,ν::S) where {T <: TVectorE, S <: TVector} = true
checkBasePoint(ξ::S,ν::T) where {T <: TVectorE, S <: TVector} = true
"""
    checkBasePoint(ξ,x)

checks, whether the base of the tangent vector `ξ` is `x`. If `ξ` is not an
extended tangent vector `TVectorE` the function returns true, assuming the base
implicitly to be correct
"""
function checkBasePoint(ξ::T,x::P) where {T <: TVectorE, P <: MPoint}
  if getValue( getBasePoint(ξ) ) != getValue(x)
    throw(
    DomainError("The tangent vector $ξ is not from the tangent space of $x")
    )
  else
    return true
  end
end
checkBasePoint(ξ::T,x::P) where {T<: TVector, P<: MPoint} = true
#
# Defaults for validation that do warnings if used but return true
#
# unary operators
*(ξ::T,s::Number) where {T <: TVectorE} = T(s*strip(ξ),getBasePoint(ξ))
*(s::Number, ξ::T) where {T <: TVectorE} = T(s*strip(ξ),getBasePoint(ξ))
/(ξ::T,s::Number) where {T <: TVectorE} = T(strip(ξ)/s,getBasePoint(ξ))
-(ξ::T) where {T <: TVectorE} = T(-strip(ξ),getBasePoint(ξ))
+(ξ::T) where {T <: TVectorE} = T(strip(ξ),getBasePoint(ξ))
# compare extended vectors
==(ξ::T,ν::T) where {T <: TVectorE} = ( checkBasePoint(ξ,ν) && all(strip(ξ)==ν.vector) )
#
# encapsulate default functions
#
distance(M::mT, x::P, y::P) where {mT <: Manifold, P <: MPointE} = _distance(M,strip(x),strip(y), x.validate&&y.validate)
distance(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPointE, Q <: MPoint} = _distance(M,strip(x),strip(y), x.validate)
distance(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPoint, Q <: MPointE} = _distance(M,strip(x),strip(y), y.validate)
function _distance(M,x,y,v)
  if v
    validateMPoint(M,x)
    validateMPoint(M,y)
  end
  return distance(M,x,y)
end

function dot(M::mT, x::P, ξ::T, ν::T)::Float64 where {mT<:Manifold, P <: MPointE, T<:TVectorE}
  checkBasePoint(ξ,x)
  checkBasePoint(ξ,ν)
  return _dot(M, strip(x), strip(ξ), strip(ν), x.validate && ξ.validate && ν.validate )
end
function dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPointE, T <: TVectorE, S <: TVector}
  checkBasePoint(ξ,x)
  return _dot(M, strip(x),strip(ξ),ν, x.validate && ξ.validate)
end
function dot(M::mT, x::P, ξ::S, ν::T) where {mT <: Manifold, P <: MPointE, T <: TVectorE, S <: TVector}
  checkBasePoint(ν,x)
  return _dot(M, strip(x), ξ, strip(ν), x.validate && ν.validate )
end
dot(M::mT, x::P, ξ::T, ν::T) where {mT <: Manifold, P <: MPointE, T <: TVector} = _dot(M,strip(x),ξ,ν, x.validate)

function dot(M::mT, x::P, ξ::T, ν::T) where {mT <: Manifold, P <: MPoint, T <: TVectorE}
  checkBasePoint(ξ,ν)
  return _dot(M,x,strip(ξ),strip(ν), ξ.validate && ν.validate )
end
function dot(M::mT, x::P, ξ::S, ν::T) where {mT <: Manifold, P <: MPoint, T <: TVectorE, S <: TVector}
  checkBasePoint(ν,x)
  return _dot(M,x,ξ,strip(ν), ν.validate )
end
function dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPoint, T <: TVectorE, S <: TVector}
  checkBasePoint(ξ,x)
  return _dot(M,x,strip(ξ),ν, ξ.validate )
end
function _dot(M,x,ξ,ν,v)
  if v
    validateMPoint(M,x)
    validateTVector(M,x,ξ)
    validateTVector(M,x,ν)
  end
  return dot(M,x,ξ,ν)
end

function exp(M::mT,x::P,ξ::T,t::Float64=1.0) where {mT<:Manifold, P<:MPointE, T<:TVectorE}
  checkBasePoint(ξ,x)
  return _exp(M, strip(x), t*strip(ξ), x.validate && ξ.validate )
end
exp(M::mT,x::P,ξ::T,t::Float64=1.0) where {mT<:Manifold, P<:MPointE, T<:TVector} = _exp(M,strip(x), t*ξ
, x.validate )
function exp(M::mT,x::P,ξ::T,t::Float64=1.0) where {mT<:Manifold, P<:MPoint, T<:TVectorE}
  checkBasePoint(ξ,x)
  return _exp(M, x, t*strip(ξ), ξ.validate )
end
function _exp(M,x,ξ,v)
  if v
    validateMPoint(M,x)
    validateTVector(M,x,ξ)
  end
  y = exp(M,x,ξ)
  if v
    validateMPoint(M,y)
  end
  return MPointE(y, v)
end

log(M::mT,x::Q,y::Q) where {mT<:Manifold, Q<:MPointE} = _log(M,strip(x),strip(y), x.validate && y.validate )
log(M::mT,x::P,y::Q) where {mT<:Manifold, P<:MPoint, Q<:MPointE} = _log(M, x, strip(y), y.validate )
log(M::mT,x::Q,y::P) where {mT<:Manifold, P<:MPoint, Q<:MPointE} = _log(M, strip(x), y, x.validate )
function _log(M,x,y,v)
  if v
    validateMPoint(M,x)
    validateMPoint(M,y)
  end
  ν = log(M,x,y)
  if v
    validateTVector(M,x,ν)
  end
  return TVectorE(ν,x, v)
end

manifoldDimension(x::P) where {P <: MPointE} = manifoldDimension(strip(x))

function norm(M::mT, x::P, ξ::T)::Float64 where {mT<:Manifold, P <: MPointE, T<:TVectorE}
  checkBasePoint(ξ,x)
  return _norm(M,strip(x),strip(ξ), x.validate && ξ.validate )
end
norm(M::mT,x::P,ξ::T) where {mT<:Manifold, P <: MPointE, T<:TVector} = _norm(M,strip(x),ξ, x.validate)
function norm(M::mT,x::P,ξ::T) where {mT<:Manifold, P <: MPoint, T<:TVectorE}
  checkBasePoint(ξ,x)
  return _norm(M,x,strip(ξ), ξ.validate)
end
function _norm(M,x,ξ,v)
  if v
    validateMPoint(M,x)
    validateTVector(M,x,ξ)
  end
  norm(M,x,ξ)
end

function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPointE, Q <: MPoint, T<: TVectorE}
  checkBasePoint(ξ,x)
  return _parallelTransport(M,strip(x),y,strip(ξ), x.validate && ξ.validate)
end
function parallelTransport(M::mT,x::P,y::P,ξ::T) where {mT <: Manifold, P <: MPointE, T<: TVectorE}
  checkBasePoint(ξ,x)
  return _parallelTransport(M,strip(x),strip(y),strip(ξ), x.validate && y.validate && ξ.validate)
end
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPoint, Q <: MPointE, T<: TVectorE}
  checkBasePoint(ξ,x)
  _parallelTransport( M, x, strip(y), strip(ξ), y.validate && ξ.validate )
end
parallelTransport(M::mT,x::P,y::P,ξ::T) where {mT <: Manifold, P <: MPointE, T<: TVector} = _parallelTransport(M, strip(x), strip(y), ξ, x.validate && y.validate )
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPoint, Q <: MPointE, T<: TVector}
  checkBasePoint(ξ,x)
  return _parallelTransport(M,x,strip(y),ξ, y.validate )
end
parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPointE, Q <: MPoint, T<: TVector} = _parallelTransport(M,strip(x),y,ξ, x.validate )
function _parallelTransport(M,x,y,ξ,v)
  if v
    validateMPoint(M,x)
    validateMPoint(M,y)
    validateTVector(M,x,ξ)
  end
  ν = parallelTransport(M,x,y,ξ)
  if v
    validateTVector(M,y,ν)
  end
  return TVectorE(ν,y,v)
end

function randomTVector(M::mT, x::P,options...) where {mT <: Manifold, P <: MPointE}
  ξ = randomTVector(M,strip(x),options...)
  if x.validate
    validateTVector(M,strip(x),ξ)
  end
  return TVectorE(ξ,x,x.validate)
end

tangentONB(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPointE, Q <: MPoint} = _tangentONB(M, strip(x), y, x.validate )
tangentONB(M::mT, x::Q, y::P) where {mT <: Manifold, P <: MPointE, Q <: MPoint} = _tangentONB(M, x, strip(y), y.validate )
tangentONB(M::mT, x::P, y::P) where {mT <: Manifold, P <: MPointE} = _tangentONB(M, strip(x), strip(y), x.validate && y.validate )

function _tangentONB(M,x::MPoint,y::MPoint, v::Bool)
  if v
    validateMPoint(M,x)
    validateMPoint(M,y)
  end
  V,κ = tangentONB(M,x,log(M,x,y))
  if v
    validateTVector.(Ref(M), Ref(x), V)
  end
  return TVectorE.(V, Ref(x), Ref(v)), κ
end
tangentONB(M::mT, x::P, ξ::T) where {mT <: Manifold, P <: MPointE, T <: TVectorE} = _tangentONB(M, strip(x), strip(ξ), x.validate && ξ.validate )
tangentONB(M::mT, x::P, ξ::T) where {mT <: Manifold, P <: MPoint, T <: TVectorE} = _tangentONB(M, x, strip(ξ), ξ.validate )
tangentONB(M::mT, x::P, ξ::T) where {mT <: Manifold, P <: MPointE, T <: TVector} = _tangentONB(M, strip(x), ξ, x.validate )
function _tangentONB(M, x::MPoint, ξ::TVector, v::Bool)
  if v
    validateMPoint(M,x)
    validateTVector(M,x,ξ)
  end
  V,κ = tangentONB(M,x,ξ)
  if v
    validateTVector.(Ref(M), Ref(x), V)
  end
  return TVectorE.(V, Ref(x), Ref(v)), κ
end

validateMPoint(M::mT,x::P) where {mT <: Manifold, P <: MPointE} = validateMPoint(M,strip(x))
validateTVector(M::mT,x::P,ξ::T) where {mT <: Manifold, P <: MPointE, T <: TVectorE} = validateTVector(M, strip(x), strip(ξ))
validateTVector(M::mT,x::P,ξ::T) where {mT <: Manifold, P <: MPoint, T <: TVectorE} = validateTVector(M, x, strip(ξ))
validateTVector(M::mT,x::P,ξ::T) where {mT <: Manifold, P <: MPointE, T <: TVector} = validateTVector(M, strip(x), ξ)
function zeroTVector(M::mT,x::P) where {mT <: Manifold, P <: MPointE}
  if x.validate
    validateMPoint(M,x)
  end
  return TVectorE( zeroTVector(M,strip(x)), strip(x), x.validate )
end