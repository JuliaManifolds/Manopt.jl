import Base: convert, promote_rule
export getValue, getBase, checkBase
export addNoise, distance, dot, exp, getValue, log, manifoldDimension, norm
export manifoldDimension, parallelTransport, tangentONB, typicalDistance, zeroTVector
export MPointE, TVectorE
export convert, proomte_rule
#
#
#
# The extended types for more information/security on base points of tangent vectors
# ---
"""
A decorator pattern based extension of MPoint to identify when to switch
to the extended `TVectorE` for functions just working on points, e.g. `log`
"""
struct MPointE{P <: MPoint} <: MPoint
    base::P
end
"""
A decorator pattern based extension of TVector to additionally store the base
point. The decorator is then used to verify, that exp and dot are only called
with correct base points.
"""
struct TVectorE{T <: TVector, P <: MPoint} <: TVector
  vector::T
  base::P
end
TVectorE(ξ::T, x::P) where {T <: TVector, P <: MPointE} = TVector(ξ,getBase(x))
TVectorE(ξ::T, x::P) where {T <: TVectorE, P <: MPointE} = checkBase(ξ,x) ? TVector(getVector(ξ),getBase(ξ)) : 0
TVectorE(ξ::T, x::P) where {T <: TVectorE, P <: MPoint} = checkBase(ξ,x) ? TVector(getVector(ξ),getBase(ξ)) : 0
#
convert(::Type{TVectorE}, x::T) where {T <: TVector} = error("Can't convert a non-extended tangent vector to an extended one, because the base is unknown.")
convert(::Type{TVectorE}, x::TVectorE) = x
getValue(ξ::T) where {T <: TVectorE}= getValue(getVector(ξ))
"""
    getBase(ξ)
returns the base point of an extended tangent vector.
"""
getBase(ξ::T) where {T <: TVectorE{Tl,Pl} where {Tl <: TVector, Pl <: MPoint }} = MPointE(ξ.base)
getBase(ξ::T) where {T <: TVectorE{Tl,Pl} where {Tl <: TVector, Pl <: MPointE}}= ξ.base
"""
    getVector(ξ)
returns the internal TVector point of an extended tangent vector.
"""
getVector(ξ::T) where {T <: TVectorE} = ξ.vector
promote_rule(::Type{MPointE}, ::Type{MPoint}) = MPointE
convert(::Type{MPointE{P}}, x::P) where {P <: MPoint} = MPointE(x)
convert(::Type{MPointE{Q}}, x::P) where {P <: MPoint,  Q <: MPoint} = error("Cannot `convert` an $(P) to MPointE{$(Q)} since the base is not of type $(P).")
convert(::Type{MPointE{P}}, x::MPointE{P}) where {P <: MPoint} = x
convert(::Type{P}, x::MPointE{P}) where {P <: MPoint} = getBase(x)
convert(::Type{P}, x::MPointE{Q}) where {P <: MPoint, Q <: MPoint} = error("Cannot `convert` an MPointE{$(Q)} to $(P) since the base is not of type $(P).")
convert(::Type{P}, x::P) where {P <: MPoint} = x
getValue(x::P) where {P <: MPointE} = getValue( getBase(x) );
show(io::IO, x::MPointE) = print(io, "$(getBase(x))E")
"""
    getBase(x)
returns the point this extended manifold point stores internally.
"""
getBase(x::P) where {P <: MPointE} = x.base
getBase(x::P) where {P <: MPoint} = x

show(io::IO, ξ::TVectorE) = print(io, "$( getValue(ξ) )_$( getValue( getBase(ξ) ) )")
function +(ξ::T,ν::T) where {T <: TVectorE}
    checkBase(ξ,ν)
    return T(getVector(ξ)+ν.vector,getBase(ξ))
end
function -(ξ::T,ν::T) where {T <: TVectorE}
    checkBase(ξ,ν)
    return T(getVector(ξ)-ν.vector,getBase(ξ))
end
"""
    checkBase(ξ,ν)
checks, whether the base of two tangent vectors is identical, if both tangent
vectors are of type `TVectorE`. If one of them is not an extended vector, the
function returns true, expecting the tangent vector implicitly to be correct.
"""
function checkBase(ξ::T,ν::T) where {T <: TVectorE}
    if getValue( getBase(ξ) ) != getValue( getBase(ν) )
        throw(
            ErrorException("The two tangent vectors $ξ and $ν do not have the same base.")
        );
    else
        return true;
    end
end
checkBase(ξ::T,ν::S) where {T <: TVectorE, S <: TVector} = true
checkBase(ξ::S,ν::T) where {T <: TVectorE, S <: TVector} = true
"""
    checkBase(ξ,x)
checks, whether the base of the tangent vector `ξ` is `x`. If `ξ` is not an
extended tangent vector `TVectorE` the function returns true, assuming the base
implicitly to be correct
"""
function checkBase(ξ::T,x::P,tAttr=" ") where {T <: TVectorE, P <: MPoint}
    if getValue( getBase(ξ) ) != getValue(x)
        throw(
            ErrorException("The$(tAttr)tangent vector $ξ is not from the tangent space of $x")
        );
    else
        return true;
    end
end
checkBase(ξ::T,x::P) where {T<: TVector, P<: MPoint} = true
# unary operators
*(ξ::T,s::Number) where {T <: TVectorE} = T(s*getVector(ξ),getBase(ξ))
*(s::Number, ξ::T) where {T <: TVectorE} = T(s*getVector(ξ),getBase(ξ))
# /
/(ξ::T,s::Number) where {T <: TVectorE} = T(getVector(ξ)/s,getBase(ξ))
/(s::Number, ξ::T) where {T <: TVectorE} = T(s/getVector(ξ),getBase(ξ))
-(ξ::T) where {T <: TVectorE} = T(-getVector(ξ),getBase(ξ))
+(ξ::T) where {T <: TVectorE} = T(getVector(ξ),getBase(ξ))

# compare extended vectors
==(ξ::T,ν::T) where {T <: TVectorE} = ( checkBase(ξ,ν) && all(getVector(ξ)==ν.vector) )
#
# encapsulate default functions
#
addNoise(M::mT,x::P,σ) where {mT <: Manifold, P <: MPointE} = MPointE( addNoise(M,getBase(x),σ) )
distance(M::mT, x::P, y::P) where {mT <: Manifold, P <: MPointE} = distance(M,getBase(x),getBase(y))
distance(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPointE, Q <: MPoint} = distance(M,getBase(x),getBase(y))
distance(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPoint, Q <: MPointE} = distance(M,getBase(x),getBase(y))

function dot(M::mT, x::P, ξ::T, ν::T)::Float64 where {mT<:Manifold, P <: MPointE, T<:TVectorE}
    checkBase(ξ,x," first ")
    checkBase(ξ,ν," second ")
    return dot(M, getVector(ξ), getVector(ν) );
end
# all others (i.e. one implicitly assumed to be correct -> pass)
# (a) MPointE 1) with different TVectorEs even because one might have an SnPointE one not.
dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPointE, T <: TVectorE, S <: TVectorE} = dot(M,getBase(x),getVector(ξ),getVector(ν))
dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPointE, T <: TVectorE, S <: TVector} = dot(M,x,getVector(ξ),ν)
dot(M::mT, x::P, ξ::S, ν::T) where {mT <: Manifold, P <: MPointE, T <: TVectorE, S <: TVector} = dot(M,x,ξ,getVector(ν) )
dot(M::mT, x::P, ξ::T, ν::T) where {mT <: Manifold, P <: MPointE, T <: TVector} = dot(M,getBase(x),ξ,ν)
# (b) MPoint
dot(M::mT, x::P, ξ::T, ν::T) where {mT <: Manifold, P <: MPoint, T <: TVectorE} = dot(M,x,getVector(ξ),ν)
dot(M::mT, x::P, ξ::S, ν::T) where {mT <: Manifold, P <: MPoint, T <: TVectorE, S <: TVector} = dot(M,x,ξ,getVector(ν) )
dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPoint, T <: TVectorE, S <: TVector} = dot(M,x,ξ,getVector(ν) )

# extended exp check base and return exp of value if that did not fail
exp(M::mT,x::S,ξ::T) where {mT<:Manifold, T<:TVectorE, S<:MPointE} = exp(M,getBase(x),ξ)
function exp(M::mT,x::S,ξ::T) where {mT<:Manifold, T<:TVectorE, S<:MPoint}
    checkBase(ξ,x);
    return exp(M,x, getVector(ξ) )
end
# for extended vectors set the base to true
log(M::mT,x::P,y::P) where {mT<:Manifold, P<:MPoint} = TVectorE(log(M,getBase(x),getBase(y)),getBase(x));
# break down to inner if base
manifoldDimension(x::P) where {P <: MPointE} = manifoldDimension(getBase(x))
# break down to inner if base is checked
function norm(M::mT, x::P, ξ::T, ν::T)::Float64 where {mT<:Manifold, P <: MPoint, T<:TVectorE}
    checkBase(ξ,x," first ")
    checkBase(ξ,ν," second ")
    return norm(M,getBase(x),getVector(ξ),ν.vector);
end
norm(M::mT,x::P,ξ::T,ν::S) where {mT<:Manifold, P <: MPointE, T<:TVector, S<:TVector} = dot(M,getBase(x),ξ,ν);
norm(M::mT,x::P,ξ::T,ν::S) where {mT<:Manifold, P <: MPoint, T<:TVectorE, S<:TVector} = dot(M,getBase(x), getVector(ξ) ,ν);
norm(M::mT,x::P,ξ::S,ν::T) where {mT<:Manifold, P <: MPoint, T<:TVectorE, S<:TVector} = dot(M,getBase(x), ξ, getVector(ν));
# (a) x,ξ extended -> check, y not -> check but strip
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPointE, Q <: MPoint, T<: TVectorE}
    checkBase(ξ,x)
    return TVectorE( parallelTransport(M,getBase(x),y,getVector(ξ)), y)
end
# (b) x,ξ extended -> y, too, Extend result
function parallelTransport(M::mT,x::P,y::P,ξ::T) where {mT <: Manifold, P <: MPointE, T<: TVectorE}
    checkBase(x,ξ)
    return TVectorE(parallelTransport(M,getBase(x),y,getVector(ξ)), y)
end
# (c) remaining combinations, just strip
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPoint, Q <: MPointE, T<: TVectorE}
    return TVectorE( parallelTransport( M,x,y,getVector(ξ) ), y)
end
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPointE, Q <: MPointE, T<: TVector}
    return TVectorE( parallelTransport( M,getBase(x),y,ξ ), y)
end
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPoint, Q <: MPoint, T<: TVectorE}
    return parallelTransport( M,x,y,getVector(ξ) )
end
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPointE, Q <: MPoint, T<: TVector}
    return parallelTransport( M,getBase(x),y,ξ )
end
#
# tangentONB (a) x is ext -> extended Tangents,
function tangentONB(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPointE, Q <: MPoint}
    V,κ = tangentONB(M,getBase(x),log(M,getBase(x),y))
    return TVectorE.(V, Ref(x)),κ
end
function tangentONB(M::mT, x::Q, y::P) where {mT <: Manifold, P <: MPointE, Q <: MPoint}
    V,κ = tangentONB(M,x,log(M,x,getBase(y)))
    return TVectorE.(V, Ref(x)),κ
end
function tangentONB(M::mT, x::P, y::P) where {mT <: Manifold, P <: MPointE}
  V,κ = tangentONB(M,getBase(x),log(M,getBase(x), getBase(y)))
  return TVectorE.(V, Ref(getBase(x)) ),κ
end
typicalDistance(x::P) where {P <: MPointE} = typicalDistance(getBase(x))

zeroTVector(x::P) where {P <: MPointE} = TVectorE( zeroTVector(getBase(x)) )
