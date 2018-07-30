export getValue, getBase, checkBase
export addNoise, distance, dot, exp, getValue, log, manifoldDimension, norm
export manifoldDimension, parallelTransport, tangentONB, typicalDistance, zeroTVector
#
#
#
# The extended types for more information/security on base points of tangent vectors
# ---
"""
A decorator pattern based extension of TVector to additionally store the base
point. The decorator is then used to verify, that exp and dot are only called
with correct base points.
"""
struct TVectorE{T <: TVector, P <: MPoint} <: TVector
    vector::T
    base::P
end
getValue{T <: TVectorE}(ξ::T) = getValue(ξ.vector)
"""
    getBase(ξ)
returns the base point of an extended tangent vector.
"""
getBase{T <: TVectorE}(ξ::T) = ξ.base
"""
    getVector(ξ)
returns the internal TVector point of an extended tangent vector.
"""
getVector{T <: TVectorE}(ξ::T) = ξ.vector
"""
A decorator pattern based extension of MPoint to identify when to switch
to the extended `TVectorE` for functions just working on points, e.g. `log`
"""
struct MPointE{P <: MPoint} <: MPoint
    base::P
end
getValue{P <: MPointE}(x::P) = getValue( getBase(x) );
show(io::IO, x::MPointE) = print(io, "$(x)E")
"""
    getBase(x)
returns the point this extended manifold point stores internally.
"""
getBase{P <: MPointE}(x::P) = x.base;

show(io::IO, ξ::TVectorE) = print(io, "$( getValue(ξ) )_$( getValue( getBase(ξ) ) )")
function +{T <: TVectorE}(ξ::T,ν::T)
    checkBase(ξ,ν)
    return T(ξ.value+ν.value,ξ.base)
end
function -{T <: TVectorE}(ξ::T,ν::T)::T
    checkBase(ξ,ν)
    return T(ξ.value-ν.value,ξ.base)
end
"""
    checkBase(ξ,ν)
checks, whether the base of two tangent vectors is identical, if both tangent
vectors are of type `TVectorE`. If one of them is not an extended vector, the
function returns true, expecting the tangent vector implicitly to be correct.
"""
function checkBase{T <: TVectorE}(ξ::T,ν::T)
    if getValue( getBase(ξ) ) != getValue( getBase(ν) )
        throw(
            ErrorException("The two tangent vectors $ξ and $ν do not have the same base.")
        );
    else
        return true;
    end
end
checkBase{T <: TVectorE, S <: TVector}(ξ::T,ν::S) = true
checkBase{T <: TVectorE, S <: TVector}(ξ::S,ν::T) = true
"""
    checkBase(ξ,x)
checks, whether the base of the tangent vector `ξ` is `x`. If `ξ` is not an
extended tangent vector `TVectorE` the function returns true, assuming the base
implicitly to be correct
"""
function checkBase{T <: TVectorE, P <: MPoint}(ξ::T,x::P)
    if getValue( getBase(ξ) ) != getValue(x)
        throw(
            ErrorException("The tangent vector $ξ is not from the tangent space of $x")
        );
    else
        return true;
    end
end
checkBase{T<: TVector, P<: MPoint}(ξ::T,x::P) = true
# unary operators
*{T <: TVectorE}(ξ::T,s::Number)::T = T(s*ξ.value,ξ.base)
*{T <: TVectorE}(s::Number, ξ::T)::T = T(s*ξ.value,ξ.base)
# /
/{T <: TVectorE}(ξ::T,s::Number)::T = T(ξ.value./s,ξ.base)
/{T <: TVectorE}(s::Number, ξ::T)::T = T(s./ξ.value,ξ.base)
-{T <: TVectorE}(ξ::T)::T = T(-ξ.value,ξ.base)
+{T <: TVectorE}(ξ::T)::T = T(ξ.value,ξ.base)

# compare extended vectors
=={T <: TVectorE}(ξ::T,ν::T)::Bool = ( checkBase(ξ,ν) && all(ξ.value==ν.value) )
#
# encapsulate default functions
#
addNoise(M::mT,x::P,σ) where {mT <: Manifold, P <: MPointE} = MPointE( addNoise(M,getBase(x),σ) )
distance(M::mT, x::T, y::T) where {mT <: Manifold, T <: MPointE} = distance(M,getBase(x),getBase(y))

function dot(M::mT, x::P, ξ::T, ν::T)::Float64 where {mT<:Manifold, P <: MPointE, T<:TVectorE}
    checkBase(ξ,x);
    checkBase(ξ,ν);
    return dot(M, getVector(ξ), getVector(ν) );
end
# all others (i.e. one implicitly assumed to be correct -> pass)
dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPointE, T <: TVectorE, S <: TVector} = dot(M,getBase(x),getVector(ξ),ν)
dot(M::mT, x::P, ξ::S, ν::T) where {mT <: Manifold, P <: MPointE, T <: TVectorE, S <: TVector} = dot(M,getBase(x),ξ,getVector(ν) )
dot(M::mT, x::P, ξ::S, ν::S) where {mT <: Manifold, P <: MPointE, S <: TVector} = dot(M,getBase(x),ξ,ν)
dot(M::mT, x::P, ξ::T, ν::T) where {mT <: Manifold, P <: MPoint, T <: TVectorE} = dot(M,x,getVector(ξ),getVector(ν))
dot(M::mT, x::P, ξ::T, ν::S) where {mT <: Manifold, P <: MPoint, T <: TVectorE, S <: TVector} = dot(M,x,getVector(ξ),ν)
dot(M::mT, x::P, ξ::S, ν::T) where {mT <: Manifold, P <: MPoint, T <: TVectorE, S <: TVector} = dot(M,x,ξ,getVector(ν) )

# extended exp check base and return exp of value if that did not fail
exp{mT<:Manifold, T<:TVectorE, S<:MPointE}(M::mT,x::S,ξ::T)::T = exp(M,getBase(x),ξ)
function exp{mT<:Manifold, T<:TVectorE, S<:MPoint}(M::mT,x::S,ξ::T)::T
    checkBase(ξ,x);
    return exp(M,x, getVector(ξ) );
end
# for extended vectors set the base to true
log{mT<:Manifold, P<:MPointE}(M::mT,x::P,y::P) = TVectorE(log(M,getBase(x),getBase(y)),x);
log{mT<:Manifold, P<:MPointE, Q<:MPoint}(M::mT,x::P,y::Q) = TVectorE(log(M,getVector(x),y),x);
log{mT<:Manifold, P<:MPointE, Q<:MPoint}(M::mT,x::Q,y::P) = TVectorE(log(M,x,getVector(y)),x);
# break down to inner if base
manifoldDimension(x::P) where {P <: MPointE} = manifoldDimension(getBase(x))
# break down to inner if base is checked
function norm(M::mT, x::P, ξ::T, ν::T)::Float64 where {mT<:Manifold, P <: MPoint, T<:TVectorE}
    checkBase(ξ,x);
    checkBase(ξ,ν);
    return norm(M,x,ξ.value,ν.value);
end
norm{mT<:Manifold, P <: MPointE, T<:TVector, S<:TVector}(M::mT,x::P,ξ::T,ν::S) = dot(M,getBase(x),ξ,ν);
norm{mT<:Manifold, P <: MPoint, T<:TVectorE, S<:TVector}(M::mT,x::P,ξ::T,ν::S) = dot(M, getVector(ξ) ,ν);
norm{mT<:Manifold, P <: MPoint, T<:TVectorE, S<:TVector}(M::mT,x::P,ξ::S,ν::T) = dot(M, ξ, getVector(ν));
# (a) x,ξ extended -> check, y not -> check but strip
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPointE, Q <: MPoint, T<: TVectorE}
    checkBase(x,ξ)
    return parallelTransport(M,getBase(x),y,getVector(ξ))
end
# (b) x,ξ extended -> y, too, Extend result
function parallelTransport(M::mT,x::P,y::P,ξ::T) where {mT <: Manifold, P <: MPointE, T<: TVectorE}
    checkBase(x,ξ)
    return TVectorE( parallelTransport(M,getBase(x),y,getVector(ξ)), y)
end
# (c) remaining combinations, just strip
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPoint, Q <: MPointE, T<: TVectorE}
    return TVectorE( parallelTransport( M,x,y,getVector(ξ) ), y);
end
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPointE, Q <: MPointE, T<: TVector}
    return TVectorE( parallelTransport( M,getBase(x),y,ξ ), y);
end
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPoint, Q <: MPoint, T<: TVectorE}
    return parallelTransport( M,x,y,getVector(ξ) );
end
function parallelTransport(M::mT,x::P,y::Q,ξ::T) where {mT <: Manifold, P <: MPointE, Q <: MPoint, T<: TVector}
    return parallelTransport( M,getBase(x),y,ξ );
end
#
# tangentONB (a) x is ext -> extended Tangents,
tangentONB(M::mT, x::P, y::Q) where {mT <: Manifold, P <: MPointE, Q <: MPoint} = TVectorE.(tangentONB(M,getBase(x),log(M,getBase(x),y)))
tangentONB(M::mT, x::Q, y::P) where {mT <: Manifold, P <: MPointE, Q <: MPoint} = tangentONB(M,x,log(M,x,getBase(y) ))
tangentONB(M::mT, x::P, y::P) where {mT <: Manifold, P <: MPointE} = TVectorE.(tangentONB(M,getBase(x),log(M,getBase(x),getBase(y))))

typicalDistance(x::P) where {P <: MPointE} = typicalDistance(getBase(x))

zeroTVector(x::P) where {P <: MPointE} = TVectorE( zeroTVector(getBase(x)) )
