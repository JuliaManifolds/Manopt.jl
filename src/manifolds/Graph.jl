#
#      Graphmanifold – an array of points of one manifold,
#          where their vicinity or neighborhoods are given by a graph
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show

export Graph, GraphVertexPoint, GraphVertexTVector, GraphEdgePoint, GraphEdgeTVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export zeroTVector
export show, getValue
@doc doc"""
    Graph{M<:Manifold} <: Manifold
The graph manifold models manifold-valued data on a graph
$\mathcal G = (\mathcal Vm \mathcal E)$, both on vertices
and edges as well as their interplay. The adjacency is stored in a matrix,
and may contain also the weights.

Since there are two possibilities in dimensions, $\lvert\mathcal V\rvert$ and
$\lvert\mathcal E\rvert$, the manifold itself will refer to the first one, while
depending on the type of `MPoint` one of them is returned.

# Fields
the default values are given in brackets
* `adjacency` – the (sparse) adjacency matrix, might also carry weights, i.e. all
  $a_{ij}>0$ refer to adjacent nodes $i$ and $j$, see [`addEdge`](@ref)
  and [`removeEdge`](@ref)
* `name` – (`A Graph manifold of \$Submanifold.`) name of the manifold
* `manifold` – the internal manifold present at vertices (edges) for [`GraphVertexPoint`](@ref) ([`GraphEdgePoint`](@ref))
* `dimension` – stores the dimension of the manifold of a `GraphVertexPoint`
* `isDirected` – (`false`) indicates whether the graph is directed or not.
"""
struct Graph{M<:Manifold} <: Manifold
  name::String
  manifold::M
  dimension::Int
  addjacency::Mat where {Mat <: AbstractMatrix}
  isDirected::Bool
  abbreviation::String
  Graph{M}(mv::M,adjacency::Mat where {Mat <: AbstractMatrix}, isDir::Bool=false) where {M <: Manifold} = new(string("A Graph Manifold of ",mv.name,"."),
    mv,size(adjacency,1)*manifoldDimension(mv),isDir,string("GraphVertex(",m.abbreviation,",",repr(size(adjacency,1)),")") )
end
@doc doc"""
    GraphVertexPoint <: MPoint
A point graph vertex power manifold
$\mathcal M = \mathcal N^{\lvert\mathcal V\rvert}$
represented by a vector of corresponding [`MPoint`](@ref)s.
"""
struct GraphVertexPoint <: MPoint
  value::Vector{P} where {P<:MPoint}
  GraphVertexPoint(v::Vector{P} where {P<:MPoint}) = new(v)
end
getValue(x::GraphVertexPoint) = x.value;

@doc doc"""
    GraphVertexTVector
A tangent vector $\xi\in T_x\mathcal M$ to the graph vertex power manifold
$\mathcal M = \mathcal N^{\lvert\mathcal V\rvert}$
represented by a vector of corresponding [`TVector`](@ref)s.
"""
struct GraphVertexTVector <: TVector
  value::Vector{T} where {T <: TVector}
  GraphVertexTVector(value::Vector{T} where {T <: TVector}) = new(value)
end
getValue(ξ::GraphVertexTVector) = ξ.value

@doc doc"""
    GraphEdgePoint <: MPoint
A point graph edge power manifold
$\mathcal M = \mathcal N^{\lvert\mathcal E\rvert}$
represented by a (sparse/not completely filled matrix of corresponding
[`MPoint`](@ref)s.
"""
struct GraphEdgePoint <: MPoint
  value::V where {V <: AbstractMatrix{P} where {P<:MPoint} }
  GraphEdgePoint(v::V where { V <: AbstractMatrix{P} where {P<:MPoint} }) = new(v)
end
getValue(x::GraphEdgePoint) = x.value;

@doc doc"""
    GraphEdgeTVector
A tangent vector $\xi\in T_x\mathcal M$ to the graph edge power manifold
$\mathcal M = \mathcal N^{\lvert\mathcal E\rvert}$
represented by a (sparse/not completely filled) matrix of corresponding
[`TVector`](@ref)s.
"""
struct GraphEdgeTVector <: TVector
  value::V where {V <: AbstractMatrix{T} where {T<:TVector}}
  GraphEdgePoint(v::V where { V <: AbstractMatrix{T} where {T<:TVector} }) = new(v)
end
getValue(ξ::GraphEdgeTVector) = ξ.value

# Functions
# ---
"""
    addNoise(M,x,δ)
computes a vectorized version of addNoise, and returns the noisy [`GraphVertexPoint`](@ref).
"""
addNoise(M::Graph, x::GraphVertexPoint,σ) = GraphVertexPoint(addNoise.(M.manifold,p.value,σ))
"""
    addNoise(M,x,δ)
computes a vectorized version of addNoise, and returns the noisy [`GraphEdgePoint`](@ref).
"""
addNoise(M::Graph, x::GraphEdgePoint,σ) = GraphEdgePoint(addNoise.(M.manifold,p.value,σ))
"""
    distance(M,x,y)
computes a vectorized version of distance, and the induced norm from the metric [`dot`](@ref).
"""
distance(M::Graph, x::GraphVertexPoint, y::GraphVertexPoint) = sqrt(sum( distance.(M.manifold, getValue(x), getValue(y) ).^2 ))
"""
    distance(M,x,y)
computes a vectorized version of distance, and the induced norm from the metric [`dot`](@ref).
"""
distance(M::Graph, x::GraphEdgePoint, y::GraphEdgePoint) = sqrt(sum( distance.(M.manifold, getValue(x), getValue(y) ).^2 ))
"""
    dot(M,x,ξ,ν)
computes the inner product as sum of the component inner products on the [`Graph`](@ref) vertices.
"""
dot(M::Graph, x::GraphVertexPoint, ξ::GraphVertexTVector, ν::GraphVertexTVector) = sum(dot.(M.manifold,getValue(x), getValue(ξ), getValue(ν) ))
"""
    dot(M,x,ξ,ν)
computes the inner product as sum of the component inner products on the [`Graph`](@ref) edges.
"""
dot(M::Graph, x::GraphEdgePoint, ξ::GraphEdgeTVector, ν::GraphEdgeTVector) = sum(dot.(M.manifold,getValue(x), getValue(ξ), getValue(ν) ))
"""
    exp(M,x,ξ)
computes the product exponential map on the [`Graph`](@ref) vertices and returns the corresponding [`GraphVertexPoint`](@ref).
"""
exp(M::Graph, x::GraphVertexPoint, ξ::GraphVertexTVector, t::Number=1.0) = GraphVertexPoint( exp.(M.manifold, getValue(p) , getValue(ξ) ))
"""
   log(M,x,y)
computes the product logarithmic map on the [`Power`](@ref) and returns the corresponding [`PowTVector`](@ref).
"""
log(M::Graph, x::GraphVertexPoint, y::GraphVertexPoint)::GraphVertexTVector = GraphVertexTVector(log.(M.manifold, getValue(p), getValue(q) ))
"""
   log(M,x,y)
computes the product logarithmic map on the [`Power`](@ref) and returns the corresponding [`PowTVector`](@ref).
"""
log(M::Graph, x::GraphEdgePoint, y::GraphEdgePoint)::GraphEdgeTVector = GraphEdgeTVector(log.(M.manifold, getValue(p), getValue(q) ))
"""
    manifoldDimension(x)
returns the (product of) dimension(s) of the [`Graph`](@ref) the
[`GraphVertexPoint`](@ref)`x` belongs to.
"""
manifoldDimension(p::GraphVertexPoint) = prod(manifoldDimension.( getValue(p) ) )
"""
    manifoldDimension(x)
returns the (product of) dimension(s) of the [`Power`](@ref) the
[`GraphEdgePoint`](@ref)`x` belongs to.
"""
manifoldDimension(p::GraphEdgePoint) = prod(manifoldDimension.( getValue(p) ) )
"""
    manifoldDimension(M)
returns the (product of) dimension(s) of the [`Graph`](@ref)` M` seen as a vertex power manifold.
"""
manifoldDimension(M::Graph) = size(M.adjacency,1) * manifoldDimension(M.manifold)
"""
    norm(M,x,ξ)
norm of the [`GraphVertexTVector`]` ξ` induced by the metric on the manifold components
of the [`Graph`](@ref)` M`.
"""
norm(M::Graph, x::GraphVertexPoint, ξ::GraphVertexTVector) = sqrt( dot.(M.manifold,x,ξ,ξ) )
"""
    norm(M,x,ξ)
norm of the [`GraphEdgeTVector`]` ξ` induced by the metric on the manifold components
of the [`Graph`](@ref)` M`.
"""
norm(M::Graph, x::GraphEdgePoint, ξ::GraphEdgeTVector) = sqrt( dot.(M.manifold,x,ξ,ξ) )

@doc doc"""
    parallelTransport(M,x,ξ)
computes the product parallelTransport map on the [`Graph`](@ref) vertex power
manifold $\mathcal M^{\lvert\mathcal V\rvert}$ and returns
the corresponding [`GraphVertexTVector`](@ref).
"""
parallelTransport(M::Graph, x::GraphVertexPoint, y::GraphVertexPoint, ξ::GraphVertexTVector) = GraphVertexTVector( parallelTransport.(M.manifold, getValue(x), getValue(y), getValue(ξ)) )
"""
    parallelTransport(M,x,ξ)
computes the product parallelTransport map on the [`Graph`](@ref) edge power manifold and returns
the corresponding [`GraphVertexTVector`](@ref).
"""
parallelTransport(M::Graph, x::GraphEdgePoint, y::GraphEdgePoint, ξ::GraphEdgeTVector) = GraphVertexTVector( parallelTransport.(M.manifold, getValue(x), getValue(y), getValue(ξ)) )
@doc doc"""
    typicalDistance(M)
returns the typical distance on the [`Graph`](@ref) vertex manifold `M`, based
on the typical distance of the base.
"""
typicalDistance(M::Graph) = sqrt( size(M.adjacency,1) ) * typicalDistance(M.manifold);
@doc doc"""
    ξ = zeroTVector(M,x)
returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`GraphVertexPoint`](@ref) $x\in\mathcal M$ on the [`Graph`](@ref) vertex manifold `M`.
"""
zeroTVector(M::Graph, x::GraphVertexPoint) = GraphVertexTVector( zeroTVector.(M.manifold, getValue(x) )  )
@doc doc"""
    ξ = zeroTVector(M,x)
returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`GraphEdgePoint`](@ref) $x\in\mathcal M$ on the [`Graph`](@ref) edge manifold `M`.
"""
zeroTVector(M::Graph, x::GraphEdgePoint) = GraphEdgeTVector( zeroTVector.(M.manifold, getValue(x) )  )
@doc doc"""
   startEdgePoint(M,x)
For a [`Graph`](@ref) manifold and a [`GraphVertexPoint`](@ref), this
function constructs the corresponding [`GraphEdgePoint`], such that each
edge has its start point vertex value assigned.
"""
function startEdgePoint(M::Graph, x::GraphVertexPoint)::GraphEdgePoint
  if length(getValue(x)) == 0
    throw( ErrorException("vertexToStartEdgePoint::No node given"))
  end
  if issparse(M.adjacency)
    (s,e) = findn(M.adjacency)
    return GraphEdgePoint( sparse(s,e, getValue(x)[ s ]) )
  else
    sA = size(M.adjacency)
    A = Matrix(typeof(x[1]),sA)
    for i=1:sA[1]
      for j=1:sA[2]
        if M.adjacency[i,j] > 0
          A[i,j] = getValue(x)[i]
        else
          A[i,j] = missing;
        end
      end
    end
  end
end
@doc doc"""
    sumEdgeTVector(M,ξ)
return the [ `GraphVertexTVector`](@ref) where edge tangents are summed in their start point.

For an [`GraphEdgeTVector`](@ref)` ξ` on a [`Graph`](@ref) manifold `M`
this function assumes that all edge tangents are attached in a tangent space
corresponding to the same point on the base manifold, i.e. all these vectors can
be summed. This sum per vectex is then returned as a
[`GraphVertexTVector`](@ref).
"""
function sumEdgeTVector(M::Graph, ξ::GraphEdgeTVector, weighted::Bool=false)
  (s,e) = findn(M.adjacency)
  init = falses(size(M.adjacency,1)) # vector initialized?
  lξ = getValue(ξ)
  ν = Vector(  eltype( lξ ),size(M,addjacency,1)  )
  for i=1:length(s) # all edges
    if !init[ s[i] ]
      ν[ s[i] ] = lξ[s[i],e[i]] # initialize (i.e. also carry base if it exists)
      init[ s[i] ] = true
    else
      ν[ s[i] ] = ν[ s[i] ] + lξ[s[i], e[i]]
    end
  end
  return GraphVertexTVector(ν)
end
#
#
# Display functions for the structs
show(io::IO, M::Graph) = print(io,string("The manifold on vertices and edges of a graph of ",repr(M.manifold), " of (vertex manifold) dimension ",repr(M.dims),".") );
show(io::IO, p::GraphVertexPoint) = print(io,string("GraphVertexV[",join(repr.(p.value),", "),"]"));
show(io::IO, ξ::GraphVertexTVector) = print(io,String("GraphVertexT[", join(repr.(ξ.value),", "),"]"));
show(io::IO, p::GraphEdgePoint) = print(io,string("GraphEdgeV[",join(repr.(p.value),", "),"]"));
show(io::IO, ξ::GraphEdgeTVector) = print(io,String("GraphEdgeT[", join(repr.(ξ.value),", "),"]"));
