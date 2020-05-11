module LinearMapDuals
using ForwardDiff
import ForwardDiff: JacobianConfig, Dual
using DiffResults
using LinearMaps
import ToeplitzMatrices: SymmetricToeplitz

export LinearMapResult, LinearMapDual

struct LinearMapDual{T,V,N}
  val::LinearMap{V}
  jacobians::NTuple{N,LinearMap{V}}
end

mutable struct LinearMapResult{V} <: DiffResults.DiffResult{1,LinearMap{V},NTuple{1, LinearMap{V}}}
  val::Union{Nothing, LinearMap{V}}
  jacobians::Vector{LinearMap{V}}
end

LinearMapResult{T}() where {T} = LinearMapResult{T}(nothing, [])


# Hacks to make ForwardDiff compatible with LinearMapResult

ForwardDiff.reshape_jacobian(a::LinearMapResult, _, _) = a

function ForwardDiff.extract_jacobian!(::Type{T}, result::LinearMapResult{V}, ydual::LinearMapDual{T,V,N}, _) where {T,V,N}
    result.jacobians = collect(ydual.jacobians)
end

function ForwardDiff.extract_jacobian_chunk!(::Type{T}, result::LinearMapResult{V}, ydual::LinearMapDual{T,V,N}, _, _) where {T,V,N}
    append!(result.jacobians, ydual.jacobians)
end

function ForwardDiff.extract_value!(::Type{T}, result::LinearMapResult{V}, ydual::LinearMapDual{T,V,N}) where {T,V,N}
    result.val = ydual.val
end


# Hack to make ForwardDiff transparently switch to LinearMapDual

LinearMaps.WrappedMap{Dual{T,V,N}, A}(lmap::AbstractMatrix{Dual{T,V,N}}, issymmetric, ishermitian, isposdef) where {T,V,N, A <: AbstractMatrix{Dual{T,V,N}}} = LinearMapDual{T,V,N}(
     LinearMaps.WrappedMap{V,AbstractMatrix{Float64}}(map(d-> ForwardDiff.value(T, d), lmap), issymmetric, ishermitian, isposdef),
    Tuple(LinearMap(map(d-> d.partials[i], lmap)) for i in 1:N))


# Kronecker product of a dual. Currently requires that all arguments are LinearMaps.
# TODO: auto convert array arguments, handle \otimes too
Base.kron(a::LinearMapDual{T,V,N}, b::LinearMapDual{T,V,N}) where {T,V,N} =
    LinearMapDual{T,V,N}(
        kron(a.val, b.val),
        Tuple(kron(a.jacobians[i], b.val) + kron(a.val, b.jacobians[i]) for i in 1:N))
         
Base.kron(maps::LinearMapDual{T,V,N}...) where {T,V,N} = reduce(kron, maps)


# Hack to make ToeplitzMatrices convert to a LinearMapDual

SymmetricToeplitz(v::Vector{Dual{T,V,N}}) where {T,V,N} = LinearMapDual{T,V,N}(
    LinearMaps.WrappedMap{V,SymmetricToeplitz{Float64}}(SymmetricToeplitz(map(d-> ForwardDiff.value(T, d), v)), true, true, true),
    Tuple(LinearMaps.WrappedMap{V,SymmetricToeplitz{Float64}}(SymmetricToeplitz{Float64}(map(d-> d.partials[i], v)), true, true, true) for i in 1:N))

LinearMaps.LinearMap(d::LinearMapDual; kwargs...) = d

# TODO: Allow adding and multiplying LinearMaps. Look at LinearMaps code for inspiration
# TODO: Add support for KroneckerSum

end # module
