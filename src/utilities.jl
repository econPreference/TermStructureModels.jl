"""
    getindex(x::Vector{<:PosteriorSample}, c::Symbol)
For `struct <: PosteriorSample`, `struct[:name]` calls objects in `struct`. `Output[i]` = ``i'``th posterior sample
"""
function getindex(x::Vector{<:PosteriorSample}, c::Symbol)
    return getproperty.(x, c)
end

"""
    getindex(x::PosteriorSample, c::Symbol)
For `struct <: PosteriorSample`, `struct[:name]` calls objects in struct.
"""
function getindex(x::PosteriorSample, c::Symbol)
    return getproperty(x, c)
end

"""
    mean(x::Vector{<:PosteriorSample})
`Output[:variable name]` returns the corresponding posterior mean.
"""
function mean(x::Vector{<:PosteriorSample})
    names = fieldnames(eltype(x))
    args = []
    for i in eachindex(names)
        push!(args, mean(x[names[i]]))
    end
    return eltype(x)(args...)
end

"""
    median(x::Vector{<:PosteriorSample})
`Output[:variable name]` returns the corresponding posterior median.
"""
function median(x::Vector{<:PosteriorSample})
    names = fieldnames(eltype(x))
    args = []
    for i in eachindex(names)
        saved = x[names[i]]
        if typeof(saved[1]) <: VecOrMat
            result = similar(saved[1])
            for j in axes(result, 1), k in axes(result, 2)
                result[j, k] = median([saved[iter][j, k] for iter in 1:length(x)])
            end
        else
            result = median(saved)
        end
        push!(args, result)
    end
    return eltype(x)(args...)
end

"""
    std(x::Vector{<:PosteriorSample})
`Output[:variable name]` returns the corresponding posterior standard deviation.
"""
function std(x::Vector{<:PosteriorSample})
    names = fieldnames(eltype(x))
    args = []
    for i in eachindex(names)
        push!(args, std(x[names[i]]))
    end
    return eltype(x)(args...)
end

"""
    var(x::Vector{<:PosteriorSample})
`Output[:variable name]` returns the corresponding posterior variance.
"""
function var(x::Vector{<:PosteriorSample})
    names = fieldnames(eltype(x))
    args = []
    for i in eachindex(names)
        push!(args, var(x[names[i]]))
    end
    return eltype(x)(args...)
end

"""
    quantile(x::Vector{<:PosteriorSample}, q)
`Output[:variable name]` returns a quantile of the corresponding posterior distribution.
"""
function quantile(x::Vector{<:PosteriorSample}, q)
    names = fieldnames(eltype(x))
    args = []
    for i in eachindex(names)
        saved = x[names[i]]
        if typeof(saved[1]) <: VecOrMat
            result = similar(saved[1])
            for j in axes(result, 1), k in axes(result, 2)
                result[j, k] = quantile([saved[iter][j, k] for iter in 1:length(x)], q)
            end
        else
            result = quantile(saved, q)
        end
        push!(args, result)
    end
    return eltype(x)(args...)
end