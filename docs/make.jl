push!(LOAD_PATH, "../src/")
using Documenter, TermStructureModels

makedocs(
    modules=[TermStructureModels],
    sitename="TermStructureModels.jl",
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo="github.com/econPreference/TermStructureModels.jl.git",
    branch="gh-pages"
)