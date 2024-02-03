push!(LOAD_PATH, "../src/")
using Documenter, TermStructureModels

makedocs(
    modules=[TermStructureModels],
    sitename="TermStructureModels.jl",
    pages=[
        "Overview" => "index.md",
        "Estimation" => "estimation.md",
        "Inference" => "inference.md",
        "Forecasting" => "scenario.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo="github.com/econPreference/TermStructureModels.jl.git",
    branch="gh-pages"
)