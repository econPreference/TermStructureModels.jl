push!(LOAD_PATH, "../src/")
using Documenter, TermStructureModels

makedocs(
    modules=[TermStructureModels],
    sitename="TermStructureModels.jl",
    pages=[
        "Overview" => "index.md",
        "Notations" => "notations.md",
        "Estimation" => "estimation.md",
        "Statistical Inference" => "inference.md",
        "Forecasting" => "scenario.md",
        "Utilization of the Output" => "output.md",
        "Other Forms of the Model" => "others.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo="github.com/econPreference/TermStructureModels.jl.git",
    branch="gh-pages"
)