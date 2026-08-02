using Documenter
using FerriteSolidMechanics

makedocs(;
    modules=[FerriteSolidMechanics],
    sitename="FerriteSolidMechanics.jl",
    repo=Remotes.GitHub("LukasLaubert", "FerriteSolidMechanics.jl"),
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://LukasLaubert.github.io/FerriteSolidMechanics.jl",
        edit_link="main",
        repolink="https://github.com/LukasLaubert/FerriteSolidMechanics.jl",
        collapselevel=1,
    ),
    pages=[
        "Home" => "index.md",
        "Material models" => [
            "Overview" => "models/index.md",
            "Linear Elasticity" => "models/linear_elasticity.md",
            "Hyperelastic" => [
                "Neo–Hookean" => "models/neo_hooke.md",
                "Arruda–Boyce" => "models/arruda_boyce.md",
                "Mooney–Rivlin" => "models/mooney_rivlin.md",
                "Ogden" => "models/ogden.md",
            ],
            "J2 Plasticity" => "models/j2_plasticity.md",
            "VEPD Detrez 2010" => "models/vepd_detrez2010.md",
            "VEVP Zhao 2021" => "models/vevp_zhao2021.md",
            "VEVP MOAMMM" => "models/vevp_moammm.md",
            "Experimental models" => "experimental.md",
        ],
        "Tutorials" => [
            "2D plate with a hole" => "tutorials/plate_with_hole.md",
            "DMA cantilever beam" => "tutorials/cantilever_dma.md",
            "Adaptive time stepping" => "tutorials/adaptive_time_stepping.md",
            "MPI four-point bending" => "tutorials/mpi_four_point_bending.md",
        ],
        "Concepts" => "concepts.md",
        "External loads" => "loads.md",
        "In-plane wrappers" => "wrappers.md",
        "Performance and parallel execution" => "performance.md",
        "Developer guide" => "developer_guide.md",
        "FAQ" => "faq.md",
        "API reference" => "api.md",
        "Material model API" => "api_models.md",
    ],
)

deploydocs(;
    repo="github.com/LukasLaubert/FerriteSolidMechanics.jl",
    devbranch="main",
    push_preview=true,
)
