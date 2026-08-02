using Test
using FerriteSolidMechanics

# A docstring detaches silently if anything separates it from its definition

@testset "every exported symbol has an attached docstring" begin
    # meta(m) holds only bindings a docstring actually attached to
    meta = Base.Docs.meta(FerriteSolidMechanics)
    undocumented = String[]
    for sym in names(FerriteSolidMechanics)
        sym === :FerriteSolidMechanics && continue
        haskey(meta, Base.Docs.Binding(FerriteSolidMechanics, sym)) || push!(undocumented, string(sym))
    end
    @test undocumented == String[]
end
