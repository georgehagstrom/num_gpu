# Writes reference/num_classes.json: per biomass class mass (ugC), lower bound and bin width, group name/type,
# as set up by setup_num_model (same as Matlab setupNUMmodel). Used by notebooks/a100_run.ipynb.
using NUMGPU
s = setup_num_model(joinpath(@__DIR__, "..", "NUMmodel", "input", "input.yaml"))
names = Dict(NUMGPU.generalist => "Generalists", NUMGPU.diatom => "Diatoms",
             NUMGPU.copepod_passive => "Passive copepods", NUMGPU.copepod_active => "Active copepods",
             NUMGPU.pom => "POM")
rows = String[]
for (gi, gr) in enumerate(s.groups), i in gr.ix
    push!(rows, "{\"m\": $(s.m[i]), \"mLower\": $(s.mLower[i]), \"mDelta\": $(s.mDelta[i]), \"group\": $gi, \"type\": \"$(names[gr.type])\"}")
end
open(joinpath(@__DIR__, "..", "reference", "num_classes.json"), "w") do io
    println(io, "[\n  ", join(rows, ",\n  "), "\n]")
end
