# Readers for the NUM input file, mirroring Fortran/read_input.f90: input.yaml (upstream
# Develop, 2026) and the older input.h (v1.0). In both, a value belongs to the section whose
# header was seen last, and if a key appears twice in a section the last occurrence wins.

const SECTION_HEADERS = [
    "! GENERAL PARAMETERS" => "general",
    "! GENERALISTS SIMPLE INPUT PARAMETERS" => "generalists_simple",
    "! GENERALISTS INPUT PARAMETERS" => "generalists",
    "! DIATOMS SIMPLE INPUT PARAMETERS" => "diatoms_simple",
    "! DIATOMS INPUT PARAMETERS" => "diatoms",
    "! COPEPODS PASSIVE INPUT PARAMETERS" => "copepods_passive",
    "! COPEPODS ACTIVE INPUT PARAMETERS" => "copepods_active",
    "! PARTICULATE ORGANIC MATTER (POM) INPUT PARAMETERS" => "POM",
]

"""
    read_input_file(path) -> Dict{String,Dict{String,Float64}}

Parse the NUM input file (input.yaml, or the v1.0 input.h) into section => (key => value).
"""
read_input_file(path::AbstractString) =
    endswith(path, ".yaml") ? read_input_yaml(path) : read_input_h(path)

# input.yaml as the Fortran reads it: a section header is a line without leading space
# ending in ':'; indented "key: value  # comment" lines belong to the current section;
# lines whose first non-space character is '#' and keys without a value are skipped.
function read_input_yaml(path::AbstractString)
    params = Dict{String,Dict{String,Float64}}()
    section = ""
    for rawline in eachline(path)
        line = strip(rawline)
        (isempty(line) || startswith(line, '#')) && continue
        if !startswith(rawline, ' ')
            endswith(line, ':') && (section = String(line[1:end-1]))
            continue
        end
        n = findfirst('#', line)
        n === nothing || (line = strip(line[1:n-1]))
        c = findfirst(':', line)
        c === nothing && continue
        key = strip(line[1:c-1]); val = strip(line[c+1:end])
        isempty(val) && continue
        get!(params, section, Dict{String,Float64}())[key] = parse(Float64, replace(val, r"[dD]" => "e"))
    end
    return params
end

# input.h (v1.0): Fortran-namelist-like "key = value ! comment" under "! ... PARAMETERS" headers
function read_input_h(path::AbstractString)
    params = Dict{String,Dict{String,Float64}}()
    section = ""
    for rawline in eachline(path)
        # Fortran compares the (blank-padded) line with the header exactly
        for (hdr, name) in SECTION_HEADERS
            if rstrip(rawline) == hdr
                section = name
            end
        end
        line = strip(rawline)
        (isempty(line) || startswith(line, '!') || section == "") && continue
        n = findfirst('!', line)
        n === nothing || (line = strip(line[1:n-1]))
        eq = findfirst('=', line)
        eq === nothing && continue
        key = strip(line[1:eq-1])
        val = replace(strip(line[eq+1:end]), r"[dD]" => "e")
        get!(params, section, Dict{String,Float64}())[key] = parse(Float64, val)
    end
    return params
end

function getparam(p, section, key)
    haskey(p, section) && haskey(p[section], key) ||
        error("parameter $key not defined for $section")
    return p[section][key]
end
