"""
    M3Utils

Shared utilities for Module 3 pre-class notebooks. Provides three
reasoning engines — all loaded from YAML knowledge bases:

1. **Production rule engine** (MYCIN-style forward chaining with
   certainty factors) — used by the MYCIN, outbreak, and statin notebooks
2. **Guideline formalization** (YAML → deontic logic formulas via
   Gamen.jl) — used by the deontic and statin notebooks
3. **Knowledge graph engine** (triple store with BFS path reasoning)
   — used by the KG notebook

All domain knowledge lives in YAML files. This module contains only
the reasoning machinery.
"""
module M3Utils

using YAML

export
    # ── Inference Engine ──
    Premise, Rule, WorkingMemory, normalize_cf,
    load_rules, show_rules, rule_to_english,
    show_facts,
    TraceEntry, run_engine, explain_why,
    # ── Guideline Logic (requires Gamen.jl loaded separately) ──
    Guideline, load_guidelines, show_guidelines,
    # ── Knowledge Graph ──
    Triple, KnowledgeGraph,
    load_kg, show_kg_summary,
    query_from, query_to, query_by_predicate,
    find_paths, show_path, what_treats,
    # ── Patient Loading ──
    load_patients


# ═══════════════════════════════════════════════════════════════════
# PART 1: Production Rule Inference Engine
# ═══════════════════════════════════════════════════════════════════

"""A single premise: (attribute, operator, value)."""
struct Premise
    attribute::Symbol
    op::Symbol        # :is or :is_not
    value::String
end

"""A production rule with premises, conclusion, certainty factor, and description."""
struct Rule
    name::String
    premises::Vector{Premise}
    conclusion_attr::Symbol
    conclusion_val::String
    cf::Float64
    text::String
end

"""Working memory: maps attribute symbols to (value, certainty_factor) tuples.
User-reported facts use a 0-10 integer scale (as in the original MYCIN).
Inferred conclusions are also stored on the 0-10 scale."""
const WorkingMemory = Dict{Symbol, Tuple{String, Float64}}

"""Normalize a user-reported CF (0-10 scale) to 0-1 for computation."""
normalize_cf(cf::Float64) = cf / 10.0

"""Load production rules from a YAML file."""
function load_rules(path::String)
    data = YAML.load_file(path)
    rules = Rule[]
    for entry in data
        premises = [Premise(Symbol(p[1]), Symbol(p[2]), string(p[3]))
                    for p in entry["premises"]]
        conc = entry["conclusion"]
        push!(rules, Rule(
            entry["name"], premises, Symbol(conc[1]), string(conc[2]),
            Float64(entry["cf"]), entry["text"],
        ))
    end
    return rules
end

"""Format a rule as a readable IF-THEN string."""
function rule_to_english(rule::Rule)
    lines = String[]
    for (i, p) in enumerate(rule.premises)
        prefix = i == 1 ? "IF  " : "AND "
        push!(lines, "  $prefix $(p.attribute) $(p.op) $(p.value)")
    end
    push!(lines, "  THEN $(rule.conclusion_attr) = $(rule.conclusion_val)  [CF: $(rule.cf)]")
    join(lines, "\n")
end

"""Display all rules in a knowledge base."""
function show_rules(rules::Vector{Rule})
    println("Knowledge base: $(length(rules)) rules\n")
    for rule in rules
        println("── $(rule.name): $(rule.text)")
        println(rule_to_english(rule))
        println()
    end
end

"""Display working memory facts with certainty bar chart (0-10 scale)."""
function show_facts(facts::WorkingMemory; label="Patient facts")
    println("$label:")
    for (attr, (val, cf)) in sort(collect(facts), by=first)
        bar = repeat("█", round(Int, cf)) * repeat("░", 10 - round(Int, cf))
        println("  $(rpad(attr, 25)) = $(rpad(val, 20))  CF: $(round(Int, cf))/10  $bar")
    end
    println()
end

"""Record of a single rule firing (for WHY traces)."""
struct TraceEntry
    rule_name::String
    premises::Vector{Tuple{Symbol, String, Float64}}
    conclusion_attr::Symbol
    conclusion_val::String
    conclusion_cf::Float64
    combined_from::Union{Float64, Nothing}
end

"""
    run_engine(rules, initial_facts; verbose=true) -> (facts, trace)

Forward-chaining inference engine with MYCIN-style certainty factors.

User-reported facts are on a 0-10 scale. Rule CFs are on a 0.0-1.0 scale.
The engine normalizes user CFs (÷10) before computing:

    conclusion CF = rule CF × min(normalized premise CFs)

The result is stored back on the 0-10 scale.

When two rules reach the same conclusion (both on 0-1 scale internally):

    combined = CF₁ + CF₂ × (1 - CF₁)

then scaled back to 0-10.
"""
function run_engine(rules::Vector{Rule}, initial_facts::WorkingMemory;
                    verbose::Bool=true)
    facts = copy(initial_facts)
    trace = TraceEntry[]
    fired_rules = Set{String}()

    verbose && println("=" ^ 60)
    verbose && println("  INFERENCE ENGINE RUNNING")
    verbose && println("=" ^ 60)

    while true
        fired_any = false
        for rule in rules
            rule.name in fired_rules && continue

            all_match = true
            premise_cfs_normalized = Float64[]
            for p in rule.premises
                if !haskey(facts, p.attribute)
                    all_match = false; break
                end
                val, cf = facts[p.attribute]
                if p.op == :is && val != p.value
                    all_match = false; break
                elseif p.op == :is_not && val == p.value
                    all_match = false; break
                end
                push!(premise_cfs_normalized, normalize_cf(cf))
            end
            !all_match && continue

            min_norm_cf = minimum(premise_cfs_normalized)
            # Conclusion on 0-1 scale: rule CF × min(normalized premise CFs)
            new_cf_01 = round(rule.cf * min_norm_cf, digits=4)
            conc_attr = rule.conclusion_attr
            conc_val = rule.conclusion_val

            if haskey(facts, conc_attr)
                existing_val, existing_cf_10 = facts[conc_attr]
                if existing_val == conc_val
                    existing_cf_01 = normalize_cf(existing_cf_10)
                    # Combine on 0-1 scale
                    combined_01 = round(existing_cf_01 + new_cf_01 * (1 - existing_cf_01), digits=4)
                    combined_10 = round(combined_01 * 10, digits=1)
                    combined_10 <= existing_cf_10 + 0.001 && continue
                    if verbose
                        println("\n  RULE $(rule.name) FIRED (combining):")
                        for p in rule.premises
                            _, c = facts[p.attribute]
                            println("    $(p.attribute) $(p.op) $(p.value)  [CF: $(round(Int, c))/10]  MATCHED")
                        end
                        println("    THEN $conc_attr = $conc_val  [rule CF: $(rule.cf) × min premise: $(round(min_norm_cf, digits=2)) = $(round(new_cf_01, digits=2))]")
                        println("    COMBINING: $(round(existing_cf_01, digits=2)) + $(round(new_cf_01, digits=2)) × (1 - $(round(existing_cf_01, digits=2))) = $(round(combined_01, digits=2))  → $(round(combined_10, digits=1))/10")
                    end
                    facts[conc_attr] = (conc_val, combined_10)
                    push!(trace, TraceEntry(rule.name,
                        [(p.attribute, facts[p.attribute]...) for p in rule.premises],
                        conc_attr, conc_val, combined_10, existing_cf_10))
                    push!(fired_rules, rule.name)
                    fired_any = true
                end
            else
                new_cf_10 = round(new_cf_01 * 10, digits=1)
                if verbose
                    println("\n  RULE $(rule.name) FIRED:")
                    for p in rule.premises
                        _, c = facts[p.attribute]
                        println("    $(p.attribute) $(p.op) $(p.value)  [CF: $(round(Int, c))/10]  MATCHED")
                    end
                    println("    THEN $conc_attr = $conc_val  [rule CF: $(rule.cf) × min premise: $(round(min_norm_cf, digits=2)) = $(round(new_cf_01, digits=2))  → $(round(new_cf_10, digits=1))/10]")
                end
                facts[conc_attr] = (conc_val, new_cf_10)
                push!(trace, TraceEntry(rule.name,
                    [(p.attribute, facts[p.attribute]...) for p in rule.premises],
                    conc_attr, conc_val, new_cf_10, nothing))
                push!(fired_rules, rule.name)
                fired_any = true
            end
        end
        !fired_any && break
    end

    if verbose
        println("\n" * "=" ^ 60)
        println("  NO MORE RULES CAN FIRE")
        println("=" ^ 60)
        new_facts = Dict(k => v for (k, v) in facts if !haskey(initial_facts, k))
        if !isempty(new_facts)
            println("\n  CONCLUSIONS:")
            for (attr, (val, cf)) in sort(collect(new_facts), by=x -> -x[2][2])
                bar = repeat("█", round(Int, cf)) * repeat("░", 10 - round(Int, cf))
                println("    $(rpad(attr, 15)) = $(rpad(val, 30))  [CF: $(round(cf, digits=1))/10]  $bar")
            end
        else
            println("\n  No conclusions could be drawn.")
        end
    end
    return facts, trace
end

"""
    explain_why(attribute, trace, initial_facts; indent=0)

Recursively trace the reasoning chain for a conclusion, displaying
the full audit trail from initial facts through intermediate rules.
"""
function explain_why(attribute::Symbol, trace::Vector{TraceEntry},
                     initial_facts::WorkingMemory; indent::Int=0,
                     fact_label::String="INITIAL FACT")
    prefix = "  " ^ indent
    entry = nothing
    for t in trace
        t.conclusion_attr == attribute && (entry = t)
    end
    if entry === nothing
        if haskey(initial_facts, attribute)
            val, cf = initial_facts[attribute]
            println("$(prefix)$(attribute) = $val [CF: $(round(cf, digits=2))] — $fact_label")
        else
            println("$(prefix)$(attribute) — NOT KNOWN")
        end
        return
    end
    println("$(prefix)WHY $(entry.conclusion_attr) = $(entry.conclusion_val) [CF: $(round(entry.conclusion_cf, digits=1))/10]?")
    println("$(prefix)  BECAUSE Rule $(entry.rule_name):")
    for (attr, val, cf) in entry.premises
        if haskey(initial_facts, attr)
            println("$(prefix)    $attr = $val [CF: $(round(Int, cf))/10] — $fact_label")
        else
            explain_why(attr, trace, initial_facts; indent=indent + 2, fact_label=fact_label)
        end
    end
end


# ═══════════════════════════════════════════════════════════════════
# PART 2: Guideline Formalization (Deontic Logic)
# ═══════════════════════════════════════════════════════════════════
#
# These functions require Gamen.jl to be loaded in the calling notebook.
# They use eval() to reference Gamen types (Box, Diamond, etc.) so that
# this module does not itself depend on Gamen.

"""A clinical guideline with its English text, deontic classification, and formal logic formula."""
struct Guideline
    id::String
    text::String
    term::String
    type::String
    formula::Any       # Gamen.Formula — typed as Any to avoid hard dependency
    atoms::Vector{Symbol}
end

"""
    load_guidelines(path; logic_module=nothing)

Load guidelines from a YAML file and construct deontic logic formulas.
Requires Gamen.jl to be available — pass the module as `logic_module`
or ensure it is loaded in Main.
"""
function load_guidelines(path::String; logic_module::Module=Main)
    data = YAML.load_file(path)
    guidelines = Guideline[]

    Atom = getfield(logic_module, :Atom)
    Box = getfield(logic_module, :Box)
    Diamond = getfield(logic_module, :Diamond)
    Not = getfield(logic_module, :Not)
    Implies = getfield(logic_module, :Implies)

    for entry in data
        f = entry["formula"]
        atom = Atom(Symbol(f["atom"]))
        atoms = [Symbol(f["atom"])]

        if f["op"] == "box"
            core = Box(atom)
        elseif f["op"] == "diamond"
            core = Diamond(atom)
        elseif f["op"] == "box_not"
            core = Box(Not(atom))
        else
            error("Unknown operator: $(f["op"])")
        end

        if haskey(f, "conditional")
            cond_atom = Atom(Symbol(f["conditional"]))
            push!(atoms, Symbol(f["conditional"]))
            formula = Implies(cond_atom, core)
        else
            formula = core
        end

        push!(guidelines, Guideline(
            entry["id"], entry["text"], entry["term"],
            entry["type"], formula, atoms,
        ))
    end
    return guidelines
end

"""Display guidelines with their deontic formalizations."""
function show_guidelines(guidelines::Vector{Guideline})
    println("Loaded $(length(guidelines)) guidelines:\n")
    for g in guidelines
        println("$(g.id): \"$(g.text)\"")
        println("   Term: $(g.term) → $(g.type)")
        println("   Formal: $(g.formula)")
        println()
    end
end


# ═══════════════════════════════════════════════════════════════════
# PART 3: Knowledge Graph Engine
# ═══════════════════════════════════════════════════════════════════

"""A subject-predicate-object triple with provenance."""
struct Triple
    subject::String
    predicate::String
    object::String
    source::String
end

"""A knowledge graph with indexed triple store for fast lookup."""
struct KnowledgeGraph
    triples::Vector{Triple}
    by_subject::Dict{String, Vector{Triple}}
    by_object::Dict{String, Vector{Triple}}
    by_predicate::Dict{String, Vector{Triple}}
    entities::Set{String}
end

"""Load a knowledge graph from one or more YAML files."""
function load_kg(paths::Vector{String})
    triples = Triple[]
    for path in paths
        data = YAML.load_file(path)
        for entry in data
            push!(triples, Triple(
                entry["subject"], entry["predicate"],
                entry["object"], get(entry, "source", ""),
            ))
        end
    end
    by_subject = Dict{String, Vector{Triple}}()
    by_object = Dict{String, Vector{Triple}}()
    by_predicate = Dict{String, Vector{Triple}}()
    entities = Set{String}()
    for t in triples
        push!(get!(by_subject, t.subject, Triple[]), t)
        push!(get!(by_object, t.object, Triple[]), t)
        push!(get!(by_predicate, t.predicate, Triple[]), t)
        push!(entities, t.subject)
        push!(entities, t.object)
    end
    return KnowledgeGraph(triples, by_subject, by_object, by_predicate, entities)
end

"""Display a summary of the knowledge graph."""
function show_kg_summary(kg::KnowledgeGraph)
    predicates = Dict{String,Int}()
    for t in kg.triples
        predicates[t.predicate] = get(predicates, t.predicate, 0) + 1
    end
    println("Knowledge graph:")
    println("  $(length(kg.triples)) triples")
    println("  $(length(kg.entities)) entities")
    println("  $(length(predicates)) relationship types:")
    for (pred, count) in sort(collect(predicates), by=x -> -x[2])
        println("    $(rpad(pred, 25)) ($count)")
    end
end

"""Query: what does this entity connect to (as subject)?"""
function query_from(kg::KnowledgeGraph, entity::String)
    triples = get(kg.by_subject, entity, Triple[])
    if isempty(triples)
        println("  No outgoing relationships from '$entity'")
        return
    end
    println("  $entity:")
    for t in triples
        println("    ── $(t.predicate) → $(t.object)")
    end
end

"""Query: what connects to this entity (as object)?"""
function query_to(kg::KnowledgeGraph, entity::String)
    triples = get(kg.by_object, entity, Triple[])
    if isempty(triples)
        println("  No incoming relationships to '$entity'")
        return
    end
    println("  → $entity:")
    for t in triples
        println("    $(t.subject) ── $(t.predicate) →")
    end
end

"""Query: find all entities related by a specific predicate."""
function query_by_predicate(kg::KnowledgeGraph, predicate::String)
    triples = get(kg.by_predicate, predicate, Triple[])
    if isempty(triples)
        println("  No relationships of type '$predicate'")
        return
    end
    println("  Relationships: '$predicate'")
    for t in triples
        println("    $(t.subject) → $(t.object)")
    end
end

"""Find all paths from start to goal via BFS (max depth)."""
function find_paths(kg::KnowledgeGraph, start::String, goal::String;
                    max_depth::Int=6)
    queue = [(start, Triple[])]
    found_paths = Vector{Vector{Triple}}()
    while !isempty(queue)
        current, path = popfirst!(queue)
        if current == goal && !isempty(path)
            push!(found_paths, copy(path))
            continue
        end
        length(path) >= max_depth && continue
        for t in get(kg.by_subject, current, Triple[])
            if t.object ∉ [tr.subject for tr in path] && t.object != start
                push!(queue, (t.object, vcat(path, [t])))
            end
        end
    end
    return found_paths
end

"""Display a reasoning chain as a readable path with sources."""
function show_path(path::Vector{Triple}; label::String="")
    !isempty(label) && println("  $label")
    for (i, t) in enumerate(path)
        i == 1 && print("  $(t.subject)")
        println()
        println("    │ $(t.predicate)")
        println("    ▼")
        print("  $(t.object)")
    end
    println()
    println()
    sources = unique([t.source for t in path if !isempty(t.source)])
    !isempty(sources) && println("  Sources: $(join(sources, ", "))")
end

"""Find all drugs that connect to a given condition via graph paths."""
function what_treats(kg::KnowledgeGraph, condition::String; max_depth::Int=8)
    drug_triples = [t for t in kg.triples if t.predicate == "is a"]
    drug_names = [t.subject for t in drug_triples]
    println("What can address '$condition'?\n")
    found_any = false
    for drug in drug_names
        paths = find_paths(kg, drug, condition; max_depth=max_depth)
        if !isempty(paths)
            found_any = true
            shortest = argmin(length, paths)
            chain = join(["$(t.subject) ─$(t.predicate)→" for t in shortest], " ")
            chain *= " $(shortest[end].object)"
            drug_class = first(t.object for t in drug_triples if t.subject == drug)
            println("  $(rpad(drug, 20)) ($drug_class)")
            println("    Chain: $chain")
            println()
        end
    end
    !found_any && println("  No drugs found that connect to '$condition'")
end


# ═══════════════════════════════════════════════════════════════════
# PART 4: Patient Loading
# ═══════════════════════════════════════════════════════════════════

"""Load test patients from a YAML file."""
function load_patients(path::String)
    data = YAML.load_file(path)
    patients = []
    for entry in data
        facts = WorkingMemory()
        for (key, val_cf) in entry["facts"]
            facts[Symbol(key)] = (string(val_cf[1]), Float64(val_cf[2]))
        end
        push!(patients, (name=entry["name"], description=entry["description"], facts=facts))
    end
    return patients
end

end # module
