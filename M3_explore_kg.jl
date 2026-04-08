# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl:percent,ipynb
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Julia 1.10
#     language: julia
#     name: julia-1.10
# ---

# %% [markdown]
# # Exploration: Knowledge Graphs for Lipid Management
#
# **PUBH 5106 -- AI in Health Applications | Module 3, Segment D**
#
# ## What This Notebook Is For
#
# This is a **pre-class exploration** -- a short, self-guided activity
# designed to build your intuition about knowledge graphs before the
# lecture. There is nothing to submit. Experiment freely.
#
# ## What You Will Do
#
# The statin guideline notebook encoded *what to do* (prescribe high-
# intensity statin for ASCVD). But **why** do statins work? What is the
# biological chain from drug to reduced heart attack risk?
#
# A **knowledge graph** (KG) encodes these relationships as a network
# of entities connected by labeled edges:
#
# > *atorvastatin* **inhibits** *HMG-CoA reductase* **→** *reduces*
# > *intracellular cholesterol* **→** *upregulates* *LDL receptor*
# > **→** *clears* *LDL-C from blood*
#
# This is the mechanism Brown & Goldstein discovered at UT Southwestern
# — represented as a traversable graph.
#
# In this notebook you will:
#
# 1. Load a biomedical knowledge graph from a YAML file
# 2. Query it: "What drugs lower LDL?" "Why does atorvastatin reduce MI risk?"
# 3. Trace multi-hop reasoning chains through the graph
# 4. Add your own knowledge and see the graph grow
# 5. Compare graph-based reasoning to what an LLM would produce
#
# **Time:** ~25 minutes
#
# **Prerequisites:** No graph theory background needed.

# %% [markdown]
# ---
# ## Setup

# %%
include("M3Utils.jl")
using .M3Utils

# %% [markdown]
# ---
# ## Part 1: Loading and Visualizing the Knowledge Graph
#
# Double-click `lipid_kg.yaml` in the JupyterLab sidebar to open it.
# Each entry is a **triple**: (subject, predicate, object) — the
# fundamental unit of a knowledge graph. Together, the triples form
# a network of biomedical facts.

# %%
kg = load_kg(["lipid_kg.yaml"])
show_kg_summary(kg)

# %% [markdown]
# ### The Graph at a Glance
#
# Below is a visualization of the entire knowledge graph. Nodes are
# biomedical entities (drugs, targets, diseases, genes). Edges are
# labeled relationships (inhibits, causes, increases risk of, etc.).
#
# Take a moment to orient yourself:
# - Find **atorvastatin** (blue) and trace its path to **MI** (red)
# - Find the **LDL receptor** (orange) — notice how many things connect to it
# - Find the two UTSW discoveries: the LDL receptor pathway (Brown &
#   Goldstein) and the PCSK9 pathway (Hobbs & Cohen)
#
# ![Lipid Management Knowledge Graph](lipid_kg_graph.png)

# %% [markdown]
# ---
# ## Part 2: Simple Queries
#
# A knowledge graph query asks: "What do we know about X?" or
# "What is related to Y by relationship Z?"

# %%
println("What do we know about atorvastatin?\n")
query_from(kg, "atorvastatin")

# %%
println("What inhibits HMG-CoA reductase?\n")
query_to(kg, "HMG-CoA reductase")

# %%
println("What causes atherosclerosis?\n")
query_to(kg, "atherosclerosis")

# %% [markdown]
# These are **one-hop queries** — they follow a single edge. But the
# real power of knowledge graphs comes from **multi-hop reasoning**:
# following chains of relationships to answer complex questions.

# %% [markdown]
# ---
# ## Part 3: Path Reasoning — "WHY Does This Drug Work?"
#
# The most important question a knowledge graph can answer is **why**.
# Why does atorvastatin reduce heart attack risk? The answer is not a
# single fact — it is a *chain* of biological relationships.

# %%
println("WHY does atorvastatin reduce LDL-C?\n")

paths = find_paths(kg, "atorvastatin", "LDL-C from blood")
if isempty(paths)
    println("  No path found.")
else
    for (i, path) in enumerate(paths)
        show_path(path; label="Path $i ($(length(path)) hops):")
        println()
    end
end

# %% [markdown]
# **Read the chain above.** This is Brown & Goldstein's discovery,
# represented as a graph traversal:
#
# 1. Atorvastatin **inhibits** HMG-CoA reductase
# 2. HMG-CoA reductase inhibition **reduces** intracellular cholesterol
# 3. Reduced intracellular cholesterol **upregulates** LDL receptor
# 4. LDL receptor **clears** LDL-C from blood
#
# Every step cites a source. The reasoning is inspectable and traceable.

# %%
println("WHY does atorvastatin reduce myocardial infarction risk?\n")

paths = find_paths(kg, "atorvastatin", "myocardial infarction"; max_depth=8)
if isempty(paths)
    println("  No path found.")
else
    # Show shortest path
    shortest = argmin(length, paths)
    show_path(shortest; label="Shortest path ($(length(shortest)) hops):")
    if length(paths) > 1
        println("\n  ($(length(paths)) total paths found)")
    end
end

# %% [markdown]
# The full causal chain from drug to disease outcome — every link
# backed by evidence. An LLM could state "statins reduce heart attack
# risk" but could not show you this structured chain with sources.

# %% [markdown]
# ---
# ## Part 4: Comparing Drug Mechanisms
#
# The graph can answer: "How do different lipid-lowering drugs work?"
# Each drug enters the graph at a different point but converges on the
# same outcome: lowering LDL-C.

# %%
drugs = ["atorvastatin", "evolocumab", "ezetimibe"]

for drug in drugs
    println("─"^50)
    println("$drug → LDL-C from blood:\n")
    local paths = find_paths(kg, drug, "LDL-C from blood")
    if isempty(paths)
        println("  No path found.\n")
    else
        shortest = argmin(length, paths)
        show_path(shortest; label="$(length(shortest)) hops:")
    end
    println()
end

# %% [markdown]
# **Three drugs, three different mechanisms, one shared outcome.**
#
# - **Atorvastatin**: inhibits synthesis → upregulates receptor → clears LDL
# - **Evolocumab**: inhibits PCSK9 → stops receptor degradation → more receptor → clears LDL
# - **Ezetimibe**: blocks intestinal absorption → less cholesterol entering blood
#
# The ACC/AHA guideline recommends adding ezetimibe or evolocumab when
# statins alone don't achieve the LDL target. The KG shows *why* this
# makes sense: they attack different parts of the pathway.

# %% [markdown]
# ---
# ## Part 5: The UTSW Story as a Graph
#
# UT Southwestern's contribution to lipid science spans two Nobel-level
# discoveries. Both are visible in the graph.

# %%
println("Brown & Goldstein's discovery (Nobel 1985):\n")
println("  The LDL receptor pathway:\n")
for entity in ["LDL receptor", "intracellular cholesterol",
               "reduced intracellular cholesterol"]
    query_from(kg, entity)
end

println("\n\nHobbs & Cohen's discovery (Dallas Heart Study):\n")
println("  The PCSK9 pathway:\n")
for entity in ["PCSK9", "PCSK9 loss-of-function mutation"]
    query_from(kg, entity)
end

# %% [markdown]
# Brown & Goldstein discovered the receptor. Hobbs & Cohen discovered
# what destroys it. Together, they gave us statins (protect the
# receptor by making more of it) and PCSK9 inhibitors (protect the
# receptor by blocking its destruction).

# %% [markdown]
# ---
# ## Part 6: Answering Clinical Questions
#
# Let's ask the graph structured clinical questions.

# %%
what_treats(kg, "LDL-C from blood")

# %%
what_treats(kg, "myocardial infarction")

# %% [markdown]
# The graph-based answer includes the **reasoning chain** — not just
# "atorvastatin treats MI" but *how* it does so through 5 intermediate
# biological steps. This is what KG-RAG (Soman et al. 2024) does at
# scale: it grounds LLM answers in structured knowledge paths.

# %% [markdown]
# ---
# ## Part 7: Add Your Own Knowledge
#
# Double-click `student_triples.yaml` in the JupyterLab sidebar to open it — it contains a starter example
# (bempedoic acid, a newer lipid-lowering drug). Run the cell below
# to load the extended graph and query it.

# %%
kg_extended = load_kg(["lipid_kg.yaml", "student_triples.yaml"])
println("Extended graph:")
println("  $(length(kg_extended.triples)) triples (was $(length(kg.triples)))")
println("  $(length(kg_extended.entities)) entities (was $(length(kg.entities)))")

println("\nNew drug: bempedoic acid\n")
query_from(kg_extended, "bempedoic acid")

println("\nDoes bempedoic acid connect to LDL-C?\n")
paths = find_paths(kg_extended, "bempedoic acid", "LDL-C from blood")
if !isempty(paths)
    shortest = argmin(length, paths)
    show_path(shortest; label="$(length(shortest)) hops:")
else
    println("  No path found — try adding more connecting triples!")
end

# %% [markdown]
# You extended the graph by editing a YAML file. The query engine
# found a new reasoning chain through the added triples. This is
# how knowledge graphs grow: new facts slot into the existing
# structure and immediately become queryable.

# %% [markdown]
# ---
# ## Part 8: Graph vs. Rules vs. LLM
#
# You have now seen three ways to represent knowledge about lipid
# management:
#
# | Approach | What It Does Well | What It Lacks |
# |----------|------------------|---------------|
# | **Production rules** (MYCIN/statin notebook) | Classifies patients, recommends therapy, auditable | No mechanism — says "prescribe statin" but not why |
# | **Knowledge graph** (this notebook) | Encodes mechanism, multi-hop reasoning, traceable paths | No uncertainty, no recommendations, no patient classification |
# | **LLM** (Module 2) | Can discuss anything fluently, handles novel phrasing | Cannot show structured reasoning chains, may hallucinate |
#
# **Neurosymbolic AI** combines these approaches:
# - Use the **KG** for structured biomedical knowledge
# - Use the **LLM** for natural language understanding
# - Use **rules/logic** for clinical decision making
#
# This is exactly what KG-RAG (Soman et al. 2024) does: the SPOKE
# knowledge graph (27 million nodes) provides structured facts, and
# the LLM translates natural language questions into graph queries.
# The result: grounded answers with traceable reasoning.

# %% [markdown]
# ---
# ## Reflection
#
# Before the lecture, think about:
#
# 1. The statin guideline says "high-intensity statin for ASCVD
#    patients." The knowledge graph shows *why* this works (the
#    mechanism from drug to reduced MI risk). Which representation
#    would a clinician prefer? Which would a regulator prefer?
#
# 2. Our graph has ~40 triples. SPOKE has 27 million. What challenges
#    arise when building and maintaining a graph at that scale?
#    (Hint: think about knowledge acquisition bottleneck from the
#    MYCIN notebook.)
#
# 3. An LLM can answer "why do statins work?" fluently. But can you
#    verify its answer? The graph provides a verifiable chain of
#    evidence. What does this mean for clinical AI trustworthiness?
#
# 4. Could you connect the statin guideline rules (from the previous
#    notebook) to this knowledge graph? What would a system look like
#    that could both classify a patient AND explain the biological
#    mechanism of the recommended therapy?
