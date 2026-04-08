# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jl:percent,ipynb
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # Exploration: How Expert Systems Reason
#
# **PUBH 5106 -- AI in Health Applications | Module 3, Segment A**
#
# ## What This Notebook Is For
#
# This is a **pre-class exploration** -- a short, self-guided activity
# designed to build your intuition about rule-based AI before the lecture.
# There is nothing to submit. Experiment freely.
#
# ## What You Will Do
#
# In Module 2, you explored how LLMs process text as statistical patterns.
# Today you will meet a fundamentally different kind of AI: one that reasons
# from explicit rules written by human experts.
#
# In 1973, researchers at Stanford built **MYCIN** -- an AI system that
# recommended antibiotic therapy for blood infections. It had approximately
# 100 IF-THEN rules, and in a 1979 blind evaluation its recommendations
# were rated acceptable more often than those of the best human expert.
#
# In this notebook you will:
#
# 1. Load a clinical knowledge base from a YAML file
# 2. Watch an inference engine identify bacteria and recommend antibiotics
# 3. Trace its reasoning chain -- something you cannot do with an LLM
# 4. Add your own rules **by editing a file, not by changing any code**
#
# **Time:** ~25 minutes
#
# **Prerequisites:** None beyond basic programming

# %% [markdown]
# ---
# ## Setup

# %%
include("M3Utils.jl")
using .M3Utils

# %% [markdown]
# ---
# ## Part 1: Separating Knowledge from Code
#
# Bruce Buchanan -- one of MYCIN's creators -- made an important
# observation: **knowledge should be separated from the code that uses it.**
#
# The inference engine is in `M3Utils.jl`. It knows nothing about gram
# stains, organisms, or antibiotics. All of the clinical knowledge lives
# in `mycin_rules.yaml`.
#
# Double-click `mycin_rules.yaml` in the JupyterLab sidebar to open it.
# Each rule has:
#
# - A **name** and **text** description
# - **Premises**: conditions that must all be true
# - A **conclusion**: what the rule infers
# - A **certainty factor (CF)**: confidence from 0.0 to 1.0
#
# MYCIN used **two different CF scales** (and so do we):
#
# - **User-reported facts** (working memory): 0-10 integer scale.
#   When the system asks "how certain are you?", the clinician
#   enters a number from 0 (no idea) to 10 (certain).
# - **Rule CFs**: 0.0-1.0 scale. These are set by the knowledge
#   engineer and represent the rule's inherent confidence.
#
# The engine normalizes user CFs (÷10) before computing conclusions.

# %% [markdown]
# ### Load the Knowledge Base

# %%
RULES = load_rules("mycin_rules.yaml")
show_rules(RULES)

# %% [markdown]
# ---
# ## Part 2: A Patient Case
#
# MYCIN began each consultation by gathering facts about the patient and
# their culture results. These facts form the system's **working memory**
# -- everything it currently knows.

# %%
patient_1 = WorkingMemory(
    :gram_stain     => ("gram-positive", 10.0),
    :morphology     => ("coccus",        10.0),
    :growth_pattern => ("chains",         8.0),  # lab tech reports 8/10
)

show_facts(patient_1)

# %% [markdown]
# Notice that `growth_pattern` has a certainty factor of 8/10 -- the lab
# technician is fairly confident but not certain. This uncertainty will
# propagate through the reasoning chain.

# %% [markdown]
# ---
# ## Part 3: The Inference Engine
#
# The engine uses **forward chaining**: it scans all rules, fires any
# whose premises match, adds conclusions to working memory, and repeats
# until nothing new can be inferred.
#
# *Note: The actual MYCIN used backward chaining (goal-driven). We use
# forward chaining here because it is easier to follow step by step.
# The rules and certainty factor math are the same.*
#
# ### Certainty Factor Math
#
# User facts are on a **0-10 scale**. Rule CFs are on a **0.0-1.0 scale**.
# The engine normalizes user CFs by dividing by 10 before computing:
#
# > **conclusion CF = rule CF × min(normalized premise CFs)**
#
# The result is scaled back to 0-10 for display.
#
# When two rules reach the **same** conclusion, their CFs combine
# (on the 0-1 scale):
#
# > **combined CF = CF₁ + CF₂ × (1 - CF₁)**
#
# More evidence increases confidence, but the result never exceeds 10/10.

# %%
facts_1, trace_1 = run_engine(RULES, patient_1);

# %% [markdown]
# **Look at the output above.**
#
# - The engine fired two rules in sequence: first it identified the
#   organism (R1: streptococcus), then it recommended therapy
#   (R10: penicillin).
# - Notice how certainty **decreased** at each step: the growth pattern
#   was only 0.8 certain, the identification rule has CF 0.7, and the
#   therapy rule has CF 0.8.
# - This is **chained inference** -- each rule's conclusion becomes the
#   next rule's premise. Uncertainty propagates honestly through the chain.

# %% [markdown]
# ---
# ## Part 4: "WHY?" -- Tracing the Reasoning
#
# One of MYCIN's most important features was the ability to ask **WHY**
# at any point. The system would display the rule it was currently
# evaluating, showing the physician exactly why it reached a conclusion.
#
# Compare this to an LLM: if ChatGPT recommends penicillin, can you ask
# it to show you the specific rule it used?

# %%
println("Reasoning chain for the therapy recommendation:\n")
explain_why(:therapy, trace_1, patient_1)

# %% [markdown]
# The entire chain of reasoning is visible -- from initial lab findings
# through organism identification to therapy recommendation. Every step
# cites a specific rule. This is what **auditability** looks like.

# %% [markdown]
# ---
# ## Part 5: Change the Patient, Change the Reasoning
#
# What happens when the clinical findings are different? Patient 2 has
# gram-negative rods from a blood culture in a compromised host.

# %%
patient_2 = WorkingMemory(
    :gram_stain       => ("gram-negative", 10.0),
    :morphology       => ("rod",           10.0),
    :aerobicity       => ("aerobic",       10.0),
    :host_compromised => ("yes",            7.0),
)

show_facts(patient_2; label="Patient 2")
facts_2, trace_2 = run_engine(RULES, patient_2);

# %% [markdown]
# **Notice what happened:**
#
# - Two rules (**R3** and **R5**) both concluded pseudomonas, with different
#   certainty factors.
# - The system combined them using the MYCIN formula:
#   **combined CF = CF₁ + CF₂ × (1 - CF₁)**
# - This is MYCIN's **certainty factor combining function** -- a pragmatic
#   formula for accumulating evidence. It is not Bayesian probability (we
#   will explore that in Module 4), but it worked well in practice.

# %%
println("\nReasoning chain for Patient 2:\n")
explain_why(:therapy, trace_2, patient_2)

# %% [markdown]
# ---
# ## Part 6: Your Turn -- Experiment
#
# Try creating your own patient and running the engine. Some ideas:
#
# - What happens with gram-negative, rod, anaerobic? *(Hint: Rule R4)*
# - What if you are only 50% certain of the gram stain?
# - What if you provide facts that match no rule at all?
# - What happens with gram-positive, rod, anaerobic?
#   *(Rule R7 — but there is no therapy rule for clostridium!)*

# %%
my_patient = WorkingMemory(
    :gram_stain => ("gram-negative", 10.0),
    :morphology => ("rod",           10.0),
    :aerobicity => ("anaerobic",     10.0),
)

show_facts(my_patient; label="Your patient")
my_facts, my_trace = run_engine(RULES, my_patient);

# %%
if !isempty(my_trace)
    last_attr = my_trace[end].conclusion_attr
    println("\nReasoning chain for $last_attr:\n")
    explain_why(last_attr, my_trace, my_patient)
end

# %% [markdown]
# ---
# ## Part 7: Be the Knowledge Engineer
#
# MYCIN's knowledge came from human experts. When the system needed to
# handle a new case, a specialist added or modified rules -- **they
# edited the knowledge base, not the engine code.**
#
# We have prepared a second YAML file called `neisseria_rules.yaml`.
# Double-click `neisseria_rules.yaml` in the JupyterLab sidebar to open it
# -- it contains two new rules for identifying and treating Neisseria
# infections.

# %%
extended_rules = vcat(
    load_rules("mycin_rules.yaml"),
    load_rules("neisseria_rules.yaml"),
)

println("Extended knowledge base: $(length(extended_rules)) rules")
println("  Original: $(length(RULES)) rules from mycin_rules.yaml")
println("  Added:    $(length(extended_rules) - length(RULES)) rules from neisseria_rules.yaml\n")

patient_neisseria = WorkingMemory(
    :gram_stain => ("gram-negative", 10.0),
    :morphology => ("coccus",        10.0),
)

show_facts(patient_neisseria; label="Neisseria test patient")
facts_n, trace_n = run_engine(extended_rules, patient_neisseria);

# %%
println("\nReasoning chain:\n")
explain_why(:therapy, trace_n, patient_neisseria)

# %% [markdown]
# You added clinical knowledge by editing a YAML file. The inference
# engine didn't change. This is Buchanan's principle in action:
# **separate the knowledge from the code.**

# %% [markdown]
# ---
# ## Part 8: Where This Breaks Down
#
# **Strengths you observed:**
# - Every conclusion is traceable to specific rules
# - You can ask WHY and get a complete audit trail
# - The system cannot "hallucinate"
# - Adding knowledge means editing a YAML file, not retraining a model
#
# **Limitations to consider:**
# - What happens when facts match no rule? (Silent failure)
# - We have 12 rules. MYCIN had ~100. Who writes and maintains thousands?
#   This is the **knowledge acquisition bottleneck**.
# - Certainty factors are a heuristic, not rigorous probability.
#   *(Module 4 introduces Bayesian probability as an alternative.)*
# - The system cannot reason about what it does not know.

# %% [markdown]
# ---
# ## Reflection
#
# 1. A modern EHR alert that fires when you prescribe a drug to an allergic
#    patient is a production rule. How is it similar to what you built here?
#
# 2. Could you build a MYCIN-like system for your own area? How many
#    rules would you need? Where would you get them?
#
# 3. Compare this to Module 2's tokenization notebook. LLMs process text
#    as statistical patterns; this system reasons from explicit rules.
#    What does each do well? What does each lack?
#
# 4. The knowledge base is in a YAML file that any clinician can read.
#    Could you say the same about the weights of an LLM?
