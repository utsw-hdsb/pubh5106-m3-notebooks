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
# # Exploration: Rule-Based Reasoning for Outbreak Investigation
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
# When a cluster of foodborne illness cases arrives at a local health
# department, epidemiologists follow a structured reasoning process:
# *What are the symptoms? What did they eat? How quickly did they get
# sick? Does the pattern point to a specific pathogen?*
#
# This process is remarkably similar to how **MYCIN** -- a 1973 AI system
# for antibiotic therapy -- worked. MYCIN used IF-THEN rules written by
# infectious disease specialists to identify bacteria and recommend
# treatment. The same architecture applies to public health investigation.
#
# In this notebook you will:
#
# 1. Load an outbreak investigation knowledge base from a YAML file
# 2. Watch an inference engine identify likely pathogens from case reports
# 3. Trace its reasoning chain -- something you cannot do with an LLM
# 4. Add your own rules **by editing a file, not by changing any code**
#
# **Time:** ~25 minutes
#
# **Prerequisites:** No clinical knowledge required. The scenario uses
# everyday concepts (food, symptoms, timing).

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
# The inference engine below is identical to the one used in the MYCIN
# notebook. It knows nothing about pathogens, symptoms, or food. All of
# the public health knowledge lives in `outbreak_rules.yaml`.
#
# Double-click `outbreak_rules.yaml` in the JupyterLab sidebar to open it.
# Each rule has:
#
# - A **name** and **text** description
# - **Premises**: conditions that must all be true
# - A **conclusion**: what the rule infers
# - A **certainty factor (CF)**: confidence from 0.0 to 1.0

# %% [markdown]
# ### Load the Knowledge Base

# %%
RULES = load_rules("outbreak_rules.yaml")
show_rules(RULES)

# %% [markdown]
# The code above is generic -- it loaded rules about foodborne pathogens,
# but it could just as easily load rules about antibiotic therapy (the
# MYCIN notebook) or any other domain. The engine doesn't care. The
# knowledge is in the file.

# %% [markdown]
# ---
# ## Part 2: An Outbreak Scenario
#
# **Scenario:** The county health department receives reports of 12 people
# who attended a church potluck on Saturday. By Monday, 8 of them are
# experiencing watery diarrhea. Interviews reveal that all affected
# individuals ate the potato salad, which contained eggs and was left
# unrefrigerated for several hours. Onset was 12-72 hours after the meal.
# Several patients have fever.
#
# Let's encode what we know.

# %%
potluck_outbreak = WorkingMemory(
    :primary_symptom  => ("watery_diarrhea", 10.0),
    :incubation_hours => ("12-72",           10.0),
    :food_exposure    => ("eggs",             8.0),   # self-reported: 8/10
    :fever_present    => ("yes",              9.0),   # most but not all patients
    :case_count       => ("multiple",        10.0),
    :single_source    => ("yes",              9.0),
)

show_facts(potluck_outbreak; label="Potluck outbreak")

# %% [markdown]
# Notice the uncertainty: food exposure is self-reported (8/10) and
# not every patient has fever (9/10). This is realistic -- outbreak
# data is always imperfect.

# %% [markdown]
# ---
# ## Part 3: The Inference Engine
#
# The engine uses **forward chaining**: it scans all rules, fires any
# whose premises match, adds conclusions to working memory, and repeats
# until nothing new can be inferred.
#
# ### Certainty Factor Math
#
# > **conclusion CF = rule CF × minimum CF among the premises**
#
# > **combined CF = CF₁ + CF₂ × (1 - CF₁)**  *(when two rules support the same conclusion)*

# %%
facts_1, trace_1 = run_engine(RULES, potluck_outbreak);

# %% [markdown]
# **Read the output above.** The engine:
#
# 1. Identified the likely pathogen (Salmonella) from symptoms, timing,
#    and food exposure
# 2. Recommended the public health response (traceback and inspection)
# 3. Also triggered the facility closure rule (multiple cases, single source)
#
# Notice how the self-reported food exposure (CF 0.8) reduced the
# certainty of the pathogen identification, which in turn reduced the
# certainty of the recommended response. Uncertainty propagates honestly.

# %% [markdown]
# ---
# ## Part 4: "WHY?" -- Tracing the Reasoning
#
# Just as MYCIN could explain its reasoning to physicians, our engine
# can explain its reasoning to epidemiologists. This matters: when you
# report to the state health department that a pathogen is likely
# Salmonella, you need to show your evidence chain.

# %%
println("Why is Salmonella the likely pathogen?\n")
explain_why(:pathogen, trace_1, potluck_outbreak)

# %%
println("\nWhy is traceback recommended?\n")
explain_why(:response, trace_1, potluck_outbreak)

# %% [markdown]
# Every conclusion traces back to specific reported facts. Compare this
# to an LLM: if you asked ChatGPT "what pathogen caused this outbreak?",
# could you trace its answer back to the specific evidence it used?

# %% [markdown]
# ---
# ## Part 5: A Different Outbreak
#
# **Scenario:** A cruise ship reports 40 passengers with sudden-onset
# vomiting and watery diarrhea within 24-48 hours. Crew members are
# also becoming ill. No single food item is implicated -- the spread
# appears to be person-to-person.

# %%
cruise_outbreak = WorkingMemory(
    :primary_symptom   => ("watery_diarrhea",  10.0),
    :person_to_person  => ("yes",               9.0),
    :setting           => ("institutional",    10.0),
    :case_count        => ("multiple",         10.0),
    :single_source     => ("yes",               8.0),
)

show_facts(cruise_outbreak; label="Cruise ship outbreak")
facts_2, trace_2 = run_engine(RULES, cruise_outbreak);

# %% [markdown]
# **Two rules** (R5 and R9) both pointed to Norovirus, and the system
# combined their certainties. Person-to-person spread is a strong
# signal (CF 0.8), and the institutional setting adds corroborating
# evidence (CF 0.6). Combined: even more confident.

# %%
println("\nWhy Norovirus?\n")
explain_why(:pathogen, trace_2, cruise_outbreak)

# %% [markdown]
# ---
# ## Part 6: Your Turn -- Experiment
#
# Try creating your own outbreak scenario. Some ideas:
#
# - A cluster of bloody diarrhea after a cookout with hamburgers
# - Cases of neurological symptoms after eating home-canned salsa
# - Children at a daycare with diarrhea and vomiting
# - What happens if you provide facts that match no rule?

# %%
my_outbreak = WorkingMemory(
    :primary_symptom  => ("bloody_diarrhea", 10.0),
    :incubation_hours => ("24-72",           10.0),
    :food_exposure    => ("ground_beef",      9.0),
    :case_count       => ("multiple",        10.0),
    :single_source    => ("yes",             10.0),
)

show_facts(my_outbreak; label="Your outbreak")
my_facts, my_trace = run_engine(RULES, my_outbreak);

# %%
if !isempty(my_trace)
    println("\nReasoning chain:\n")
    # Explain the response if one was reached, otherwise the pathogen
    if haskey(my_facts, :response)
        explain_why(:response, my_trace, my_outbreak)
    elseif haskey(my_facts, :pathogen)
        explain_why(:pathogen, my_trace, my_outbreak)
    end
end

# %% [markdown]
# ---
# ## Part 7: Be the Knowledge Engineer
#
# The state epidemiologist wants the system to handle **Listeria**
# outbreaks. Listeria is associated with:
#
# - Deli meats, soft cheeses, unpasteurized dairy
# - Long incubation period (1-4 weeks, unlike most foodborne pathogens)
# - Fever and muscle aches (not primarily diarrhea)
# - Especially dangerous for pregnant women and elderly
# - Response: immediate product recall + consumer advisory
#
# We have prepared `student_outbreak_rules.yaml` with two Listeria rules.
# Double-click `student_outbreak_rules.yaml` in the JupyterLab sidebar
# to open it. Then run the cell below -- the
# engine loads both files and tests the combined knowledge base.

# %%
extended_rules = vcat(
    load_rules("outbreak_rules.yaml"),
    load_rules("student_outbreak_rules.yaml"),
)

println("Extended knowledge base: $(length(extended_rules)) rules")
println("  Original: $(length(RULES)) rules from outbreak_rules.yaml")
println("  Added:    $(length(extended_rules) - length(RULES)) rules from student_outbreak_rules.yaml")
println()

# A Listeria-compatible scenario
listeria_outbreak = WorkingMemory(
    :primary_symptom  => ("fever_and_myalgia", 10.0),
    :incubation_hours => ("168-672",            7.0),  # estimated from interviews
    :food_exposure    => ("deli_meat",          8.0),
)

show_facts(listeria_outbreak; label="Suspected Listeria outbreak")
facts_l, trace_l = run_engine(extended_rules, listeria_outbreak);

# %%
println("\nReasoning chain:\n")
explain_why(:response, trace_l, listeria_outbreak)

# %% [markdown]
# You extended the system's capabilities by adding two rules to a YAML
# file. The engine code is unchanged. This is the same principle that
# made MYCIN extensible: **separate the knowledge from the code.**
#
# Try modifying `student_outbreak_rules.yaml` to add rules for another
# pathogen -- perhaps Hepatitis A (associated with raw shellfish,
# incubation 15-50 days) or Shigella (associated with daycares,
# person-to-person spread, bloody diarrhea).

# %% [markdown]
# ---
# ## Part 8: Where This Breaks Down
#
# **Strengths you observed:**
#
# - Every conclusion traces back to specific evidence
# - You can ask WHY and get an audit trail for the state report
# - The system cannot "hallucinate" a pathogen -- it only concludes
#   what the rules support
# - Adding knowledge means editing a YAML file, not retraining a model
# - The rules are readable by any epidemiologist
#
# **Limitations to consider:**
#
# - Real outbreaks are messy: mixed symptoms, multiple exposures,
#   uncertain timelines. What happens when the data doesn't cleanly
#   match any rule?
# - Our knowledge base has 17 rules. The CDC's foodborne outbreak
#   database covers dozens of pathogens with overlapping presentations.
#   Who writes and maintains all those rules?
# - Certainty factors are a practical heuristic. In Module 4, you will
#   see how Bayesian probability provides a more principled framework
#   for reasoning under uncertainty.
# - The system has no memory: it cannot learn from past outbreaks.
#   Every investigation starts from the same fixed rule set.

# %% [markdown]
# ---
# ## Reflection
#
# Before the lecture, think about:
#
# 1. When a health department investigates an outbreak, epidemiologists
#    follow a structured process (hypothesis generation, case definition,
#    source identification). How is this similar to the rule engine?
#    How is it different?
#
# 2. Could you build a rule-based system for a public health problem
#    you care about? (Syndromic surveillance? Contact tracing? Vaccine
#    eligibility?) How many rules would you need?
#
# 3. Compare this to an LLM: if you described the potluck outbreak to
#    ChatGPT, it would probably say "Salmonella" too. But could you
#    trust that answer in a regulatory report? What would you need from
#    the AI to trust it?
#
# 4. The knowledge base is in a YAML file that any epidemiologist can
#    read and modify. What are the implications of this transparency
#    for public health practice?
