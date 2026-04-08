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
# # Case Study: Formalizing the ACC/AHA Statin Guidelines
#
# **PUBH 5106 -- AI in Health Applications | Module 3**
#
# ## What This Notebook Is For
#
# This notebook brings together everything from Module 3's pre-class
# material: **production rules** (Segment A), **deontic logic**
# (Segment B), and the principle of **separating knowledge from code**
# (Buchanan).
#
# We apply all three to a real clinical guideline: the **2018 ACC/AHA
# Cholesterol Management Guideline** (Grundy et al. 2018), which
# recommends statin therapy based on cardiovascular risk.
#
# ## Why This Guideline?
#
# UT Southwestern has a unique connection to statin science. The Nobel
# Prize-winning research by Michael Brown and Joseph Goldstein at UTSW
# in the 1970s explained the mechanism by which statins work. The 2018
# guideline builds on decades of evidence that followed from their
# discovery.
#
# The guideline is also a perfect example of **mixed logic**:
#
# - **Propositional logic** classifies patients into risk groups
#   (IF ASCVD AND age 40-75 THEN high-risk group)
# - **Deontic logic** recommends therapy at different obligation levels
#   (Class I = "is recommended," Class IIa = "is reasonable,"
#    Class IIb = "may be considered")
#
# The ACC/AHA Class of Recommendation system **is** a deontic scale.
#
# ## What You Will Do
#
# 1. Load the guideline rules and test patients from YAML files
# 2. Watch the inference engine classify patients and recommend therapy
# 3. See how the deontic strength (Class I vs IIa vs IIb) maps to
#    different EHR implementation decisions
# 4. Use the Gamen.jl theorem prover to verify guideline consistency
#
# **Time:** ~30 minutes

# %% [markdown]
# ---
# ## Setup

# %%
using Gamen
include("M3Utils.jl")
using .M3Utils

# %% [markdown]
# ---
# ## Part 1: The Guideline as a Knowledge Base
#
# Double-click `statin_rules.yaml` in the JupyterLab sidebar to open it. The rules are organized in
# two layers, following the same architecture as MYCIN:
#
# **Layer 1 — Patient classification** (propositional logic):
# Rules C1-C10 classify patients into risk groups based on clinical
# facts (ASCVD history, LDL level, diabetes, 10-year risk score,
# risk enhancers).
#
# **Layer 2 — Therapy recommendation** (deontic logic):
# Rules T1-T9 recommend statin therapy based on the risk group.
# Each rule's certainty factor encodes the ACC/AHA Class of
# Recommendation:
#
# | COR | Meaning | Deontic | CF |
# |-----|---------|---------|-----|
# | Class I | "Is recommended" | Obligation | 0.95 |
# | Class IIa | "Is reasonable" | Strong should | 0.80 |
# | Class IIb | "May be considered" | Permission | 0.60 |

# %%
RULES = load_rules("statin_rules.yaml")
println("Loaded $(length(RULES)) guideline rules:")
println("  Classification rules (C): $(count(r -> startswith(r.name, "C"), RULES))")
println("  Therapy rules (T):        $(count(r -> startswith(r.name, "T"), RULES))")

# %% [markdown]
# ---
# ## Part 2: Test Patients
#
# Double-click `test_patients.yaml` in the JupyterLab sidebar to open it. Each patient has a
# clinical description and a set of facts that the engine will use.

# %%
patients = load_patients("test_patients.yaml")
println("Loaded $(length(patients)) test patients:\n")
for p in patients
    println("  $(p.name): $(p.description)")
end

# %% [markdown]
# ---
# ## Part 3: Run the Guideline on Each Patient

# %%
results = []

for p in patients
    println("\n" * "="^70)
    println("  $(p.name)")
    println("  $(p.description)")
    println("="^70)

    facts, trace = run_engine(RULES, p.facts)

    # Extract conclusions
    risk_group = get(facts, :risk_group, ("none", 0.0))
    statin_rec = get(facts, :statin_recommendation, ("none", 0.0))

    println("\n  RESULT:")
    println("    Risk group:    $(risk_group[1]) [CF: $(round(risk_group[2], digits=2))]")
    println("    Statin rec:    $(statin_rec[1]) [CF: $(round(statin_rec[2], digits=2))]")

    # Map CF to COR
    cor = if statin_rec[2] >= 0.90
        "Class I (is recommended → OBLIGATION)"
    elseif statin_rec[2] >= 0.70
        "Class IIa (is reasonable → STRONG SHOULD)"
    elseif statin_rec[2] >= 0.50
        "Class IIb (may be considered → PERMISSION)"
    elseif statin_rec[1] != "none"
        "Weak (may be considered → PERMISSION)"
    else
        "No recommendation"
    end
    println("    Deontic level: $cor")

    push!(results, (patient=p, facts=facts, trace=trace,
                    risk_group=risk_group, statin_rec=statin_rec, cor=cor))
end

# %% [markdown]
# ---
# ## Part 4: Summary Table

# %%
let
    println("\n" * rpad("Patient", 18) * rpad("Risk Group", 30) *
            rpad("Statin", 25) * rpad("CF", 8) * "COR Level")
    println("-"^110)
    for r in results
        println(rpad(r.patient.name, 18) *
                rpad(r.risk_group[1], 30) *
                rpad(r.statin_rec[1], 25) *
                rpad(string(round(r.statin_rec[2], digits=2)), 8) *
                r.cor)
    end
end

# %% [markdown]
# **Read the table.** Notice:
#
# - **Mr. Johnson** (prior MI): Class I — high-intensity statin is an
#   **obligation**. An EHR should generate a hard stop if not prescribed.
# - **Ms. Garcia** (diabetes + risk enhancers): Gets two classifications —
#   diabetes primary (Class I) and high-risk diabetes (Class IIa). The
#   engine combines evidence.
# - **Mr. Patel** (LDL 245): Class I — severe hypercholesterolemia
#   triggers high-intensity statin without any risk calculation.
# - **Ms. Chen** (intermediate risk + enhancers): Class IIa — "is
#   reasonable." The EHR should strongly suggest but not block.
# - **Mr. Williams** (borderline, no enhancers): **No recommendation
#   fires.** The guideline says nothing for borderline risk without
#   enhancers — a deliberate gap that requires clinical judgment.

# %% [markdown]
# ---
# ## Part 5: Trace the Reasoning
#
# Let's trace why Ms. Garcia gets her recommendation.

# %%
garcia = results[2]
println("WHY does Ms. Garcia get her statin recommendation?\n")
explain_why(:statin_recommendation, garcia.trace, garcia.patient.facts)

# %%
println("\nWHY is she classified in her risk group?\n")
explain_why(:risk_group, garcia.trace, garcia.patient.facts)

# %% [markdown]
# Every step in the reasoning chain is visible — from clinical facts
# through risk classification to therapy recommendation. This is what
# an EHR audit trail should look like.

# %% [markdown]
# ---
# ## Part 6: Deontic Verification with Gamen.jl
#
# The inference engine classified patients and recommended therapy.
# But are the guideline's deontic recommendations **internally
# consistent**? Can we formally prove that no two recommendations
# contradict each other?
#
# We use the Gamen.jl theorem prover to check.

# %%
# Define atoms for the therapy recommendations
high_intensity = Atom(:high_intensity_statin)
moderate_intensity = Atom(:moderate_intensity_statin)
no_statin = Atom(:no_statin)

# The guideline makes these obligations for different patient groups:
# Class I ASCVD: O(high_intensity)
# Class I diabetes: O(moderate_intensity)
# Class IIb borderline: P(moderate_intensity)  [permission, not obligation]

# Are these three consistent? (Different patient groups, so yes)
println("Consistency checks:\n")

# Check 1: Obligation + permission of same therapy
# O(moderate) and P(moderate) — consistent (obligation implies permission)
check1 = tableau_consistent(TABLEAU_KD,
    Formula[Box(moderate_intensity), Diamond(moderate_intensity)])
if check1
    println("  O(moderate) + P(moderate): consistent")
    println("    Obligation implies permission — no contradiction.")
else
    println("  O(moderate) + P(moderate): INCONSISTENT")
    println("    Unexpected: obligation and permission conflict!")
end

# Check 2: Obligation of different intensities for same patient?
# O(high) and O(moderate) — is this consistent?
check2 = tableau_consistent(TABLEAU_KD,
    Formula[Box(high_intensity), Box(moderate_intensity)])
if check2
    println("  O(high) + O(moderate):     consistent")
    println("    Both can be obligatory — no contradiction.")
else
    println("  O(high) + O(moderate):     INCONSISTENT")
    println("    These two obligations cannot coexist!")
end

# Check 3: O(statin) and O(no_statin) — genuine conflict
check3 = tableau_consistent(TABLEAU_KD,
    Formula[Box(high_intensity), Box(Not(high_intensity))])
if check3
    println("  O(high) + O(¬high):        consistent")
    println("    Surprisingly, no contradiction detected.")
else
    println("  O(high) + O(¬high):        INCONSISTENT")
    println("    A real guideline contradiction — cannot oblige both.")
end

# %% [markdown]
# ### The D Axiom in Clinical Context
#
# The D axiom (□p → ◇p) ensures that every obligation is satisfiable.
# In clinical terms: **if the guideline says you must prescribe a statin,
# it must be possible to prescribe a statin.**
#
# When might this be violated?

# %%
# Scenario: Patient has a statin allergy
statin_allergy = Atom(:statin_tolerated)

# Guideline says: O(prescribe_statin) — must prescribe
# Reality says: patient cannot tolerate any statin
# D axiom says: O(p) → P(p) — obligation implies possibility

# If both hold, we need: the obligation is defeatable
println("The statin allergy problem:\n")
println("  Guideline:  O(statin)")
println("  Patient:    cannot tolerate statin")
println("  D axiom:    O(statin) → P(statin)")
println()
println("  The D axiom tells us something important:")
println("  if a patient truly cannot tolerate any statin,")
println("  then the obligation O(statin) cannot hold for them.")
println("  The guideline must have an exception mechanism —")
println("  which is exactly what 'maximally tolerated statin")
println("  therapy' provides. The guideline authors anticipated")
println("  this deontic constraint.")

# %% [markdown]
# ---
# ## Part 7: The COR-Deontic Mapping in Practice
#
# The ACC/AHA Class of Recommendation system maps directly to deontic
# operators. This has real consequences for how EHR systems should
# implement each recommendation:

# %%
println("""
┌─────────┬──────────────────────┬───────────┬────────────────────────────────┐
│ COR     │ Guideline Language   │ Deontic   │ EHR Implementation             │
├─────────┼──────────────────────┼───────────┼────────────────────────────────┤
│ Class I │ "is recommended"     │ O(therapy)│ Hard stop / mandatory order    │
│         │ "should be performed"│           │ set / quality metric           │
├─────────┼──────────────────────┼───────────┼────────────────────────────────┤
│ Class   │ "is reasonable"      │ Should    │ Prominent suggestion / BPA     │
│ IIa     │ "can be useful"      │           │ with documented override       │
├─────────┼──────────────────────┼───────────┼────────────────────────────────┤
│ Class   │ "may be considered"  │ P(therapy)│ Informational alert / shared   │
│ IIb     │ "usefulness unknown" │           │ decision-making prompt         │
├─────────┼──────────────────────┼───────────┼────────────────────────────────┤
│ Class   │ "no benefit" / "harm"│ F(therapy)│ Hard stop / contraindication   │
│ III     │ "should not"         │           │ alert / quality metric         │
└─────────┴──────────────────────┴───────────┴────────────────────────────────┘
""")

# %% [markdown]
# Yet Lomotan et al. (2010) showed that clinicians interpret "is
# reasonable" and "may be considered" with wide, overlapping ranges of
# obligation. If the guideline authors intended different deontic
# strengths, but clinicians (and EHR vendors) cannot reliably
# distinguish them, the formal structure is lost in implementation.
#
# This is precisely the problem that deontic logic can address:
# **formalize the intended obligation level, so the EHR implements
# exactly what the guideline authors meant.**

# %% [markdown]
# ---
# ## Part 8: Mr. Williams — The Deliberate Gap
#
# Mr. Williams (borderline risk, no enhancers) received **no
# recommendation**. This is not a bug — it is a design choice.
#
# The 2018 guideline deliberately leaves this group to clinical
# judgment and shared decision-making. In deontic terms, the guideline
# is **silent** — neither obligating nor prohibiting statin therapy.
#
# This is different from all three deontic operators:
#
# - O(statin): "you must prescribe" — not what the guideline says
# - P(statin): "you may prescribe" — weakly supported
# - F(statin): "you must not prescribe" — not what the guideline says
#
# Silence is a fourth option: **the guideline takes no normative
# position.** A rule-based system correctly produces no output. An LLM
# might hallucinate a recommendation.

# %%
# Verify: does the engine produce anything for Mr. Williams?
williams = results[5]
println("Mr. Williams (borderline risk, no enhancers):")
println("  Risk group:  $(williams.risk_group)")
println("  Statin rec:  $(williams.statin_rec)")
println("  Rules fired: $(length(williams.trace))")
if williams.statin_rec[1] == "none"
    println("\n  The system correctly produced NO recommendation.")
    println("    This gap is intentional — the guideline defers to")
    println("    clinician-patient shared decision-making.")
else
    println("\n  Unexpected: a recommendation was produced.")
    println("    Recommendation: $(williams.statin_rec[1]) [CF: $(round(williams.statin_rec[2], digits=2))]")
    println("    Check the rules — borderline risk without enhancers should have no recommendation.")
end

# %% [markdown]
# ---
# ## Reflection
#
# 1. The certainty factors in our rules encode the ACC/AHA Class of
#    Recommendation. Is this a valid use of CFs, or does it conflate
#    two different kinds of uncertainty (confidence in the evidence vs.
#    strength of obligation)?
#
# 2. UTSW researchers found that less than 40% of ASCVD patients were
#    on guideline-recommended statin intensity. The knowledge exists in
#    the guideline. The EHR could implement it. Why the gap?
#
# 3. Mr. Williams received no recommendation. If an LLM were asked
#    "should this patient take a statin?", what would it likely say?
#    How would you evaluate that answer?
#
# 4. Could you extend `statin_rules.yaml` to handle PCSK9 inhibitor
#    recommendations for very high-risk ASCVD patients (LDL ≥ 70 on
#    maximally tolerated statin)? What new rules would you add?
