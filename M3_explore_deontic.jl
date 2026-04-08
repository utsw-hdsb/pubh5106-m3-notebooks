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
# # Exploration: Deontic Logic in Clinical Guidelines
#
# **PUBH 5106 -- AI in Health Applications | Module 3, Segment B**
#
# ## What This Notebook Is For
#
# This is a **pre-class exploration** -- a short, self-guided activity
# designed to build your intuition about deontic logic before the lecture.
# There is nothing to submit. Experiment freely.
#
# ## What You Will Do
#
# Clinical guidelines are full of words like "must," "should," and "may."
# These words carry different levels of obligation -- but how different?
# Lomotan et al. (2010) found that clinicians agree on what "must" means
# (near-universal obligation) but disagree substantially on "should,"
# "is recommended," and "may be considered."
#
# **Deontic logic** formalizes these concepts:
# - **Obligation** (O): "must" -- the action is required
# - **Permission** (P): "may" -- the action is allowed
# - **Prohibition** (F): "must not" -- the action is forbidden
#
# In this notebook you will:
#
# 1. Load clinical guidelines from a YAML file and translate them into
#    formal deontic logic automatically
# 2. Build models of clinical scenarios and check which obligations hold
# 3. Use automated theorem proving to verify guideline consistency
# 4. Discover how the same English sentence produces different logical
#    consequences depending on how you formalize "should"
#
# Following Buchanan's principle, the **guidelines live in YAML files**
# and the **logic engine lives in this notebook**. To add or modify
# guidelines, you edit the YAML -- not the code.
#
# **Time:** ~25 minutes
#
# **Prerequisites:** No prior logic experience. The notebook introduces
# everything you need.

# %% [markdown]
# ---
# ## Setup

# %%
using Gamen
include("M3Utils.jl")
using .M3Utils

# %% [markdown]
# ---
# ## Part 1: From YAML to Logic
#
# Double-click `guidelines.yaml` in the JupyterLab sidebar to open it. Each guideline
# has an English sentence, a deontic term ("must," "should," "may"),
# and a formula specification that tells the engine how to formalize it.
#
# The mapping from deontic terms to logic:
#
# | English | Deontic | Symbol | Gamen.jl |
# |---------|---------|--------|----------|
# | "must" | Obligation | O(p) | `Box(p)` or `□(p)` |
# | "may" | Permission | P(p) | `Diamond(p)` or `◇(p)` |
# | "must not" | Prohibition | F(p) | `Box(Not(p))` |
#
# These are connected by a fundamental equivalence:
#
# > **Permission is the absence of a prohibition:**
# > P(p) ≡ ¬O(¬p)  -- "you may do p" means "it is not obligatory that you not do p"

# %%
guidelines = load_guidelines("guidelines.yaml"; logic_module=Main)
show_guidelines(guidelines)

# %% [markdown]
# The code above knows nothing about consent, antibiotics, or
# thrombolytics. It reads a YAML file, sees `op: box`, and constructs
# `□(atom)`. All clinical knowledge is in the file.

# %% [markdown]
# ---
# ## Part 2: Worlds and Obligations
#
# To evaluate deontic formulas, we need a **model** -- a set of possible
# worlds connected by an accessibility relation.
#
# In deontic logic, think of it this way:
# - Each **world** represents a possible state of affairs
# - The **accessibility relation** connects a world to all the worlds
#   that are **normatively acceptable** from it (worlds where all
#   obligations are met)
# - **□(p)** is true at a world if p is true in **all** acceptable worlds
# - **◇(p)** is true at a world if p is true in **at least one** acceptable world
#
# ### The D Axiom: Obligations Must Be Satisfiable
#
# Deontic logic uses the **KD system**, which adds one crucial axiom:
#
# > **D: □(p) → ◇(p)**
# > "If something is obligatory, it must be possible."
#
# This means every world must have at least one accessible (acceptable)
# world. You cannot have an obligation with no way to fulfill it.

# %%
# ── Build a model of a clinical scenario ──
#
# Scenario: A patient presents with suspected bacteremia.
#
# World :actual  = current situation (no cultures drawn yet)
# World :good    = ideal practice (cultures drawn, consent obtained, plan started)
# World :partial = cultures drawn but discharge planning delayed
# World :bad     = antibiotics started without cultures

frame = KripkeFrame(
    [:actual, :good, :partial, :bad],
    [
        :actual  => :good,
        :actual  => :partial,
        :good    => :good,
        :partial => :partial,
        :bad     => :bad,
    ]
)

model = KripkeModel(frame, [
    :consent         => [:good, :partial, :bad],
    :blood_cultures  => [:good, :partial],
    :antibiotics     => [:good, :partial, :bad],
    :discharge_plan  => [:good],
    :active_bleeding => Symbol[],
    :thrombolytic    => Symbol[],
    :adjust_duration => [:good, :partial],
])

println("Clinical scenario model:")
println("  Worlds: ", frame.worlds)
println()
for w in [:actual, :good, :partial, :bad]
    println("  $w → accessible: $(accessible(frame, w))")
end

# %%
println("\nIs this frame serial (valid for deontic logic)? ", is_serial(frame))

# %% [markdown]
# ### Checking Obligations
#
# Now let's check which guidelines hold at each world.

# %%
println("Guideline evaluation at each world:\n")

let worlds = [:actual, :good, :partial, :bad]
    header = rpad("Guideline", 40)
    for w in worlds
        header *= rpad(string(w), 12)
    end
    println(header)
    println("-"^length(header))

    for g in guidelines
        row = rpad("$(g.id): $(g.term) $(first(g.atoms))", 40)
        for w in worlds
            result = satisfies(model, w, g.formula)
            row *= rpad(result ? "TRUE" : "false", 12)
        end
        println(row)
    end
end

# %% [markdown]
# **Read the table above.** Notice:
#
# - At `:actual`, G1 (consent) is TRUE because consent is obtained in
#   **all** accessible worlds (`:good` and `:partial`). The obligation holds.
# - At `:actual`, G2 (cultures) is TRUE because cultures are drawn in
#   **all** accessible worlds.
# - At `:actual`, G5 (discharge planning) is FALSE because discharge
#   planning is not done in `:partial` -- one accessible world fails it.
# - G3 (permission to adjust) depends on whether at least one accessible
#   world makes it true.
#
# This is the key insight: **obligations require unanimity among
# acceptable worlds; permissions require only one.**

# %% [markdown]
# ---
# ## Part 3: The Cross-Definitions
#
# Obligation, permission, and prohibition are not independent concepts.
# They are connected by logical equivalences:
#
# | Concept | Definition |
# |---------|-----------|
# | Permission | P(p) ≡ ¬O(¬p) |
# | Prohibition | F(p) ≡ O(¬p) |
#
# Let's verify these with the theorem prover.

# %%
p = Atom(:p)

# Permission ≡ ¬Obligation(¬p)
cross_def = Iff(Diamond(p), Not(Box(Not(p))))
result = tableau_proves(TABLEAU_KD, Formula[], cross_def)
println("P(p) ≡ ¬O(¬p):  ", result ? "VALID in KD" : "not valid")

# If obligatory, then permitted (D axiom)
d_axiom = Implies(Box(p), Diamond(p))
result2 = tableau_proves(TABLEAU_KD, Formula[], d_axiom)
println("O(p) → P(p):    ", result2 ? "VALID in KD" : "not valid")

# Sanity check: O(p) → p is NOT valid in KD
# (Just because something is obligatory doesn't mean it happens!)
ought_implies_is = Implies(Box(p), p)
result3 = tableau_proves(TABLEAU_KD, Formula[], ought_implies_is)
println("O(p) → p:       ", result3 ? "VALID in KD" : "NOT VALID (correct!)")

# %% [markdown]
# That last result is important: **deontic logic separates "ought" from
# "is."** Just because something is obligatory does not mean it actually
# happens. People violate obligations all the time.

# %% [markdown]
# ---
# ## Part 4: Checking Guideline Consistency
#
# When multiple guidelines apply to the same patient, can they conflict?
# The theorem prover can check whether a set of guidelines is
# **consistent** -- whether there exists any possible world where all
# of them can be simultaneously satisfied.

# %%
# ── Are our guidelines consistent with each other? ──

obligation_formulas = Formula[g.formula for g in guidelines if g.type == "obligation"]
consistent = tableau_consistent(TABLEAU_KD, obligation_formulas)
if consistent
    println("All obligation guidelines consistent? YES")
    println("  All three obligations can coexist — no conflict.")
else
    println("All obligation guidelines consistent? NO")
    println("  The guidelines contain a contradiction!")
end

# %% [markdown]
# ### Introducing a Conflict
#
# Now let's load `conflict_test.yaml`, which contains guidelines that
# intentionally contradict the originals.

# %%
conflicts = load_guidelines("conflict_test.yaml"; logic_module=Main)
show_guidelines(conflicts)

# %%
# G5 says: O(discharge_plan)
# G6 says: O(¬discharge_plan)
# These should be inconsistent.

g5_formula = guidelines[5].formula   # □(discharge_plan)
g6_formula = conflicts[1].formula    # □(¬discharge_plan)

consistent2 = tableau_consistent(TABLEAU_KD, Formula[g5_formula, g6_formula])
if consistent2
    println("G5 + G6 (discharge plan: must vs must not): consistent")
    println("  Surprisingly, these two guidelines are compatible.")
else
    println("G5 + G6 (discharge plan: must vs must not): INCONSISTENT")
    println("  The system detects the contradiction!")
end

# %%
# For a bleeding patient who needs thrombolytics:
# G4 gives O(¬thrombolytic), G7 gives O(thrombolytic)

conflict_set = Formula[Box(Not(Atom(:thrombolytic))), Box(Atom(:thrombolytic))]
consistent3 = tableau_consistent(TABLEAU_KD, conflict_set)
if consistent3
    println("O(¬thrombolytic) + O(thrombolytic): consistent")
    println("  These obligations can coexist (unexpected!).")
else
    println("O(¬thrombolytic) + O(thrombolytic): INCONSISTENT")
    println("  Real clinical dilemma: bleeding patient who needs thrombolytics.")
    println("  This is why clinical judgment cannot be fully automated.")
end

# %% [markdown]
# The theorem prover formally detected both conflicts. In a real EHR
# system, this kind of automated consistency checking could flag
# contradictory clinical decision support rules *before* they
# reach clinicians.

# %% [markdown]
# ---
# ## Part 5: The Ambiguity of "Should"
#
# Lomotan et al. (2010) surveyed 445 health professionals and found
# that while **"must"** is interpreted consistently (median obligation
# score: 100/100, IQR = 5), the word **"should"** has a wide range of
# interpretations (IQR spanning 30+ points).
#
# This matters for implementation. Consider G2:
#
# > *"Patients with bacteremia **should** receive blood cultures before
# > initiating antibiotics."*
#
# How you formalize "should" determines what the EHR does:

# %%
cultures = Atom(:blood_cultures)

# ── Interpretation A: "should" = obligation (hard stop) ──
# ── Interpretation B: "should" = permission (soft reminder) ──

println("Interpretation A (obligation):")
result_a = tableau_proves(TABLEAU_KD, Formula[Box(cultures)], Diamond(cultures))
if result_a
    println("  O(cultures) → P(cultures)? true")
    println("  If it's obligatory, it's certainly permitted.")
else
    println("  O(cultures) → P(cultures)? false")
    println("  Obligation does not imply permission (unexpected!).")
end

println("\nInterpretation B (permission):")
result_b = tableau_proves(TABLEAU_KD, Formula[Diamond(cultures)], Box(cultures))
if result_b
    println("  P(cultures) → O(cultures)? true")
    println("  Permission implies obligation (unexpected!).")
else
    println("  P(cultures) → O(cultures)? false")
    println("  Permission does NOT imply obligation.")
end

# %% [markdown]
# **This is the core problem Lomotan et al. identified.** The same
# English word -- "should" -- can be formalized as either obligation
# or permission. The choice has real consequences:
#
# | | Obligation (hard stop) | Permission (soft reminder) |
# |---|---|---|
# | EHR behavior | Blocks antibiotic order | Shows yellow banner |
# | If not followed | Documented violation | No violation |
# | Patient safety | Higher (prevents errors) | Lower (allows override) |
# | Clinician autonomy | Lower (restricted) | Higher (informed choice) |
# | Alert fatigue risk | Higher | Lower |
#
# Three EHR vendors implementing the same guideline with different
# interpretations of "should" will produce three different clinical
# decision support systems. This is not a hypothetical — it is the
# current state of affairs.

# %% [markdown]
# ---
# ## Part 6: Your Turn -- Add a Guideline
#
# Double-click `student_guidelines.yaml` in the JupyterLab sidebar to open it. It contains one
# example guideline (DVT prophylaxis). You can modify it or add more.
#
# Then run the cell below -- the engine loads your file and formalizes
# your guidelines automatically.

# %%
student_guidelines = load_guidelines("student_guidelines.yaml"; logic_module=Main)
show_guidelines(student_guidelines)

# %%
# Check consistency of your guidelines with the originals
all_formulas = Formula[g.formula for g in vcat(guidelines, student_guidelines)]
consistent_all = tableau_consistent(TABLEAU_KD, all_formulas)
if consistent_all
    println("All guidelines (original + yours) consistent? YES")
    println("  Your guideline is compatible with the existing set.")
else
    println("All guidelines (original + yours) consistent? NO")
    println("  Your guideline conflicts with one or more existing guidelines!")
end

# %% [markdown]
# You added a guideline by editing a YAML file. The logic engine
# automatically formalized it and checked it for consistency with the
# existing guidelines. This is Buchanan's principle: **separate the
# knowledge from the code.**

# %% [markdown]
# ---
# ## Part 7: Why This Matters
#
# You have now seen three things that deontic logic can do:
#
# 1. **Formalize ambiguous guidelines** — translate "should" into
#    precise logical claims with testable consequences
# 2. **Detect conflicts** — automatically find when two guidelines
#    cannot both be satisfied (O(p) and O(¬p))
# 3. **Distinguish obligation from permission** — show that the choice
#    of formalization determines what the EHR system does
#
# ### Connections
#
# - **Module 3 Segment A (MYCIN)**: MYCIN's production rules were a
#   form of logic-based AI. Deontic logic extends that idea to reasoning
#   about what *should* be done, not just what *is* true.
# - **Module 2 (LLMs)**: An LLM can generate text that sounds like a
#   guideline, but it cannot formally verify that two guidelines are
#   consistent. Logic can.
# - **Module 3 Lab**: Knowledge graphs combine structured knowledge
#   (like our deontic rules) with neural methods. This is the
#   **neurosymbolic** approach.
#
# ### The Deeper Point
#
# The Lomotan paper's recommendation was simple: guideline authors
# should use only three terms — **"must," "should," "may"** — and
# define them precisely. Deontic logic is the tool for making that
# precision formal and machine-checkable.

# %% [markdown]
# ---
# ## Reflection
#
# Before the lecture, think about:
#
# 1. Pick a clinical guideline you know (from a rotation, a textbook,
#    or a public guidelines database). What deontic terms does it use?
#    Could you add it to `student_guidelines.yaml`?
#
# 2. If an EHR vendor asked you whether a guideline "should" trigger a
#    hard stop or a soft reminder, how would you decide? What information
#    would you need from the guideline authors?
#
# 3. The MYCIN notebook showed you inspectable, rule-based reasoning.
#    This notebook showed you formal verification of normative rules.
#    How might you combine the two?
#
# 4. The guidelines are in YAML files that any clinician can read.
#    The logic engine verified them automatically. What are the
#    implications for guideline development and EHR implementation?
