# M3 Pre-Class Explorations: The Role of Logic in AI for Health

**PUBH 5106 — AI in Health Applications | Module 3**

Interactive Julia notebooks for the four segments of Module 3. All notebooks share a single Julia environment — once Binder builds the image for one notebook, the others load instantly.

## Notebooks

| Notebook | Segment | Topic | Dependencies |
|----------|---------|-------|-------------|
| `M3_explore_mycin.ipynb` | A | MYCIN-style rule engine (clinical: bacteria → antibiotics) | YAML |
| `M3_explore_outbreak.ipynb` | A | Rule engine (public health: symptoms → pathogen → response) | YAML |
| `M3_explore_deontic.ipynb` | B | Deontic logic in clinical guidelines | Gamen, YAML |
| `M3_statin_guideline.ipynb` | A+B | ACC/AHA statin guideline: rules + deontic verification | Gamen, YAML |
| `M3_explore_kg.ipynb` | D | Knowledge graph for lipid management | YAML |

## Knowledge Base Files (YAML)

All domain knowledge is in YAML files — following Buchanan's principle of separating knowledge from code. Students extend the system by editing YAML, not code.

| File | Used by | Contents |
|------|---------|----------|
| `mycin_rules.yaml` | MYCIN notebook | 12 rules: organism identification + antibiotic therapy |
| `neisseria_rules.yaml` | MYCIN notebook | Student extension: 2 Neisseria rules |
| `outbreak_rules.yaml` | Outbreak notebook | 17 rules: foodborne pathogen ID + public health response |
| `student_outbreak_rules.yaml` | Outbreak notebook | Student extension: 2 Listeria rules |
| `guidelines.yaml` | Deontic notebook | 5 clinical guidelines with deontic operators |
| `conflict_test.yaml` | Deontic notebook | 2 intentionally contradictory guidelines |
| `student_guidelines.yaml` | Deontic notebook | Student extension: DVT prophylaxis guideline |
| `statin_rules.yaml` | Statin notebook | 19 ACC/AHA guideline rules (classification + therapy) |
| `test_patients.yaml` | Statin notebook | 5 test patients across the cardiovascular risk spectrum |
| `lipid_kg.yaml` | KG notebook | 41 triples: drugs, targets, diseases, genes, pathways |
| `student_triples.yaml` | KG notebook | Student extension: bempedoic acid triples |

## Launch on Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/utsw-hdsb/pubh5106-m3-notebooks/HEAD)

## Run Locally

```bash
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/chapmanbe/Gamen.jl"); Pkg.instantiate()'
jupyter notebook
```

Requires Julia 1.10+.
