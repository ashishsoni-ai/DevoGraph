# DHGNN — Dynamic Hypergraph Neural Network for C. elegans Embryogenesis

**Open Source Cohort / DevoWorm Project 2026**

---

## Project Overview

This project extends the DevoGraph framework by building a **Dynamic Hypergraph Neural Network (DHGNN)** to model C. elegans embryogenesis. It integrates three biologically distinct hyperedge types into a unified incidence matrix and learns per-cell embeddings across developmental time. The model is trained on **cell fate** (founder-lineage classification) and evaluated by fate accuracy and by the **lineage-coherence** of the learned embeddings.

### Three Hyperedge Types

| Type | Source | Fixed? | Biological Meaning |
|---|---|---|---|
| **Spatial** | `ce_temporal_data.csv` + DBSCAN | Rebuilt each t | Cells in physical proximity at timepoint t |
| **Lineage** | `cells_birth_and_pos.csv` | Fixed | Parent–daughter division bonds |
| **Functional** | `Connectome.csv` + `Alignment_map_csv.csv` | Fixed | Adult FFL neural circuit motifs |


---
---

## Repository Structure

```
DHGNN/
├── raw_dataset/                        ← raw input files (read-only)
│   ├── ce_temporal_data.csv            — 250,113 rows: cell positions across 190 frames
│   ├── cells_birth_and_pos.csv         — 642 rows: cell division records
│   ├── Connectome.csv                  — adult C. elegans synaptic connectome
│   ├── Alignment_map_csv.csv           — WormBase neuron→lineage name mapping
│   └── CE_cell_graph_data_processed.csv
│
├── output_tables/                      ← generated lookup tables (3 hyperedge sources)
│   ├── lineage_lookup_table.csv        — 642 parent/daughter-triplet hyperedges
│   ├── spatial_lookup_table.csv        — 5,336 per-timepoint DBSCAN community hyperedges
│   └── functional_lookup_table.csv     — 4,436 FFL motif hyperedges
│
├── dhgnn_lib/                           ← shared package imported by the notebooks
│   ├── cell_universe.py                — fixed cell name <-> index universe (N = 1,625)
│   ├── node_features.py                — per-timepoint (N, 10) node feature matrices
│   ├── hyperedges.py                   — spatial/lineage/functional builders + k-means/k-NN re-clustering
│   ├── incidence.py                    — HyperedgeSet / HyperedgeBatch + H_aug(t) assembly (3 blocks)
│   ├── hypersagnn.py                   — Hyper-SAGNN V2E2V convolution layer
│   ├── dhgnn_model.py                  — DynamicHypergraphStack + DHGNNModel (fate head)
│   ├── training.py                     — fate loss, metrics, train loop
│   ├── labels.py                       — fate label
│   ├── visualization.py                — plotting helpers + lineage_coherence()
│   └── build_functional_lookup_table_with_viz.py ← builds functional_lookup_table.csv
│
├── notebooks/
│   ├── run_functional.ipynb            — builds the functional lookup table
│   ├── 01_data_pipeline.ipynb          — cell universe, node features, lineage & spatial lookup tables
│   ├── 02_incidence_matrix.ipynb       — assembles & visualizes H_aug(t)
│   ├── 03_hypersagnn_convolution.ipynb — Hyper-SAGNN V2E2V vertex convolution
│   ├── 04_dhgnn_training.ipynb         — DHGNN fate training (dynamic re-clustering)
│   ├── 05_visualizations_and_simulation.ipynb — embeddings / development visualizations
│   └── 06_testing_and_validation.ipynb — test suite + lineage-coherence
│
├── checkpoints/                        ← trained model + history + metadata
└── README.md
```

---

## How to Run

**Step 1 — Install dependencies**
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn networkx torch jupyter nbconvert ipykernel
```

**Step 2 — (Optional) Regenerate the functional lookup table**

Run from the project root:
```bash
python dhgnn_lib/build_functional_lookup_table_with_viz.py     # -> output_tables/functional_lookup_table.csv (4,436 rows)
```
or open `notebooks/run_functional.ipynb` and run all cells.

**Step 3 — Run the pipeline notebooks (in order)**

Open `notebooks/01_data_pipeline.ipynb` … `06_testing_and_validation.ipynb` in Jupyter and run all cells. Notebook 01 writes the lineage/spatial lookup tables; 02 assembles `H_aug(t)`; 04 trains the model; 06 validates it.

---

## Key Design Decisions

- **DBSCAN (spatial hyperedges + boundary feature)**: `eps=15 voxels, min_samples=3`.
- **Dynamic re-clustering (DHGNN)**: between layers the spatial block is rebuilt from the *learned embeddings* using **k-means (global) + k-NN (local)**.
- **Node features (10)**: `x, y, z, size, lineage_generation, presence, vx, vy, vz, boundary_dist`.
- **Cell-disjoint fate split**: fate is supervised on 80% of cells and evaluated on the held-out 20%, so accuracy reflects genuine generalisation, not identity memorisation.
- **Duplicate handling**: `drop_duplicates('cell')` — first occurrence per cell name per frame.

---

## Incidence Matrix

`H_aug(t) = [ H_spatial(t) | H_lineage | H_functional ]`

- **Rows**: 1,625 cells (fixed alphabetical universe)
- **Columns**: 642 lineage + 4,436 functional + per-t spatial edges (~5,100 total)
- **Storage**: sparse COO-style tensors (cell_index, edge_index, edge_type, edge_weight)

---

## Metrics

- **Cell fate accuracy** (held-out, cell-disjoint) — the supervised headline metric.
- **Lineage-coherence** (silhouette + k-NN purity of the embeddings vs. a raw-feature baseline) — representation quality.
