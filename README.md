<div align="center">

# BrainSymphony  
### A Lightweight, Modular Transformer-Driven Fusion of fMRI Time Series and Structural Connectivity

</div>

> **BrainSymphony** is a **parameter-efficient multimodal foundation model** that jointly represents **fMRI time series** and **diffusion MRI–derived structural connectivity** in a unified ROI embedding space. It is designed to be **modular and plug-and-play**: you can use the fMRI pathway, the structural pathway, or both with adaptive fusion.  
>  
> Paper: *BrainSymphony: A Lightweight, Modular Transformer-Driven Fusion of fMRI Time Series and Structural Connectivity* (Khajehnejad, Habibollahi, Stoliker, Razi):contentReference[oaicite:3]{index=3}

---

## Highlights

- **Multimodal by design:** parallel fMRI encoders + a **Signed Graph Transformer** for structural connectomes, fused by an **adaptive gating** mechanism:contentReference[oaicite:4]{index=4}.  
- **Efficient, not over-scaled:** BrainSymphony (fusion) achieves top benchmark performance with **~5.6M parameters**, far fewer than much larger neuroimaging foundation models:contentReference[oaicite:5]{index=5}.  
- **Interpretable mechanisms:** attention maps provide directed, network-level signatures that reveal **drug-induced, context-dependent reorganization** in an external psilocybin dataset:contentReference[oaicite:6]{index=6}.

---

## Architecture (at a glance)

BrainSymphony contains:

1) **Spatio–Temporal fMRI encoder**  
   - **Spatial Transformer**: models inter-regional dependencies (ROI-wise attention)  
   - **Temporal Transformer**: models neural dynamics across time  
   - **1D-CNN context extractor**: captures local temporal patterns  
   These streams are distilled into compact latents by a **Perceiver** module:contentReference[oaicite:7]{index=7}.

2) **Structural encoder (dMRI-SC)**  
   - **Signed Graph Transformer** encoding the weighted structural connectome:contentReference[oaicite:8]{index=8}.

3) **Adaptive fusion gate**  
   - Dynamically weights functional vs. structural embeddings per task:contentReference[oaicite:9]{index=9}.

<p align="center">
  <img src="assets/fig1_architecture.png" width="900" alt="BrainSymphony architecture (Fig. 1)" />
</p>

**Figure to add:** export **Fig. 1** from the PDF as `assets/fig1_architecture.png`:contentReference[oaicite:10]{index=10}.

---

## Key results

### State-of-the-art performance with orders-of-magnitude fewer parameters
Across HCP-Aging benchmarks, BrainSymphony’s multimodal fusion variant outperforms strong baselines while remaining compact (5.6M params):contentReference[oaicite:11]{index=11}.

<p align="center">
  <img src="assets/fig2_benchmarks.png" width="900" alt="Benchmark performance and parameter efficiency (Fig. 2)" />
</p>

**Figure to add:** export **Fig. 2** as `assets/fig2_benchmarks.png`:contentReference[oaicite:12]{index=12}.

### External validation + interpretability on psilocybin (PsiConnect)
Without any psychedelic training, BrainSymphony reconstructs held-out ROI time series on PsiConnect and yields interpretable attention/influence patterns that reveal context-dependent drug effects:contentReference[oaicite:13]{index=13}.

<p align="center">
  <img src="assets/fig4_psiconnect_attention.png" width="900" alt="PsiConnect reconstruction and attention mapping (Fig. 4)" />
</p>

**Figure to add:** export **Fig. 4** as `assets/fig4_psiconnect_attention.png`:contentReference[oaicite:14]{index=14}.

(Optional, if you want to feature subjective intensity / MEQ effects:)

<p align="center">
  <img src="assets/fig5_meq_subgroups.png" width="900" alt="MEQ subgroup differences under psilocybin (Fig. 5)" />
</p>

**Figure to add:** export **Fig. 5** as `assets/fig5_meq_subgroups.png`:contentReference[oaicite:15]{index=15}.

---

## What’s in this repository (planned)

- `brainsymphony/` — model components (fMRI encoders, Perceiver fusion, Signed Graph Transformer, fusion gate)
- `configs/` — training and evaluation configs
- `scripts/` — data prep, pretraining, finetuning, evaluation
- `notebooks/` — examples and reproductions
- `assets/` — README figures

> This README is intentionally minimal and paper-focused for the initial release.  
> “Running / training / checkpoints” instructions will be added as the codebase is finalized.

---

## Data and preprocessing (important)

BrainSymphony expects **ROI-parcellated fMRI** and (optionally) **ROI-aligned structural connectivity**:

- fMRI should be robustly scaled per ROI (median-centered, divided by IQR) and use the same 450-ROI ordering:
  - 1–50: Tian-Scale III subcortex  
  - 51–250: Schaefer-400 left hemisphere  
  - 251–450: Schaefer-400 right hemisphere:contentReference[oaicite:16]{index=16}  
- Structural connectivity: streamline counts normalized by ROI volume, log10-transformed, aggregated to the same 450-ROI parcellation:contentReference[oaicite:17]{index=17}.

---

## Citation

If you use BrainSymphony in academic work, please cite:

```bibtex
@article{khajehnejad_brainsymphony,
  title     = {BrainSymphony: A Lightweight, Modular Transformer-Driven Fusion of fMRI Time Series and Structural Connectivity},
  author    = {Khajehnejad, Moein and Habibollahi, Forough and Stoliker, Devon and Razi, Adeel},
  note      = {Manuscript},
}
