# AI-Powered Radiomics for Assessment of Muscle Disorders

Bachelor thesis project (GUC MET): ultrasound-based radiomics, machine learning, and deep learning for muscle disease classification and FSHD severity assessment.

**Author:** Eyad Ghonem

## What is in this repository

| Included | Excluded (`.gitignore`) |
|----------|-------------------------|
| Thesis drafts & methodology docs | Raw ultrasound PNGs (`data/`) |
| Training & evaluation Python scripts | Trained `.keras` / `.pkl` weights |
| Streamlit GUI source (`gui_demo/`) | `venv/`, large `output/` logs |
| `scripts/train_gui_on_real_ultrasound.py` | Multi-GB CSV feature dumps |

**Real image sources (local only):**

1. `data/ULTRASOUND_LABELD_1_FSHD/images` — FSHD  
2. `data/images_extracted_from_mat_LABELED/` — Normal, IBM, Dermatomyositis, Polymyositis  

~28,199 labeled PNGs total. `ULTRASOUND_LABELD_2` is tabular-only and not used for image training.

## For Claude / thesis writing

Start here:

- **[THESIS_DRAFT_FOR_CLAUDE.md](THESIS_DRAFT_FOR_CLAUDE.md)** — instructions, abstract, Chapters 3–5 outline, honest limitations  
- **[BACHELOR_THESIS_SOURCE_PACK.md](BACHELOR_THESIS_SOURCE_PACK.md)** — dataset & results summary  
- **[CLAUDE.md](CLAUDE.md)** — repo conventions for AI assistants  

## Quick start (after cloning)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r gui_demo\requirements_gui.txt
# Place datasets under data/ (see paths in gui_demo/paths.py)

# Train all GUI models on real images (hours; needs data/ locally)
python scripts\train_gui_on_real_ultrasound.py --skip-features --epochs 10

# Demo app
streamlit run gui_demo\app.py
```

## GitHub upload

```powershell
git init
git add .
git commit -m "Initial thesis codebase and documentation for Claude"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

Use a **private** repository if your institution requires it. Do not commit patient images or `.env` secrets.
