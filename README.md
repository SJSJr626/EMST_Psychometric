# EMST_Psychometric

Beta-tested scripts for the **ORCS-VC** project — *Organization of Reliable Stimuli into Valid Conditions*.

All current code is in Python, with minimal R integration via `rpy2`. These scripts will support a future Docker image to make the pipeline fully stable and reproducible.

## What the toolkit will achieve

- **Block- or rater-level exclusion**  
  Detect and remove unreliable responses using response times (RTs) and IRT-derived metrics to preserve high-quality data.

- **Stimulus cleaning & winnowing**  
  - *2a.* Identify and drop unintended/outlier ratings to improve stimulus estimates.  
  - *2b.* Remove imprecise (i.e., unreliable) stimuli based on the target construct and metric type (scalar or categorical) to improve **divergent** validity.

- **Dynamic alignment of stimulus sets**  
  Align unbalanced sets intended to reflect the same construct or experimental conditions, improving **convergent** validity by balancing feature distributions.

## Project status

**As of Aug 26, 2025, 18:00 ET:**
- In progress: **Python** implementation for **categorical winnowing** (with manual tie-break rules).  
- In progress: **dynamic alignment** with **residual-guided swaps** and **early stopping** (beta).  
- Planned: Docker packaging.

## Installation

```bash
# Python (recommended: 3.10+)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# R integration from Python
# If using rpy2, ensure R (>= 4.3) is installed.
pip install rpy2
```

## Roadmap

- Response-quality control (QC) features  
- Scalar-based winnowing  
- Docker image with pinned Python/R dependencies for reproducible runs  
- Documentation site with examples and API reference

## Contributing

Issues and PRs are welcome. Please:
1. Open an issue describing the change/bug.
2. Include minimal repro data or a synthetic example.
3. Add/adjust tests where applicable.

## Citation

Pending.

## Contact

Questions or collaboration ideas: open an issue or reach out via the contact info in the repo profile.
