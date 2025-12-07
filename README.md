# GLens: Gravitational Lensing Research Projects

This repository contains research projects exploring gravitational lensing detection and analysis using machine learning.

---

## Projects

### 1. [Kinematics](./kinematics/)

**Multi-Modal Subhalo Detection via Lensed Kinematics**

Explores whether combining galaxy brightness (flux) with velocity maps (kinematics) can improve detection of dark matter subhalos in gravitationally lensed systems.

**Key components:**
- MaNGA IFU data processing pipeline
- SIS gravitational lensing simulation
- CNN-based subhalo detection experiments
- Ablation studies: flux-only vs velocity-only vs combined

See [`kinematics/README.md`](./kinematics/README.md) for detailed documentation.

---

## Repository Structure

```
glens/
├── kinematics/           # Subhalo detection via lensed kinematics
│   ├── src/              # Source code
│   ├── data/             # Data files (mostly gitignored)
│   ├── models/           # Trained model weights
│   ├── docs/             # Documentation
│   ├── work_log/         # Development logs
│   ├── README.md         # Project-specific README
│   └── requirements.txt  # Python dependencies
│
└── (future projects...)
```

---

## Getting Started

Each project has its own virtual environment and dependencies. Navigate to the specific project directory and follow its README instructions.

```bash
cd kinematics/
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

---

## License

This project is for research and educational purposes.
