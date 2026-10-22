# Kartezio Illumination Robustness

## Overview

This project investigates whether Kartezio can evolve interpretable image-segmentation pipelines that remain reliable under changing illumination and noise conditions.

The system combines traditional computer-vision operations—such as thresholding, filtering, edge detection, and morphology—into graph-based pipelines evolved using Cartesian Genetic Programming.

## Project goals

The project aims to:

- Evolve compact and explainable segmentation pipelines.
- Improve robustness across illumination and noise conditions.
- Test generalisation to unseen scenes.
- Compare evolved pipelines with fixed Otsu and Sobel baselines.
- Analyse pipeline structure, complexity, and failure modes.

## Dataset

The project uses the hardware-items benchmark containing:

- 4 scenes.
- 13 shading levels.
- 14 noise levels.
- 728 grayscale images.
- Scene-level segmentation annotations.

The benchmark must be available under:

```text
hardware_items/data/
hardware_items/truth/
```

The original dataset is available from the [hardware-items benchmark repository on SourceForge](https://sourceforge.net/projects/gait-cad/files/Benchmarks/hardware_items/).

The benchmark is described in:

[Khan, Mikut, and Reischl (2015), “A Benchmark Data Set to Evaluate the Illumination Robustness of Image Processing Algorithms for Object Segmentation and Classification”](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131098)

## Project workflow

```text
Validate benchmark
    ↓
Evaluate fixed baselines
    ↓
Run synthetic pilot evolution
    ↓
Validate fitness strategies
    ↓
Run scene-held-out evolution
    ↓
Evaluate robustness and generalisation
```

## Evolution

Kartezio evolves directed image-processing graphs composed of project primitives, including:

- Intensity transformations.
- Thresholding.
- Blur and smoothing.
- Edge processing.
- Morphology.
- Hole filling.
- Border clearing.
- Small-object removal.

The main fitness objective is:

```text
loss = 1 - Q_seg
```

where `Q_seg` is the project’s segmentation-quality score. Lower loss indicates better segmentation.

Two fitness strategies are planned:

- Quality-only fitness.
- Robustness-aware fitness across easy, medium, and difficult conditions.

Final experiments will use leave-one-scene-out evaluation, with one scene held out for testing in each rotation.

## Pilot experiments

Synthetic pilot fixtures are used to validate the evolutionary workflow before benchmark-scale experiments. They include single and multiple objects, different shapes, touching objects, holes, border objects, illumination gradients, noise, empty images, and constant images.

Pilot outputs include:

- Fitness-history plots.
- Predictions for all test images.
- Machine-readable pipeline files.
- Human-readable pipeline descriptions.
- Run summaries.

## Baselines

The project includes Python and MATLAB-compatible Otsu and Sobel baselines.

Python baselines are used for primary comparisons with evolved Python pipelines. MATLAB results are retained as a paper-oriented reference. The Python Sobel implementation remains an approximation of MATLAB’s `edge(..., 'sobel')`, so the two Sobel results are not directly interchangeable.

## Metrics

The project reports:

- Dice.
- IoU.
- Object-count error.
- `Q_seg`.
- Segmentation-only robustness `R_seg`.
- Pipeline complexity.
- Per-scene and per-condition performance.

The original paper includes a classification term that is not currently implemented. Therefore, this project reports segmentation-only results and does not claim complete reproduction of the paper’s full score.

## Current status

Implemented:

- Dataset and ground-truth validation.
- Segmentation and robustness metrics.
- Python and MATLAB baseline workflows.
- Synthetic pilot fixtures.
- Kartezio primitives and pilot evolution.
- Pipeline export and visualisation.
- Baseline and pilot documentation.

Planned:

- Finalise quality-only and robustness-aware fitness.
- Run scene-held-out evolutionary experiments.
- Repeat selected runs across multiple seeds.
- Compare evolved pipelines with fixed baselines.
- Analyse robustness, complexity, and failure cases.

## Documentation

See:

- `docs/implementation_spec_illumination_robustness.md`
- `docs/benchmark_protocol.md`
- `docs/baseline_reproduction_report.md`
- `docs/pilot_test_report.md`

## Citation

This project uses the Kartezio framework. Please cite:

```bibtex
@article{cortacero2023evolutionary,
  title={Evolutionary design of explainable algorithms for biomedical image segmentation},
  author={Cortacero, K{\'e}vin and McKenzie, Brienne and M{\"u}ller, Sabina and Khazen, Roxana and Lafouresse, Fanny and Corsaut, Ga{\"e}lle and Van Acker, Nathalie and Frenois, Fran{\c{c}}ois-Xavier and Lamant, Laurence and Meyer, Nicolas and others},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7112},
  year={2023},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41467-023-42664-x},
  url={https://www.nature.com/articles/s41467-023-42664-x}
}
```

Citation for the benchmark:

```bibtex
@article{khan2015benchmark,
  title={A Benchmark Data Set to Evaluate the Illumination Robustness of Image Processing Algorithms for Object Segmentation and Classification},
  author={Khan, Arif ul Maula and Mikut, Ralf and Reischl, Markus},
  journal={PLOS ONE},
  volume={10},
  number={7},
  pages={e0131098},
  year={2015},
  doi={10.1371/journal.pone.0131098},
  url={https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131098}
}
```

## License

See `LICENSE` for the licensing terms of the original Kartezio project and this repository.
