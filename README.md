# TGIF: Multimodal Temporal Segmentation of Worker Activity States

TGIF is a research project on **multimodal fusion for temporal segmentation of worker activity states** in civil infrastructure and construction environments.

> **Status:** Active development. The repository structure, dataset interface, models, and evaluation pipeline are being prepared for a reproducible research release.

## Research objective

Given synchronized observations from multiple sensing modalities, TGIF aims to assign a worker-state label to each time step while accurately identifying transitions between activities. The broader goal is to build robust temporal understanding systems for complex, dynamic civil environments.

## Core research questions

- How should complementary modalities be fused when individual streams are noisy, incomplete, or asynchronous?
- How can a model combine local motion cues with long-range temporal context?
- How can rare states and short transition segments be recognized under class imbalance?
- How well do learned representations generalize across workers, tasks, and environments?

## Method overview

~~~mermaid
flowchart LR
    A[Multimodal sensor streams] --> B[Time alignment and quality control]
    B --> C[Modality-specific encoders]
    C --> D[Multimodal fusion]
    D --> E[Temporal segmentation model]
    E --> F[Worker-state timeline]
    F --> G[Frame, segment, and boundary evaluation]
~~~

The framework is designed to support controlled comparisons among early, late, and learned cross-modal fusion strategies, followed by temporal modeling and structured evaluation.

## Planned repository structure

~~~text
.
├── configs/                 # Experiment and dataset configurations
├── data/                    # Local data mount points; raw data is not committed
├── docs/                    # Dataset protocol and research notes
├── notebooks/               # Exploratory analysis and qualitative review
├── scripts/                 # Training, evaluation, and preprocessing entry points
├── src/tgif/
│   ├── data/                # Loading, alignment, augmentation, and sampling
│   ├── encoders/            # Modality-specific feature encoders
│   ├── fusion/              # Multimodal fusion modules
│   ├── temporal/            # Temporal segmentation architectures
│   ├── evaluation/          # Frame-, segment-, and boundary-level metrics
│   └── visualization/       # Timeline and error-analysis utilities
└── tests/                   # Unit and integration tests
~~~

## Evaluation plan

The project will report:

- **Frame-level performance:** macro F1, balanced accuracy, and per-class recall
- **Segment-level performance:** segmental F1 and intersection-over-union
- **Boundary quality:** transition timing error and edit-based metrics
- **Robustness:** missing-modality and sensor-noise stress tests
- **Generalization:** evaluation across workers, tasks, and environments
- **Ablations:** modality contribution, fusion strategy, and temporal context

## Development roadmap

- [ ] Finalize the state taxonomy and annotation protocol
- [ ] Implement synchronized multimodal data loaders
- [ ] Establish unimodal and fusion baselines
- [ ] Add temporal segmentation architectures
- [ ] Build reproducible training and evaluation pipelines
- [ ] Release ablations, qualitative timelines, and benchmark results
- [ ] Publish a model card and dataset documentation where release terms permit

## Reproducibility

Experiments will be configuration-driven and will record dataset splits, random seeds, preprocessing parameters, model settings, and evaluation outputs. Data and checkpoints will only be released after privacy, ownership, and project-release conditions are confirmed.

## Contact

**Jingxuan Duan**  
Academic: [jingxuad@andrew.cmu.edu](mailto:jingxuad@andrew.cmu.edu)  
Personal: [shureduan0912@gmail.com](mailto:shureduan0912@gmail.com)

## License

A license will be added after the research code and data-release conditions are finalized.
