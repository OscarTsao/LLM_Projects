# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Foundation Documents for NO-AUG → AUG Transformation** (2025-01-25)
  - INVENTORY.md: Complete codebase mapping (84KB, 1,954 lines)
    - Module tree and dependency analysis
    - Augmentation infrastructure status (60% complete, needs activation)
    - Entry points, data flow, and integration points
  - QUALITY-GATES.md: Production readiness criteria (7.3KB, 351 lines)
    - 10 quality gates: linting, formatting, type checking, tests, integration, performance, security, reproducibility, build, documentation
    - Validation commands and troubleshooting guides
    - Production readiness checklist
  - PR_PLAN.md: 5-PR transformation roadmap (14KB, 576 lines)
    - PR#1: Quality Gates & CI (8-12h)
    - PR#2: Augmentation Integration (16-24h)
    - PR#3: HPO Integration (12-16h)
    - PR#4: Packaging & Security (10-14h)
    - PR#5: Documentation (8-12h)
    - Total effort: 54-78 hours over 3-week timeline

- **Augmentation Configuration System**
  - configs/augmentation/default.yaml: Default augmentation settings
  - Configurable augmentation scope (train_only/all/none)
  - Deterministic augmentation with seed control
  - Per-method configuration overrides
  - Cache configuration for performance optimization

### Infrastructure
- Existing augmentation infrastructure identified and documented:
  - 17 CPU-light augmenters in registry (nlpaug + TextAttack)
  - Pipeline class with deterministic seeding
  - Worker initialization for multi-GPU support
  - Dataset hooks already in place (60% complete)

### Documentation
- Comprehensive codebase analysis and mapping
- Production-ready quality gates and acceptance criteria
- Detailed transformation roadmap with timeline
- Configuration examples and usage patterns

### Key Insights
- **60% of augmentation infrastructure already exists**
- Transformation requires **activation, not building from scratch**
- Infrastructure includes: registry (17 methods), pipeline, worker init, dataset hooks
- Main gap: Integration testing and HPO search space

---

## [0.1.0] - 2024-XX-XX

### Added
- Initial NO-AUG baseline implementation
- Four architectures: Criteria, Evidence, Share, Joint
- Strict field separation validation (criteria uses `status`, evidence uses `cases`)
- HPO system:
  - Multi-stage HPO (stage0-stage3: sanity, coarse, fine, refit)
  - Maximal HPO (600-1200 trials per architecture)
  - Sequential wrapper for all architectures
- MLflow experiment tracking
- Hydra configuration system with composition
- Reproducibility guarantees:
  - Seed management and deterministic algorithms
  - Hardware-optimized DataLoader settings
  - Mixed precision support (Float16/BFloat16)
- Production-ready training infrastructure:
  - Standalone Criteria training/evaluation scripts
  - HPO integration with real redsm5 data
  - Checkpoint management and early stopping
- Comprehensive testing:
  - 400+ test functions across 23 test files
  - STRICT field separation tests
  - Integration tests for HPO pipeline
- Development environment:
  - VS Code Dev Container with CUDA 12.1
  - Poetry dependency management
  - Pre-commit hooks for code quality

### Infrastructure
- GitHub Actions CI/CD workflows
- Pre-commit hooks (ruff, black, mypy)
- Security scanning (audit_security.py)
- SBOM generation (generate_sbom.py)
- License generation (generate_licenses.py)
- DataLoader benchmarking (bench_dataloader.py)
- Determinism verification (verify_determinism.py)

### Documentation
- README with installation and usage
- CLAUDE.md with project overview and guidance
- Training guides and HPO documentation
- CLI and Makefile reference
- Comprehensive docstrings and type hints

---

## Project Status

**Current State:** NO-AUG Baseline (Production-Ready)
**Next Milestone:** AUG-Enabled System (Foundation Documents Complete)
**Transformation Progress:** Planning Phase Complete, Implementation Phase Ready to Begin

### Transformation Roadmap

The project is transitioning from NO-AUG baseline to AUG-enabled production system through 5 sequential PRs:

1. ✅ **Foundation (Complete):** Planning documents, codebase analysis, quality gates
2. ⏳ **PR#1:** Quality Gates & CI Infrastructure (8-12h)
3. ⏳ **PR#2:** Augmentation Integration & Core Tests (16-24h)
4. ⏳ **PR#3:** HPO Integration & Observability (12-16h)
5. ⏳ **PR#4:** Packaging, Docker & Security (10-14h)
6. ⏳ **PR#5:** Documentation & Release (8-12h)

**Total Estimated Effort:** 54-78 developer-hours (6.75-9.75 days)
**Timeline:** 3 weeks

---

## Links

- [GitHub Repository](https://github.com/OscarTsao/DataAug_Criteria_Evidence)
- [Issues](https://github.com/OscarTsao/DataAug_Criteria_Evidence/issues)
- [Pull Requests](https://github.com/OscarTsao/DataAug_Criteria_Evidence/pulls)

---

[Unreleased]: https://github.com/OscarTsao/DataAug_Criteria_Evidence/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OscarTsao/DataAug_Criteria_Evidence/releases/tag/v0.1.0
