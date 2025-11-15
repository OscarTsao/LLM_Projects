# Feature Specification: Storage-Optimized Training & HPO Pipeline

**Feature Branch**: `002-storage-optimized-training`  
**Created**: 2025-10-10  
**Status**: Draft  
**Input**: User description: "Storage-optimized training + HPO pipeline: limit artifacts/checkpoints while retaining metrics; load models from Hugging Face; keep MLflow tracking while pruning checkpoints; portable dev container setup for multi-machine training; evaluate best model per trial on test set and save metrics, config, and model checkpoint reference as JSON in trial directory."

## Clarifications

### Session 2025-10-10

- Q: What is the expected scale of a typical HPO workload (e.g., number of trials, model size, data size)? → A: 1000trials, 1-10gb models, <10gb data
- Q: How should the system authenticate with Hugging Face and the experiment tracking service? → A: huggingface-cli is logged in and experiments tracking with mlflow db in the project folder
- Q: What container technology should be used for the portable environment? → A: Docker
- Q: What are some specific out-of-scope items for this feature? → A: All of the above.
- Q: How should the system handle versioning of external dependencies like the MLflow API or Hugging Face models? → A: A and B.
- Q: When multiple checkpoints qualify as "best" (e.g., same validation metric), which tie-breaking rule should be applied? → A: Keep all tied checkpoints and mark as co-best.
- Q: When the tracking backend is temporarily unreachable and metrics are buffered locally, what should be the buffer limit before failing? → A: Buffer to disk with no hard limit, only warn after 100MB.
- Q: When should the retention policy trigger proactively to prevent storage exhaustion? → A: When available disk space drops below 10% of total capacity.
- Q: Where should the test set data be located and in what format? → A: Hugging Face datasets hub identifier (loaded via datasets library).
- Q: How should the system notify users of warnings and errors during long-running training/HPO jobs? → A: Structured JSON log file + human-readable stdout.
- Q: Where should the training and validation data be located? → A: Same Hugging Face dataset as test set (different splits).
- Q: Should HPO trials execute sequentially (one at a time) or in parallel (multiple trials concurrently)? → A: Sequential execution only (one trial at a time).
- Q: What should be the minimum checkpoint frequency to prevent excessive storage churn? → A: Every epoch (complete pass through training data).
- Q: How should the system determine which checkpoint is "best" for a trial? → A: User specifies a single metric to optimize (e.g., "accuracy").
- Q: When optimizing the specified metric, should it be maximized or minimized? → A: Always maximize (higher is better).
- Q: When a checkpoint write is interrupted (e.g., system crash during save), how should the system detect and handle corrupted checkpoints during resume? → A: Combination of atomic writes to prevent corruption + validation on resume as safety net.
- Q: When aggressive pruning (triggered at <10% disk space) cannot free sufficient space to continue training, what specific information should the error message provide to help the user resolve the issue? → A: Current disk usage, space needed for next checkpoint, list of largest artifacts, and commands to manually clean or adjust policy.
- Q: When the system buffers metrics to disk during a tracking backend outage, how should it handle the buffered data once the backend becomes available again? → A: Automatically replay with exponential backoff retry; keep buffer file until successful upload confirmed.
- Q: When loading models from Hugging Face and the model source is temporarily unavailable or rate-limited, what retry strategy should the system use before failing? → A: Check local cache first; if unavailable, retry with exponential backoff up to 5 attempts; then fail.
- Q: When the containerized environment is launched on a new machine, what is the expected maximum time for initial setup (pulling images, mounting data, etc.) before the environment is ready for training? → A: 15 minutes (moderate network, first-time setup).

- Q: Default retention policy when not specified? → A: keep_last_n=1, keep_best_k=1, max_total_size=10GB
- Q: Max retained best checkpoints per trial? → A: 2 (cap; co-best ties may exceed cap)

### Session 2025-10-11

- Q: When a Hugging Face authentication token expires during a long-running multi-day HPO study (1000 trials), how should the system handle token refresh? → A: Pause the study, emit a notification, and poll for token validity every 5 minutes before resuming automatically
- Q: Are there any specific compliance or regulatory standards (e.g., GDPR, HIPAA) that the system must adhere to, particularly concerning the storage and processing of ReDSM5-related data? → A: Both HIPAA and GDPR compliance are required.
- Q: Given the new requirement for HIPAA and GDPR compliance, the current assumption of using a local, unauthenticated MLflow database file (A-003) presents a significant compliance risk. How should the MLflow backend be configured to meet these new security and privacy requirements? → A: Use a remote, managed MLflow tracking server with authentication and encrypted communication (TLS).
- Q: The specification now requires a remote, managed MLflow tracking server. What authentication mechanism should the system use to connect to this server? → A: Short-lived tokens obtained via a secure token service (like HashiCorp Vault).
- Q: The lifecycle of an HPO trial is implicitly defined but not explicitly stated. To improve clarity, which of the following state machines best represents the lifecycle of a trial? → A: queued -> preparing -> running -> finishing -> completed / failed.
- Q: The specification currently mandates sequential HPO trial execution (FR-021). To provide context for future development, what is the primary reason for rejecting parallel trial execution at this stage? → A: To ensure deterministic behavior and reproducibility, which is harder with parallelism.

## User Scenarios & Testing *(mandatory)*


### User Story 1 - Run storage-optimized training/HPO with resume (Priority: P1)

An ML engineer can run model training and hyperparameter optimization (HPO) without exhausting storage. The system retains only necessary checkpoints for resume and best-model preservation while continuously logging metrics.

**Why this priority**: Enables reliable long-running experiments on limited storage while preserving observability and the ability to resume after interruption.

**Independent Test**: Launch a training job with HPO and an aggressive retention policy; verify metrics are fully logged, only the latest N and best-k checkpoints are retained, and an interrupted job resumes successfully from the latest retained checkpoint.

**Acceptance Scenarios**:

1. **Given** a configured retention policy (e.g., keep last N and keep best K), **When** training produces checkpoints, **Then** the system prunes older non-best checkpoints while metrics remain available in experiment tracking.
2. **Given** a running job is interrupted, **When** it is restarted, **Then** it resumes from the latest retained checkpoint and continues logging metrics without duplication.

---

### User Story 2 - Portable environment across machines (Priority: P2)

An ML engineer can spin up a portable, containerized environment on different machines (workstation, server, cloud instance) and run training/HPO consistently.

**Why this priority**: Reduces environment drift and speeds onboarding and reproducibility across hardware.

**Independent Test**: On a fresh machine with a standard container runtime and moderate network connectivity, start the development/training container (including image pull and initialization) and successfully run a sample training with experiment tracking enabled, completing the entire process within 15 minutes.

**Acceptance Scenarios**:

1. **Given** a host with a supported container runtime, **When** the provided container environment is launched, **Then** training/HPO runs without dependency issues and can access data, accelerators, and tracking endpoints as configured.

---

### User Story 3 - Per-study test evaluation and JSON report (Priority: P3)

A researcher can, after an HPO study completes, evaluate the best model from the entire study on the held-out test set and store a machine-readable JSON report containing the test metrics, the exact configuration used, and a reference to the model checkpoint.

**Why this priority**: Ensures comparable, auditable results while preventing test set overfitting across 1000+ trials. Each trial evaluates on validation set during training; test set evaluation occurs once per study.

**Independent Test**: After HPO completes, verify that the study directory contains a JSON file with required fields (study_id, best_trial_id, test_metrics, config, checkpoint reference) and that metrics correspond to evaluating the best model from the entire study on the test set.

**Acceptance Scenarios**:

1. **Given** a completed HPO study, **When** the evaluation step runs, **Then** the best model from the entire study is evaluated on the test set (loaded from Hugging Face datasets with split="test") and a JSON file with metrics, config, and checkpoint reference is saved in the study directory.
2. **Given** a tie in validation performance across checkpoints within the best trial, **When** selecting the "best" model, **Then** all tied checkpoints are marked as co-best, retained, evaluated on the test set, and their references included in the JSON report as an array.
3. **Given** an HPO study in progress, **When** individual trials complete, **Then** each trial evaluates its best model on the validation set only (not test set) to guide HPO optimization.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- Storage near exhaustion mid-run → retention triggers proactively when available disk space drops below 10% of total capacity; if aggressive pruning cannot free sufficient space, job fails gracefully with detailed error message including: current disk usage, space needed for next checkpoint, list of largest artifacts (with sizes and paths), and actionable commands to manually clean artifacts or adjust retention policy; error logged to both JSON log file and stdout; all metrics to-date preserved in tracking.
- Tracking backend unreachable temporarily → metrics buffered to disk; system automatically replays buffered metrics with exponential backoff retry when backend becomes available; buffer file retained until successful upload is confirmed; emit WARNING to JSON log and stdout when buffer exceeds 100MB; no hard limit, allowing training to continue even during extended tracking outages.
- Model source unavailable or rate-limited → check local Hugging Face cache first; if model not cached, retry download with exponential backoff (increasing delays: 1s, 2s, 4s, 8s, 16s) up to 5 attempts; if all retries fail, terminate with actionable ERROR message logged to JSON log and stdout including cache location and manual download instructions.
- Excessive checkpoint frequency → enforce minimum interval of one epoch between checkpoints to prevent churn and storage thrash.
- Interrupted job during checkpoint write → use atomic write pattern (write to temporary file, then rename) to prevent corruption; on resume, validate checkpoint integrity via checksum/hash before loading; fall back to previous valid checkpoint if validation fails.
- Trials with identical validation performance → keep all tied checkpoints and mark as co-best; retention policy must preserve all co-best checkpoints.
- Test set leakage prevention → ensure evaluation uses strictly held-out test split from Hugging Face datasets (loaded via datasets library with split="test") and is run only after training/HPO concludes for a trial.
- Hugging Face token expiration during long-running study → when authentication token expires mid-study, pause the study immediately (do not start new trials or checkpoints), emit WARNING to JSON log and stdout with timestamp and instructions for manual re-authentication, poll for token validity every 5 minutes using lightweight API validation; once token is valid again (user has re-authenticated via `huggingface-cli login`), automatically resume the study from the paused trial without losing progress or duplicating metrics; log successful token refresh and resumption.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The training/HPO system MUST implement a storage retention policy that limits retained checkpoints/artifacts while guaranteeing ability to resume and preserving the best model(s).
- **FR-002**: The retention policy MUST be configurable (e.g., keep last N, keep best K, maximum total size, checkpoint interval in epochs) and enforceable during long runs. If unspecified, defaults apply: keep_last_n=1, keep_best_k=1, keep_best_k_max=2, max_total_size=10GB (co-best ties may exceed cap).
- **FR-003**: Metrics and metadata MUST be continuously logged to an experiment tracking system and remain intact regardless of artifact pruning.
- **FR-004**: The system MUST support resuming training from the latest valid retained checkpoint without duplicating logged metrics; MUST validate checkpoint integrity (via checksum/hash) before loading and fall back to the previous valid checkpoint if corruption is detected.
- **FR-005**: The system MUST load base and/or fine-tunable models from the Hugging Face model hub given user-specified model identifiers; MUST check local Hugging Face cache first; if not cached, MUST retry download with exponential backoff (delays: 1s, 2s, 4s, 8s, 16s) up to 5 attempts before failing with an actionable error message.
- **FR-006**: Each HPO trial MUST maintain its own directory for artifacts and logs to ensure isolation and traceability. Trial directories MUST follow the canonical path pattern: `experiments/trial_<uuid>/` where `<uuid>` is the trial's unique identifier.
- **FR-007**: For every trial, the best model(s) (determined by maximizing a user-specified optimization metric such as "accuracy" or "val_f1"; when tied, all co-best checkpoints are preserved and marked) MUST be evaluated on the validation set. For large-scale HPO studies (1000+ trials), test set evaluation MUST occur once per study (after all trials complete), evaluating only the best model from the entire study.
- **FR-008**: For large-scale HPO studies (1000+ trials), a study-level JSON report MUST be saved in the study directory containing at least: test metrics (by name/value), full resolved configuration of the best trial (including model id, data, seeds, hyperparameters), study_id, best_trial_id, and references to the best checkpoint(s) (path or unique id; array when multiple co-best checkpoints exist within the best trial). Study directories MUST follow the canonical path pattern: `experiments/study_<uuid>/` where `<uuid>` is the study's unique identifier.
- **FR-009**: The system MUST prevent runaway storage growth by pruning immediately after checkpoint creation when limits are exceeded, without blocking metric logging.
- **FR-010**: The system MUST expose dry-run and disabling options for checkpointing (e.g., evaluation-only) while still logging metrics.
- **FR-011**: The system MUST provide deterministic seed control for reproducibility of trials where feasible.
- **FR-012**: Auditability: users MUST be able to reconstruct training curves and trial outcomes from tracking data even after artifact pruning.
- **FR-013**: The environment for training/HPO MUST be portable across machines using a containerized setup and documented mount/device requirements.
- **FR-014**: Failure handling MUST be explicit: if retention cannot keep at least one resume-capable checkpoint and the best model, the job must stop with a detailed error message containing: current disk usage, space needed for the next checkpoint, a list of the largest artifacts (with sizes and paths), and actionable commands to manually clean artifacts or adjust the retention policy.
- **FR-015**: The user-specified optimization metric (used to determine the "best" checkpoint) MUST be explicitly configured, validated, and recorded alongside outputs in the trial JSON report.
- **FR-016**: All external dependencies, including the MLflow API and Hugging Face models, MUST be pinned to exact versions via Poetry (`poetry.lock`) as the source of truth. For Docker builds, an exported `requirements.txt` may be used for performance, but it MUST be generated from `poetry.lock` and kept in sync.
- **FR-017**: When the experiment tracking backend is unreachable, the system MUST buffer metrics to disk; MUST automatically replay buffered metrics with exponential backoff retry (FR-005: delays 1s, 2s, 4s, 8s, 16s) when the backend becomes available; MUST retain the buffer file until successful upload is confirmed; MUST warn when buffered metrics exceed 100MB; MUST NOT impose a hard limit that would block training progress.
- **FR-018**: The system MUST monitor available disk space and trigger proactive retention pruning when available space drops below 10% of total disk capacity; MUST attempt aggressive pruning before failing the job.
- **FR-019**: The system MUST load all data (training, validation, and test) from a single Hugging Face dataset using a user-specified dataset identifier with appropriate split names (split="train", split="validation", split="test"), ensuring strict separation between splits.
- **FR-020**: The system MUST emit warnings and errors to both a structured JSON log file (machine-readable, with timestamp, severity, context) and human-readable stdout; critical errors MUST be immediately visible in stdout.
- **FR-021**: HPO trials MUST execute sequentially (one trial at a time), completing each trial's training, evaluation, and artifact cleanup before starting the next trial. *Note: This is to ensure deterministic behavior and reproducibility, which is harder to guarantee with parallel execution.*
- **FR-022**: The system MUST enforce a minimum checkpoint interval of one epoch (one complete pass through the training data) to prevent excessive storage churn and I/O overhead.
- **FR-023**: The system MUST require the user to specify a single optimization metric name (e.g., "accuracy", "val_f1") to maximize when determining the best checkpoint; the metric MUST be validated to exist in logged metrics; higher values are always considered better.
- **FR-024**: The system MUST use atomic checkpoint writes (write to temporary file, then atomic rename) to prevent partial/corrupted checkpoints from being created during interruptions.

// Additional requirements to address identified gaps
- **FR-025**: Checkpoint compatibility and versioning. Checkpoints MUST embed code version, model architecture signature, and head configurations. On resume, the system MUST validate compatibility (model shapes/types, head types). If incompatible (e.g., code version change that alters shapes), the system MUST fail with an actionable error and fallback to the last compatible checkpoint if available. Optional converters MAY be invoked if configured, but silent loading of incompatible checkpoints is prohibited.
- **FR-026**: Dataset identifier validation and revision pinning. The dataset identifier (default: `irlab-udc/redsm5`) MUST be validated at startup. Required splits (train/validation/test) MUST exist (configurable mapping allowed). If `revision` is set (default: `main`), the exact revision/tag/commit MUST be used and logged. The resolved dataset revision/hash MUST be logged to MLflow and included in evaluation reports.
- **FR-027**: Deterministic seeding scope. Seeds MUST be applied across Python (`random`), NumPy, PyTorch CPU and CUDA, and DataLoader workers. All seeds MUST be recorded in config, MLflow, and per-trial reports.
- **FR-028**: Aggressive pruning policy quantification. When available disk drops below 10% (Principle II), the system MUST reduce retention pressure by applying steps sequentially until sufficient space is freed: (1) prune all non-protected checkpoints within the current trial; (2) if step 1 fails to free required space, reduce to `keep_best_k=1` and `keep_last_n=0` for subsequent trials; (3) if steps 1-2 fail to free required space, preserve only the single best checkpoint across the entire study as a last resort. Metrics MUST remain fully available; co-best checkpoints and non-best trial checkpoints may be pruned under this mode.
- **FR-029**: Dataset loading failure handling. If dataset identifier is invalid or required splits are missing/corrupted, the system MUST abort with an actionable error containing: attempted dataset id and revision, required splits, detected splits, and instructions to correct config or pin a revision.
- **FR-030**: Zero-checkpoint scenarios. If interruption occurs before any checkpoint is created (e.g., during the first epoch), resume MUST start from epoch 0 (initial state) with no metric duplication; retention invariants MUST still hold.
- **FR-031**: Concurrent interruption during resume. Resume operations MUST be idempotent; if interrupted during resume, subsequent resumes MUST re-validate checkpoint integrity and avoid partial state writes (use lock files or atomic state updates), guaranteeing no duplicated metrics.
- **FR-032**: Authentication failure handling and log sanitization. Hugging Face authentication failures (missing/expired token) MUST produce actionable errors (how to login, link to docs). When a Hugging Face authentication token expires during a long-running HPO study, the system MUST pause the study, emit a WARNING notification to both JSON log and stdout, and poll for token validity every 5 minutes (check via `huggingface_hub.HfFolder.get_token()` and validate with a lightweight API call); once token is valid, the study MUST resume automatically from the paused state without user intervention. Secrets/tokens MUST NOT appear in logs; log sanitization MUST mask sensitive patterns using regex: (1) Hugging Face tokens: `hf_[A-Za-z0-9]{20,}` → `hf_***REDACTED***`, (2) API keys: `[A-Za-z0-9_-]{32,}` → `***REDACTED***`, (3) Bearer tokens: `Bearer [A-Za-z0-9_-]+` → `Bearer ***REDACTED***`, (4) Passwords in URLs: `://[^:]+:([^@]+)@` → `://user:***@`, (5) Email addresses: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` → `***@***.***`. Sanitization MUST apply to all log outputs (JSON logs, stdout, error messages, MLflow tags).
- **FR-033**: HPO progress observability. The system MUST emit trial-level progress signals (trial index/total, completion rate, best-so-far), and ETA where possible, to both MLflow (params/tags) and JSON logs.
- **FR-034**: Preflight storage checks for large models. Before training starts and before each checkpoint save, the system MUST estimate checkpoint size using the formula: `estimated_size = model.state_dict() byte size + optimizer.state_dict() byte size + metadata overhead (1MB)`. The system MUST calculate required free space given the retention policy (keep_last_n + keep_best_k checkpoints). If predicted to exceed available capacity even after aggressive pruning, the system MUST abort with an actionable error (current usage, projected checkpoint size, required free space, largest artifacts, and remediation options).
- **FR-035** (Optional): The system MAY generate a study-level summary JSON report that aggregates metrics across all trials and references the best trial's evaluation report. This is supplementary to per-study evaluation reports and aids in cross-study comparison.
- **FR-036**: The system MUST implement the requirements outlined in the `## Compliance` section.
- **FR-037**: The system MUST connect to a remote MLflow tracking server using a secure, authenticated connection (e.g., over HTTPS/TLS). All communication with the tracking server MUST be encrypted.
- **FR-038**: The system MUST obtain short-lived authentication tokens from a secure token service (e.g., HashiCorp Vault) to authenticate with the remote MLflow tracking server. The system MUST handle token expiration and renewal automatically.

### Key Entities *(include if feature involves data)*

- **Trial**: Unique id, search parameters, resolved config, optimization metric name, logs, artifact directory, best checkpoint reference(s), status.
- **Trial Lifecycle**: `queued` -> `preparing` -> `running` -> `finishing` -> `completed` / `failed`.
- **Checkpoint**: Trial id, step/epoch, metrics snapshot, path/id, created_at, retained flag, co_best flag (true when tied for maximum value of the optimization metric), integrity_hash (checksum for validation on resume).
- **RetentionPolicy**: keep_last_n, keep_best_k, keep_best_k_max=2, max_total_size, min_interval_epochs (minimum: 1 epoch), pruning_strategy, disk_space_threshold_percent (default: 10% of total capacity). Co-best ties may exceed cap.
- **ExperimentRun**: Tracking id(s), metrics time series, params, tags.
- **ModelSource**: Provider and model identifier(s) used for initialization.
- **DataSource**: Hugging Face dataset identifier, split name (train/validation/test), version/revision.
- **EvaluationReport**: JSON artifact with test metrics, config, checkpoint reference(s), and the optimization metric name used to determine best checkpoint.
- **EnvironmentProfile**: Containerized environment constraints and capabilities required for portability.
- **LogEvent**: Timestamp, severity (DEBUG/INFO/WARNING/ERROR/CRITICAL), message, structured context (trial_id, step, component), written to both JSON log file and human-readable stdout.

## Assumptions

- **A-001**: A typical HPO workload is assumed to be up to 1000 trials, with model sizes between 1-10GB and data sizes under 10GB. The initial model catalog includes 5 validated models (mental-bert, psychbert, clinicalbert, bert-base, roberta-base), expandable to 30+ models after validation.
- **A-002**: The environment is assumed to have Hugging Face authentication pre-configured (e.g., via `huggingface-cli login`).
- **A-003**: The MLflow experiment tracking backend is a remote, managed MLflow tracking server that requires authentication and uses encrypted communication (TLS).
- **A-004**: The containerized environment will be based on Docker.
- **A-005**: All external dependencies, including the MLflow API and Hugging Face models, will be pinned to exact versions via Poetry (`poetry.lock`) as source of truth; Docker builds may use an exported `requirements.txt` generated from the lock file.
- **A-006**: All training, validation, and test data reside in a single Hugging Face dataset (`irlab-udc/redsm5`) with standard split names (train, validation, test).
- **A-007**: Sequential trial execution is assumed, simplifying resource allocation and storage management (no concurrent trials competing for disk/memory/accelerators).
- **A-008**: The optimization metric is always maximized (higher is better); users should transform metrics accordingly (e.g., log negative loss if loss minimization is desired).
- **A-009**: A secure token service (e.g., HashiCorp Vault) is available and configured for the environment, providing short-lived tokens for MLflow authentication.

## Out of Scope

- Data collection and initial labeling.
- User interface (UI) for interacting with the model.
- Business strategy and product management decisions.

## Compliance

- **C-001 (HIPAA)**: The system MUST adhere to all relevant provisions of the Health Insurance Portability and Accountability Act (HIPAA) concerning the privacy and security of Protected Health Information (PHI). All data classified as PHI must be encrypted at rest and in transit. Access controls must be in place to ensure only authorized personnel can access PHI. Audit logs of PHI access must be maintained.
- **C-002 (GDPR)**: The system MUST adhere to the General Data Protection Regulation (GDPR). This includes provisions for data subject rights (e.g., right to access, right to be forgotten), data protection by design and by default, and requirements for data processing agreements with any third-party services.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Storage usage for checkpoints/artifacts is reduced by ≥60% versus naive “keep all” over equivalent runs, while preserving resume capability and best model(s).
- **SC-002**: 100% of interrupted training jobs resume successfully from the latest retained checkpoint in ≤2 minutes setup time.
- **SC-003**: 100% of metrics and parameters are present in the tracking system for all runs and trials; no missing time series due to pruning.
- Clarification for SC-003: "Present" means all metric series declared by the configuration and training loop are queryable from MLflow with timestamps and step indices. Verification method: compare expected metric keys against MLflow run data for each trial; any missing key constitutes failure.
- **SC-004**: For 100% of trials, a JSON evaluation report exists with required fields and reflects test-set evaluation of that trial’s best model.
- **SC-005**: 100% of runs that specify valid model ids successfully initialize models from the Hugging Face hub.
- **SC-006**: On a fresh supported machine with moderate network connectivity, the portable environment is operational and can execute a sample training within 15 minutes from setup start (including image pull, container launch, and environment initialization).
- Clarification for SC-006: "Moderate network connectivity" assumes downstream 50–100 Mbps, upstream 10–20 Mbps, and cold HF cache. If connectivity is below this range, setup time may exceed 15 minutes without failing this criterion.
- **SC-007**: Cross-machine reproducibility. Running the same trial configuration and seed on two supported machines (containerized) MUST produce matching results within tolerance (metric deltas ≤ 1e-4 absolute for scalar validation/test metrics; checkpoint integrity hashes may differ if paths embed host-specific prefixes, but model weights must be equivalent).

## Recovery & Rollback Requirements

- **RR-001 (MLflow DB recovery)**: On MLflow backend corruption or unavailability, the system MUST continue buffering metrics to disk and attempt automatic reconnection with backoff. If corruption is detected, emit actionable guidance to repair/restore the DB and do not drop buffered metrics.
- **RR-002 (Orphaned artifacts)**: The system MUST provide a cleanup utility (dry-run by default) that detects and removes orphaned checkpoints and logs not referenced by any active trial metadata, preserving all tracking data.
- **RR-003 (Partial study completion)**: If some trials fail, the study MUST be resumable; successful trials remain valid. Reporting and progress summaries MUST reflect partial completion without blocking continuation.
- **RR-004 (Disk full, cannot prune)**: If space cannot be freed to meet retention guarantees, the system MUST abort with the detailed error payload specified in FR-014 and leave tracking data intact.
- **RR-005 (Rollback to previous checkpoint)**: The system MUST document procedures to roll back a running trial to a previous checkpoint (e.g., on divergence), including CLI usage and required config changes.
- **RR-006 (Dependency rollback)**: The system MUST document procedures to roll back to a previous `poetry.lock` if a dependency upgrade fails (recorded in version control), ensuring reproducibility.
