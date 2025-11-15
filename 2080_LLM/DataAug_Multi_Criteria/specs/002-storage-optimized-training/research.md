# Research: Storage-Optimized Training & HPO Pipeline

**Feature**: Storage-Optimized Training & HPO Pipeline  
**Phase**: 0 (Outline & Research)  
**Date**: 2025-01-14

## Overview

This document consolidates research findings for implementing a storage-optimized training and HPO pipeline with dual-agent architecture, remote MLflow tracking with authentication, and HIPAA/GDPR compliance.

## Research Tasks

### 1. Checkpoint Management Strategies for Large-Scale HPO

**Task**: Research best practices for checkpoint retention in long-running HPO studies (1000+ trials, 1-10GB models).

**Decision**: Implement a tiered retention policy with configurable parameters:
- `keep_last_n`: Number of most recent checkpoints to retain (default: 1)
- `keep_best_k`: Number of best checkpoints by validation metric (default: 1, max: 2)
- `max_total_size`: Maximum total checkpoint storage (default: 10GB)
- `min_interval_epochs`: Minimum interval between checkpoints (default: 1 epoch)
- Proactive pruning triggers at <10% available disk space

**Rationale**:
- PyTorch Lightning's checkpoint callback system provides a proven pattern for retention policies
- MLflow's artifact logging decouples metrics from checkpoints, allowing aggressive pruning
- Tiered policies balance resume capability (keep_last_n) with best model preservation (keep_best_k)
- Epoch-based minimum intervals prevent storage churn from excessive checkpointing
- Proactive disk monitoring prevents mid-training storage exhaustion

**Alternatives Considered**:
- Time-based retention (keep checkpoints from last N hours): Rejected due to variable epoch durations
- Fixed interval checkpointing (every N steps): Rejected as it doesn't align with epoch boundaries for validation evaluation
- Unlimited checkpoint retention with manual cleanup: Rejected as it risks storage exhaustion on long runs

**Implementation References**:
- PyTorch Lightning `ModelCheckpoint` callback: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
- Ray Tune checkpoint management: https://docs.ray.io/en/latest/tune/tutorials/tune-storage.html

---

### 2. Atomic Checkpoint Writes and Resume Validation

**Task**: Research patterns for atomic checkpoint writes and integrity validation to handle interruptions safely.

**Decision**: Use atomic write pattern (write to temp file + atomic rename) combined with SHA256 checksum validation on resume.

**Rationale**:
- POSIX `os.rename()` is atomic on the same filesystem, preventing partial checkpoint writes
- SHA256 checksums provide collision-resistant integrity verification
- PyTorch's `torch.save()` with `_use_new_zipfile_serialization=True` creates robust serialized state
- On resume, validate checksum before loading; fall back to previous checkpoint if validation fails

**Alternatives Considered**:
- Database-backed checkpoint metadata: Rejected for simplicity; filesystem + MLflow tracking is sufficient
- Copy-on-write filesystems (Btrfs, ZFS snapshots): Rejected as it requires specific filesystem support
- Distributed checkpointing (DeepSpeed): Rejected as out of scope for single-machine sequential trials

**Implementation Pattern**:
```python
def save_checkpoint_atomic(checkpoint_data, path):
    temp_path = f"{path}.tmp.{uuid.uuid4()}"
    torch.save(checkpoint_data, temp_path)
    checksum = compute_sha256(temp_path)
    checkpoint_data['integrity_hash'] = checksum
    torch.save(checkpoint_data, temp_path)  # Re-save with hash
    os.rename(temp_path, path)  # Atomic
    return checksum

def load_checkpoint_validated(path):
    checkpoint = torch.load(path)
    expected_hash = checkpoint.pop('integrity_hash', None)
    if expected_hash:
        actual_hash = compute_sha256_from_checkpoint(checkpoint)
        if actual_hash != expected_hash:
            raise CheckpointCorruptionError(f"Checksum mismatch: {path}")
    return checkpoint
```

**References**:
- PyTorch checkpointing guide: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- Atomicity guarantees in POSIX: https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html

---

### 3. Remote MLflow Tracking with Authentication and TLS

**Task**: Research secure MLflow tracking patterns for HIPAA/GDPR compliance with remote authenticated servers.

**Decision**: Use remote MLflow tracking server with TLS encryption and short-lived token authentication from HashiCorp Vault.

**Rationale**:
- HIPAA and GDPR require encrypted communication (TLS) and access controls
- Short-lived tokens (TTL: 1-24 hours) minimize credential exposure risk
- HashiCorp Vault provides battle-tested secret management with automatic rotation
- MLflow tracking server supports token-based authentication via `MLFLOW_TRACKING_TOKEN` environment variable
- Disk buffering with exponential backoff retry ensures metrics aren't lost during tracking outages

**Alternatives Considered**:
- Local unauthenticated MLflow database: Rejected due to compliance requirements (C-001, C-002)
- Username/password authentication: Rejected as long-lived credentials pose security risks
- Client-side certificates (mTLS): Rejected for complexity; token-based auth is simpler and sufficient

**Implementation Pattern**:
```python
class MLflowAuthClient:
    def __init__(self, tracking_uri, vault_client):
        self.tracking_uri = tracking_uri
        self.vault = vault_client
        self.token = None
        self.token_expiry = None
    
    def get_token(self):
        if not self.token or datetime.now() > self.token_expiry:
            # Fetch from Vault
            secret = self.vault.secrets.kv.v2.read_secret(path='mlflow/token')
            self.token = secret['data']['data']['token']
            self.token_expiry = datetime.now() + timedelta(hours=1)
        return self.token
    
    def log_metric(self, run_id, key, value, step):
        os.environ['MLFLOW_TRACKING_TOKEN'] = self.get_token()
        mlflow.log_metric(key, value, step)
```

**References**:
- MLflow authentication: https://mlflow.org/docs/latest/auth/index.html
- HashiCorp Vault Python client: https://hvac.readthedocs.io/
- HIPAA Security Rule: https://www.hhs.gov/hipaa/for-professionals/security/index.html

---

### 4. Metric Buffering During Tracking Outages

**Task**: Research strategies for buffering metrics to disk when MLflow tracking server is unreachable.

**Decision**: Implement append-only JSON Lines (JSONL) buffer with automatic replay using exponential backoff retry.

**Rationale**:
- JSONL format is simple, append-only, and easily parseable
- Append-only writes are robust to interruptions (no partial JSON corruption)
- Exponential backoff (1s, 2s, 4s, 8s, 16s) prevents overwhelming the tracking server during recovery
- Buffer file retained until successful upload confirmed by MLflow API response
- 100MB warning threshold alerts users without blocking training

**Alternatives Considered**:
- SQLite-based buffer: Rejected for simplicity; JSONL is sufficient and more portable
- In-memory buffer with periodic flush: Rejected as it risks data loss on process crashes
- Synchronous blocking until tracking recovers: Rejected as it would halt training during outages

**Implementation Pattern**:
```python
class MetricsBuffer:
    def __init__(self, buffer_path='metrics_buffer.jsonl'):
        self.buffer_path = buffer_path
        self.buffer_size = 0
    
    def buffer_metric(self, run_id, key, value, step, timestamp):
        entry = {'run_id': run_id, 'key': key, 'value': value, 
                 'step': step, 'timestamp': timestamp}
        with open(self.buffer_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        self.buffer_size += len(json.dumps(entry))
        if self.buffer_size > 100 * 1024 * 1024:  # 100MB
            logger.warning(f"Buffer exceeds 100MB: {self.buffer_size / 1e6:.2f} MB")
    
    def replay_buffer(self, mlflow_client):
        with open(self.buffer_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                retry_with_backoff(
                    lambda: mlflow_client.log_metric(
                        entry['run_id'], entry['key'], 
                        entry['value'], entry['step'], entry['timestamp']
                    )
                )
        os.remove(self.buffer_path)
        self.buffer_size = 0
```

**References**:
- JSON Lines format: https://jsonlines.org/
- Exponential backoff best practices: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

---

### 5. Hugging Face Model and Dataset Loading with Retry Logic

**Task**: Research best practices for loading models/datasets from Hugging Face with cache-first strategy and retry logic.

**Decision**: Implement cache-first loading with exponential backoff retry (5 attempts, delays: 1s, 2s, 4s, 8s, 16s).

**Rationale**:
- `transformers` library caches models locally in `~/.cache/huggingface/hub/` by default
- Cache lookup is fast and avoids network calls when models are pre-downloaded
- Exponential backoff handles rate limiting (HTTP 429) and transient network issues
- 5 retries with max delay of 16s provides ~30s total retry window, sufficient for most transient issues
- Actionable error messages guide users to manually download if network is unavailable

**Alternatives Considered**:
- Fixed retry intervals: Rejected as they don't account for rate limiting backoff requirements
- Unlimited retries: Rejected as it could cause indefinite hangs on persistent network issues
- No retries, fail immediately: Rejected as it's too fragile for production environments

**Implementation Pattern**:
```python
def load_model_with_retry(model_id, retries=5):
    for attempt in range(retries):
        try:
            # Cache lookup first
            model = AutoModel.from_pretrained(
                model_id,
                cache_dir=os.environ.get('HF_HOME', '~/.cache/huggingface')
            )
            return model
        except (HTTPError, ConnectionError, Timeout) as e:
            if attempt < retries - 1:
                delay = 2 ** attempt  # Exponential: 1, 2, 4, 8, 16
                logger.warning(f"Model load attempt {attempt+1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
            else:
                raise ModelLoadError(
                    f"Failed to load model '{model_id}' after {retries} attempts. "
                    f"Check network connectivity or manually download: "
                    f"huggingface-cli download {model_id}"
                )
```

**References**:
- Transformers model loading: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
- Datasets library caching: https://huggingface.co/docs/datasets/cache

---

### 6. Dual-Agent Architecture with Shared Encoder

**Task**: Research patterns for training dual-agent models (criteria matching + evidence binding) with shared encoder.

**Decision**: Implement multi-task learning with shared BERT-based encoder and task-specific heads.

**Rationale**:
- Shared encoder learns common representations between criteria matching (classification) and evidence binding (span extraction)
- PyTorch's `nn.ModuleDict` allows flexible head management for different tasks
- Joint training with weighted loss enables balancing between tasks
- Separate metric logging for each agent enables independent evaluation

**Alternatives Considered**:
- Separate models for each agent: Rejected as it duplicates encoder parameters and doesn't leverage shared representations
- Sequential training (train one agent, freeze, train other): Rejected as it's less effective than joint multi-task learning
- Hard parameter sharing only in lower layers: Rejected for simplicity; full encoder sharing is standard for BERT-based multi-task models

**Implementation Pattern**:
```python
class DualAgentModel(nn.Module):
    def __init__(self, encoder_name, num_criteria_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            'criteria_matcher': nn.Linear(self.encoder.config.hidden_size, num_criteria_classes),
            'evidence_binder': nn.Linear(self.encoder.config.hidden_size, 2)  # start/end logits
        })
    
    def forward(self, input_ids, attention_mask, task):
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = encoder_output.pooler_output if task == 'criteria_matcher' else encoder_output.last_hidden_state
        return self.heads[task](pooled)
```

**References**:
- Multi-task learning with BERT: https://arxiv.org/abs/1901.11504
- PyTorch multi-task patterns: https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html

---

### 7. Per-Study Test Evaluation Strategy

**Task**: Research best practices for test set evaluation in large-scale HPO to prevent overfitting.

**Decision**: Implement per-study test evaluation where only the best model from the entire study is evaluated on the test set once after all trials complete.

**Rationale**:
- Evaluating 1000 trials on the test set would leak test set information and overfit
- Per-trial validation evaluation guides HPO optimization
- Single per-study test evaluation on the best model provides unbiased generalization estimate
- Study-level JSON report records test metrics, config, and checkpoint reference for reproducibility

**Alternatives Considered**:
- Per-trial test evaluation: Rejected due to test set overfitting risk with 1000+ trials
- Holdout validation set + final test: Implemented as chosen approach
- Cross-validation: Rejected for computational cost (1000 trials already expensive)

**Implementation Pattern**:
```python
def evaluate_best_model_on_test(study, test_dataset):
    best_trial = study.best_trial
    checkpoint = load_checkpoint(best_trial.user_attrs['best_checkpoint_path'])
    
    model = DualAgentModel.from_checkpoint(checkpoint)
    test_metrics = evaluate_model(model, test_dataset)
    
    report = {
        'study_id': study.study_name,
        'best_trial_id': best_trial.number,
        'test_metrics': test_metrics,
        'config': best_trial.params,
        'checkpoint_reference': best_trial.user_attrs['best_checkpoint_path']
    }
    
    with open(f'experiments/study_{study.study_name}/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report
```

**References**:
- Optuna best practices: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
- Test set best practices: https://machinelearningmastery.com/difference-test-validation-datasets/

---

### 8. Docker Environment with Poetry for Reproducibility

**Task**: Research best practices for containerized ML environments with Poetry dependency management.

**Decision**: Use multi-stage Docker build with Poetry for dependency installation and runtime image optimization.

**Rationale**:
- Multi-stage builds keep final image size small by excluding Poetry build tools
- Poetry lock file (`poetry.lock`) ensures exact dependency versions across environments
- Docker layer caching speeds up rebuilds when dependencies don't change
- `.devcontainer` configuration enables VS Code Remote Containers for consistent development

**Alternatives Considered**:
- pip with requirements.txt: Rejected as Poetry provides better dependency resolution and lock files
- Conda in Docker: Rejected for larger image sizes and slower environment creation
- Docker + virtualenv: Rejected as containers already provide isolation; Poetry is sufficient

**Implementation Pattern**:
```dockerfile
# Multi-stage build
FROM python:3.10-slim as builder

RUN pip install poetry==1.7.1
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10-slim as runtime

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
COPY . .
ENV PYTHONPATH=/workspace/src

CMD ["python", "-m", "dataaug_multi_both.cli.train"]
```

**References**:
- Poetry in Docker: https://python-poetry.org/docs/faq/#i-want-to-use-poetry-in-a-docker-container
- Multi-stage builds: https://docs.docker.com/build/building/multi-stage/

---

### 9. HIPAA and GDPR Compliance for ML Training

**Task**: Research compliance requirements for HIPAA and GDPR in ML training pipelines.

**Decision**: Implement encryption at rest and in transit, access controls, audit logging, and data subject rights support.

**Rationale**:
- HIPAA requires encryption of PHI at rest and in transit (Security Rule 164.312)
- GDPR requires data protection by design, access controls, and audit trails (Articles 25, 32)
- TLS for MLflow tracking satisfies encryption in transit
- Filesystem encryption (LUKS, dm-crypt) or cloud provider encryption satisfies encryption at rest
- MLflow audit logs track all metric/artifact access
- Log sanitization prevents accidental leakage of PII/PHI in logs

**Alternatives Considered**:
- Application-level encryption of each artifact: Rejected for performance overhead; filesystem encryption is sufficient
- No audit logging: Rejected as it violates compliance requirements
- Manual log scrubbing: Rejected as it's error-prone; automated regex-based sanitization is required

**Compliance Checklist**:
- ✅ Encryption in transit: TLS for MLflow (FR-037)
- ✅ Encryption at rest: Filesystem/cloud encryption (deployment requirement)
- ✅ Access controls: Token-based authentication (FR-038)
- ✅ Audit logging: MLflow tracking logs (FR-003)
- ✅ Log sanitization: Automated regex masking (FR-032)
- ✅ Data subject rights: Manual procedures for access/deletion (deployment SOP)

**References**:
- HIPAA Security Rule: https://www.hhs.gov/hipaa/for-professionals/security/index.html
- GDPR Article 32: https://gdpr-info.eu/art-32-gdpr/

---

### 10. Hugging Face Token Expiration Handling

**Task**: Research patterns for handling Hugging Face token expiration during long-running HPO studies.

**Decision**: Implement token validation polling with automatic pause/resume when token expires.

**Rationale**:
- Hugging Face tokens can expire during multi-day 1000-trial studies
- Polling every 5 minutes using `huggingface_hub.HfFolder.get_token()` + lightweight API validation is non-intrusive
- Pausing trials (not starting new checkpoints) prevents partial work that can't complete
- Automatic resume after token refresh (user runs `huggingface-cli login`) avoids manual study restart

**Alternatives Considered**:
- Fail immediately on token expiration: Rejected as it loses study progress
- Continue without token (skip model downloads): Rejected as it causes cryptic failures downstream
- Refresh token automatically: Rejected as Hugging Face doesn't support programmatic token refresh without user credentials

**Implementation Pattern**:
```python
def validate_hf_token():
    token = HfFolder.get_token()
    if not token:
        return False
    try:
        # Lightweight API call to validate token
        hf_hub_download(repo_id='gpt2', filename='config.json', token=token, cache_dir='/tmp/validate')
        return True
    except Exception:
        return False

def run_hpo_with_token_monitoring(study):
    while not study.is_complete():
        if not validate_hf_token():
            logger.warning("Hugging Face token invalid/expired. Pausing study. Run 'huggingface-cli login' to resume.")
            while not validate_hf_token():
                time.sleep(300)  # Poll every 5 minutes
            logger.info("Hugging Face token validated. Resuming study.")
        
        trial = study.ask()
        run_trial(trial)
```

**References**:
- Hugging Face Hub authentication: https://huggingface.co/docs/huggingface_hub/quick-start#authentication

---

## Summary

All research tasks have been completed with concrete decisions, rationale, and implementation patterns. Key decisions include:

1. **Checkpoint Management**: Tiered retention policy with proactive disk monitoring
2. **Resume Safety**: Atomic writes + SHA256 validation
3. **MLflow Authentication**: Remote server with Vault-based short-lived tokens
4. **Metric Buffering**: JSONL append-only buffer with exponential backoff replay
5. **HF Loading**: Cache-first with exponential backoff retry (5 attempts)
6. **Dual-Agent**: Multi-task learning with shared BERT encoder
7. **Test Evaluation**: Per-study test set evaluation (once per HPO study)
8. **Docker**: Multi-stage builds with Poetry for reproducibility
9. **Compliance**: TLS + filesystem encryption + access controls + audit logs
10. **Token Expiration**: Polling with pause/resume for long-running studies

These decisions resolve all "NEEDS CLARIFICATION" items from the technical context and provide implementation guidance for Phase 1 design.
