# CI/CD Guide - Phase 31

This guide explains the Continuous Integration and Continuous Deployment infrastructure for the PSY Agents NO-AUG project.

## Overview

Phase 31 implements comprehensive CI/CD automation including:
- Automated testing on every push/PR
- Multi-environment deployment pipelines
- Docker containerization
- End-to-end integration testing

## GitHub Actions Workflows

### CI Pipeline (`.github/workflows/ci.yml`)

Runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**Jobs:**
1. **lint-and-format**: Code quality checks
   - Ruff linting
   - Black formatting
   - MyPy type checking (optional)

2. **unit-tests**: Unit test suite
   - Runs on Python 3.10 and 3.11
   - Pytest with coverage reporting
   - Uploads coverage to Codecov

3. **component-tests**: Component-level tests
   - Tests all phases (26-30)
   - Monitoring, explainability, registry, serving, deployment

4. **security-scan**: Security checks
   - Safety vulnerability scanning
   - Bandit security linting

5. **build-check**: Package build verification
   - Poetry build
   - Installation verification

### CD Pipeline (`.github/workflows/cd.yml`)

**Triggers:**
- Push to `main` → Deploy to staging
- Tags matching `v*.*.*` → Deploy to production
- Manual workflow dispatch

**Environments:**
- **Staging**: Automatic deployment from main
- **Production**: Tag-based or manual deployment

**Deployment Strategies:**
- Direct (default for staging)
- Blue-Green (default for production)
- Canary (configurable)
- Rolling (configurable)

**Features:**
- Pre-deployment testing
- Post-deployment health checks
- Automatic rollback on failure
- GitHub Release creation for tags

## Docker Configuration

### Building Images

```bash
# Build production image
make docker-build
# or
docker build -t psy-agents-noaug:latest .
```

### Docker Compose

```bash
# Start all services
make docker-up

# Stop services
make docker-down

# Run tests in container
make docker-test

# Clean up
make docker-clean
```

**Services:**
- **app**: Main application
- **mlflow**: MLflow tracking server (http://localhost:5000)
- **test**: Test runner (profile: test)

### Production Deployment

The Docker image uses multi-stage builds:
1. **Builder stage**: Installs dependencies
2. **Production stage**: Lean runtime image

Features:
- Non-root user for security
- Health checks
- Optimized layer caching
- Minimal runtime dependencies

## Integration Testing

### Running Integration Tests

```bash
# Run integration test suite
make test-integration

# Or directly
poetry run python scripts/test_integration.py
```

### What's Tested

1. **End-to-End Workflow**:
   - Model registration → versioning → deployment
   - Full lifecycle with monitoring

2. **Model Lifecycle**:
   - Development → Staging → Production promotion
   - Multi-version management
   - Canary deployments

3. **Monitoring Integration**:
   - Performance monitoring
   - Drift detection
   - Alert management

## Manual Deployment

### Using GitHub Actions UI

1. Go to **Actions** tab
2. Select **CD Deployment** workflow
3. Click **Run workflow**
4. Choose:
   - Environment (staging/production)
   - Strategy (direct/blue_green/canary/rolling)
5. Click **Run workflow**

### Local Deployment Simulation

```bash
# Create deployment config
python -c "
from psy_agents_noaug.deployment import *

config = create_deployment_config(
    deployment_id='manual_deploy_001',
    model_name='my_model',
    model_version='v1.0.0',
    environment=DeploymentEnvironment.STAGING,
    strategy=StrategyEnum.BLUE_GREEN,
)

orchestrator = DeploymentOrchestrator()
record = orchestrator.deploy(config)
print(f'Status: {record.status.value}')
"
```

## Makefile Targets

### CI/CD Targets

```bash
make test-integration    # Integration tests
make docker-build        # Build Docker image
make docker-up           # Start services
make docker-down         # Stop services
make docker-test         # Run tests in Docker
make docker-clean        # Clean Docker resources
```

### Development Targets

```bash
make lint               # Lint code
make format             # Format code
make test               # Run all tests
make test-cov           # Tests with coverage
```

## Environment Variables

### For Local Development

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export PYTHONUNBUFFERED=1
```

### For Production

Set these in your deployment environment:
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MODEL_REGISTRY_PATH`: Model registry storage path
- `DEPLOYMENT_ENV`: Environment name (staging/production)

## Monitoring Deployments

### Check Deployment Status

```python
from psy_agents_noaug.deployment import DeploymentOrchestrator

orchestrator = DeploymentOrchestrator()

# List all deployments
deployments = orchestrator.list_deployments()

# Get statistics
stats = orchestrator.get_deployment_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
```

### View Logs

```python
record = orchestrator.get_deployment("deploy_001")
for log in record.logs:
    print(log)
```

## Troubleshooting

### CI Pipeline Failures

1. **Lint failures**: Run `make format` locally
2. **Test failures**: Run `make test` locally to debug
3. **Build failures**: Check Poetry dependencies

### Docker Issues

1. **Build failures**: Check Dockerfile and .dockerignore
2. **Permission errors**: Ensure non-root user has access
3. **Network issues**: Check docker-compose network configuration

### Deployment Failures

1. **Check deployment logs**: View in GitHub Actions
2. **Review rollback**: Automatic rollback triggers on health check failures
3. **Manual rollback**: Use orchestrator.rollback_deployment()

## Best Practices

### For Development

1. Run `make lint` and `make test` before pushing
2. Write tests for new features
3. Keep dependencies up to date

### For Deployment

1. Always deploy to staging first
2. Monitor metrics post-deployment
3. Use blue-green or canary for production
4. Tag releases with semantic versioning

### For Rollback

1. Auto-rollback is enabled by default
2. Monitor health checks and error rates
3. Keep previous versions for quick rollback
4. Document rollback reasons

## Related Documentation

- Phase 26: Model Monitoring & Observability
- Phase 27: Model Explainability & Interpretability
- Phase 28: Model Registry & Versioning
- Phase 29: Model Serving & Deployment
- Phase 30: Deployment Automation & CI/CD

## Support

For issues or questions:
1. Check GitHub Actions logs
2. Review Docker logs: `docker-compose logs`
3. Check MLflow UI: http://localhost:5000
4. Review deployment records in orchestrator
