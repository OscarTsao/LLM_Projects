# CI/CD Workflows Quick Reference

## Workflow Triggers Summary

| Workflow | Auto (Push/PR) | Schedule | Manual |
|----------|----------------|----------|--------|
| **CI** | ✓ (main, develop) | - | - |
| **Security** | ✓ (main) | ✓ (Weekly Mon) | ✓ |
| **Benchmark** | ✓ (main) | - | ✓ |

## Quick Commands

### Manual Workflow Trigger (GitHub CLI)
```bash
# Security scan
gh workflow run security.yml

# Performance benchmark
gh workflow run benchmark.yml

# On specific branch
gh workflow run security.yml --ref develop
```

### Local Security Scans
```bash
# Bandit
poetry run bandit -r src/ -ll -i

# pip-audit
poetry run pip install pip-audit
poetry run pip-audit --desc

# Generate SBOM
poetry run pip install pipdeptree
poetry run pipdeptree

# License report
poetry run pip install pip-licenses
poetry run pip-licenses --with-urls --format=markdown
```

## Artifacts Generated

### CI Workflow
- `bandit_report.json`
- `pip_audit_report.json`
- Retention: 30 days

### Security Workflow
- `bandit_report.json` & `.txt`
- `pip_audit_report.json`
- `SBOM.json` & `.txt`
- `THIRD_PARTY_LICENSES.md`
- `licenses_summary.json`
- Retention: 90 days

### Benchmark Workflow
- `benchmark_*.json`
- `profiling_*.csv`
- Retention: 30 days

## Key Features

### CI Workflow (`ci.yml`)
- Matrix: Python 3.10, 3.11
- Caching: Poetry dependencies
- Non-blocking security scans
- Coverage upload to Codecov

### Security Workflow (`security.yml`)
- Weekly automated scans
- SBOM generation
- License compliance reports
- **BLOCKS** on critical vulnerabilities

### Benchmark Workflow (`benchmark.yml`)
- DataLoader throughput testing
- Augmentation profiling
- Non-blocking (all steps)

## Viewing Results

### GitHub UI
1. Go to **Actions** tab
2. Select workflow run
3. View **Summary** for formatted results
4. Download artifacts from **Artifacts** section

### Codecov
- Visit: https://codecov.io/gh/{org}/{repo}
- View coverage trends and reports

## Troubleshooting

### Workflow Not Running
- Check branch name matches trigger pattern
- Verify `.github/workflows/` directory permissions
- Check YAML syntax with: `yamllint .github/workflows/*.yml`

### Security Scan Failures
- Review Bandit report: Download `bandit_report.json`
- Check pip-audit: Download `pip_audit_report.json`
- Run locally for detailed output

### Missing Artifacts
- Check if workflow completed successfully
- Verify artifact retention period hasn't expired
- Ensure upload step executed without errors

## See Full Documentation
For comprehensive details, see: `docs/CI_CD_SECURITY_WORKFLOWS.md`
