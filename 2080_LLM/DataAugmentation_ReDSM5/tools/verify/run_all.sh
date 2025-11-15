#!/bin/bash
# Comprehensive verification orchestration script
# Runs test suite, benchmarks, and generates reports

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

# Output files
LOG_FILE="${PROJECT_ROOT}/verification.log"
TEST_RESULTS="${PROJECT_ROOT}/test_results.json"
BENCH_RESULTS="${PROJECT_ROOT}/tools/verify/bench_results.json"
SUMMARY_JSON="${PROJECT_ROOT}/verification_summary.json"
REPORT_MD="${PROJECT_ROOT}/verification_report.md"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

# Print banner
print_banner() {
    echo "======================================================================" | tee -a "${LOG_FILE}"
    echo "DSM-5 Data Augmentation Pipeline - Verification Suite" | tee -a "${LOG_FILE}"
    echo "======================================================================" | tee -a "${LOG_FILE}"
    echo "Started: $(date)" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install Python 3."
        exit 1
    fi

    # Check for pytest
    if ! python3 -c "import pytest" 2>/dev/null; then
        log_warning "pytest not found. Installing..."
        pip install pytest pytest-json-report
    fi

    # Check for pytest-json-report
    if ! python3 -c "import pytest_jsonreport" 2>/dev/null; then
        log_warning "pytest-json-report not found. Installing..."
        pip install pytest-json-report
    fi

    log_success "All dependencies available"
}

# Clean previous outputs
clean_previous() {
    log_info "Cleaning previous outputs..."

    # Remove temporary directories
    rm -rf /tmp/verify_* 2>/dev/null || true

    # Remove previous result files
    rm -f "${TEST_RESULTS}" 2>/dev/null || true
    rm -f "${BENCH_RESULTS}" 2>/dev/null || true
    rm -f "${SUMMARY_JSON}" 2>/dev/null || true
    rm -f "${REPORT_MD}" 2>/dev/null || true
    rm -f "${LOG_FILE}" 2>/dev/null || true

    log_success "Cleanup complete"
}

# Run test suite
run_tests() {
    log_info "Running pytest verification suite..."
    echo "" | tee -a "${LOG_FILE}"

    local exit_code=0

    if pytest tests/verify/ -v --tb=short --json-report --json-report-file="${TEST_RESULTS}" 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Test suite completed successfully"
    else
        exit_code=$?
        log_warning "Test suite completed with failures (exit code: ${exit_code})"
    fi

    echo "" | tee -a "${LOG_FILE}"
    return ${exit_code}
}

# Run benchmarks
run_benchmarks() {
    log_info "Running micro-benchmarks..."
    echo "" | tee -a "${LOG_FILE}"

    local exit_code=0

    if python3 "${SCRIPT_DIR}/bench_small.py" 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Benchmarks completed successfully"
    else
        exit_code=$?
        log_warning "Benchmarks completed with failures (exit code: ${exit_code})"
    fi

    echo "" | tee -a "${LOG_FILE}"
    return ${exit_code}
}

# Generate reports
generate_reports() {
    log_info "Generating verification reports..."
    echo "" | tee -a "${LOG_FILE}"

    local exit_code=0

    if python3 "${SCRIPT_DIR}/generate_report.py" 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Reports generated successfully"
    else
        exit_code=$?
        log_error "Report generation failed (exit code: ${exit_code})"
    fi

    echo "" | tee -a "${LOG_FILE}"
    return ${exit_code}
}

# Print summary
print_summary() {
    echo "" | tee -a "${LOG_FILE}"
    echo "======================================================================" | tee -a "${LOG_FILE}"
    echo "Verification Summary" | tee -a "${LOG_FILE}"
    echo "======================================================================" | tee -a "${LOG_FILE}"

    # Check if summary exists
    if [ -f "${SUMMARY_JSON}" ]; then
        # Extract status from summary JSON
        local overall_status=$(python3 -c "import json; print(json.load(open('${SUMMARY_JSON}'))['overall'])" 2>/dev/null || echo "UNKNOWN")

        if [ "${overall_status}" = "PASS" ]; then
            log_success "Overall Status: PASS"
        else
            log_error "Overall Status: ${overall_status}"
        fi

        echo "" | tee -a "${LOG_FILE}"
        echo "Generated Files:" | tee -a "${LOG_FILE}"
        echo "  - Test Results:    ${TEST_RESULTS}" | tee -a "${LOG_FILE}"
        echo "  - Benchmark Results: ${BENCH_RESULTS}" | tee -a "${LOG_FILE}"
        echo "  - Summary JSON:    ${SUMMARY_JSON}" | tee -a "${LOG_FILE}"
        echo "  - Report MD:       ${REPORT_MD}" | tee -a "${LOG_FILE}"
        echo "  - Log File:        ${LOG_FILE}" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"

        # Print quick stats from summary
        echo "Quick Stats:" | tee -a "${LOG_FILE}"
        python3 -c "
import json
with open('${SUMMARY_JSON}') as f:
    data = json.load(f)
    tests = data['tests']
    perf = data['performance']
    print(f\"  Tests: {tests['passed']}/{tests['total']} passed\")
    if perf.get('cpu_rows_per_sec'):
        print(f\"  CPU Throughput: {perf['cpu_rows_per_sec']:.1f} rows/sec\")
    if perf.get('disk_cache_speedup'):
        print(f\"  Cache Speedup: {perf['disk_cache_speedup']:.2f}x\")
    if perf.get('multiprocessing_speedup'):
        print(f\"  MP Speedup: {perf['multiprocessing_speedup']:.2f}x\")
" 2>/dev/null | tee -a "${LOG_FILE}"

    else
        log_error "Summary file not found"
    fi

    echo "" | tee -a "${LOG_FILE}"
    echo "Completed: $(date)" | tee -a "${LOG_FILE}"
    echo "======================================================================" | tee -a "${LOG_FILE}"
}

# Main execution
main() {
    local test_exit=0
    local bench_exit=0
    local report_exit=0

    # Print banner
    print_banner

    # Check dependencies
    check_dependencies || exit 1

    # Clean previous outputs
    clean_previous

    # Run test suite (continue on failure)
    run_tests || test_exit=$?

    # Run benchmarks (continue on failure)
    run_benchmarks || bench_exit=$?

    # Generate reports (must succeed)
    generate_reports || report_exit=$?

    # Print summary
    print_summary

    # Determine overall exit code
    if [ ${test_exit} -ne 0 ] || [ ${bench_exit} -ne 0 ] || [ ${report_exit} -ne 0 ]; then
        log_warning "Verification completed with some failures"
        return 1
    else
        log_success "All verification steps passed"
        return 0
    fi
}

# Run main and capture exit code
main
exit $?
