#!/bin/bash
# ============================================================================
# Docker build script with optimizations
# PSY Agents NO-AUG Baseline Repository
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

print_header "PSY Agents NO-AUG Docker Build"

# Build arguments
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=$(poetry version -s 2>/dev/null || echo "0.1.0")

echo "Build Configuration:"
echo "  Date:    ${BUILD_DATE}"
echo "  Git:     ${VCS_REF}"
echo "  Version: ${VERSION}"
echo ""

# Parse command line arguments
SKIP_BUILDER=false
SKIP_RUNTIME=false
NO_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-builder)
            SKIP_BUILDER=true
            shift
            ;;
        --skip-runtime)
            SKIP_RUNTIME=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-builder    Skip building the builder stage"
            echo "  --skip-runtime    Skip building the runtime stage"
            echo "  --no-cache        Build without using cache"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build options
BUILD_OPTS="--progress=plain"
if [ "$NO_CACHE" = true ]; then
    BUILD_OPTS="$BUILD_OPTS --no-cache"
    print_warning "Building without cache (this will be slower)"
else
    BUILD_OPTS="$BUILD_OPTS --cache-from psy-agents-noaug:builder --cache-from psy-agents-noaug:latest"
fi

# ============================================================================
# Build Builder Stage
# ============================================================================
if [ "$SKIP_BUILDER" = false ]; then
    print_header "Building Builder Stage"

    START_TIME=$(date +%s)

    if docker build \
        --target builder \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${VCS_REF}" \
        --build-arg VERSION="${VERSION}" \
        --tag psy-agents-noaug:builder \
        $BUILD_OPTS \
        .; then

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        print_success "Builder stage completed in ${DURATION}s"
    else
        print_error "Builder stage failed"
        exit 1
    fi
else
    print_warning "Skipping builder stage"
fi

# ============================================================================
# Build Runtime Stage
# ============================================================================
if [ "$SKIP_RUNTIME" = false ]; then
    print_header "Building Runtime Stage"

    START_TIME=$(date +%s)

    if docker build \
        --target runtime \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${VCS_REF}" \
        --build-arg VERSION="${VERSION}" \
        --tag psy-agents-noaug:${VERSION} \
        --tag psy-agents-noaug:latest \
        $BUILD_OPTS \
        .; then

        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        print_success "Runtime stage completed in ${DURATION}s"
    else
        print_error "Runtime stage failed"
        exit 1
    fi
else
    print_warning "Skipping runtime stage"
fi

# ============================================================================
# Summary
# ============================================================================
print_header "Build Summary"

echo "Images created:"
if [ "$SKIP_BUILDER" = false ]; then
    echo "  - psy-agents-noaug:builder (dev/testing)"
fi
if [ "$SKIP_RUNTIME" = false ]; then
    echo "  - psy-agents-noaug:${VERSION} (production)"
    echo "  - psy-agents-noaug:latest (production)"
fi
echo ""

echo "Image sizes:"
docker images psy-agents-noaug --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo ""

print_success "Build complete!"
echo ""
echo "Next steps:"
echo "  1. Run tests:        make docker-test"
echo "  2. Start runtime:    make docker-run"
echo "  3. Open shell:       make docker-shell"
echo "  4. View UI:          docker-compose -f docker-compose.test.yml --profile mlflow up"
echo ""
