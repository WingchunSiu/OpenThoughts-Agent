#!/usr/bin/env bash
#
# Build and push OpenThoughts Docker images to GitHub Container Registry (ghcr.io)
#
# Usage:
#   ./docker/build_and_push.sh              # Build and push all images
#   ./docker/build_and_push.sh --build-only # Build without pushing
#   ./docker/build_and_push.sh --push-only  # Push previously built images
#   ./docker/build_and_push.sh gpu-1x       # Build/push specific image
#
# Prerequisites:
#   1. Docker installed and running
#   2. Authenticated to ghcr.io:
#      echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# GitHub Container Registry settings
REGISTRY="ghcr.io"
ORG="open-thoughts"
IMAGE_NAME="openthoughts-agent"
IMAGE_BASE="${REGISTRY}/${ORG}/${IMAGE_NAME}"

# Available image variants
VARIANTS=("gpu-1x" "gpu-4x" "gpu-8x")

# Parse arguments
BUILD=true
PUSH=true
SELECTED_VARIANTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            PUSH=false
            shift
            ;;
        --push-only)
            BUILD=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--build-only|--push-only] [variant...]"
            echo ""
            echo "Options:"
            echo "  --build-only    Build images without pushing"
            echo "  --push-only     Push previously built images"
            echo ""
            echo "Variants: ${VARIANTS[*]}"
            echo ""
            echo "Examples:"
            echo "  $0                    # Build and push all"
            echo "  $0 gpu-1x             # Build and push gpu-1x only"
            echo "  $0 --build-only       # Build all without pushing"
            exit 0
            ;;
        gpu-*)
            SELECTED_VARIANTS+=("$1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to all variants if none selected
if [[ ${#SELECTED_VARIANTS[@]} -eq 0 ]]; then
    SELECTED_VARIANTS=("${VARIANTS[@]}")
fi

# Get git commit for tagging
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "============================================"
echo "OpenThoughts Docker Build"
echo "============================================"
echo "Registry:  ${REGISTRY}"
echo "Image:     ${ORG}/${IMAGE_NAME}"
echo "Variants:  ${SELECTED_VARIANTS[*]}"
echo "Git SHA:   ${GIT_SHA}"
echo "Build:     ${BUILD}"
echo "Push:      ${PUSH}"
echo "============================================"

cd "$REPO_ROOT"

for variant in "${SELECTED_VARIANTS[@]}"; do
    dockerfile="docker/Dockerfile.${variant}"

    if [[ ! -f "$dockerfile" ]]; then
        echo "ERROR: Dockerfile not found: $dockerfile"
        exit 1
    fi

    image_tag="${IMAGE_BASE}:${variant}"
    image_tag_sha="${IMAGE_BASE}:${variant}-${GIT_SHA}"

    if [[ "$BUILD" == "true" ]]; then
        echo ""
        echo ">>> Building ${variant}..."
        docker build \
            --platform linux/amd64 \
            -f "$dockerfile" \
            -t "$image_tag" \
            -t "$image_tag_sha" \
            .
        echo ">>> Built: ${image_tag}"
    fi

    if [[ "$PUSH" == "true" ]]; then
        echo ""
        echo ">>> Pushing ${variant}..."
        docker push "$image_tag"
        docker push "$image_tag_sha"
        echo ">>> Pushed: ${image_tag}"
    fi
done

echo ""
echo "============================================"
echo "Done!"
echo ""
echo "Images available:"
for variant in "${SELECTED_VARIANTS[@]}"; do
    echo "  ${IMAGE_BASE}:${variant}"
done
echo "============================================"
