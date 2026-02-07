#!/bin/bash
# ==============================================================================
# Container Runtime Setup for HPC
# ==============================================================================
# This script provides functions for setting up Docker/Podman container runtimes
# on HPC clusters. It handles:
#   - Docker daemon setup via setup_docker_runtime.sh
#   - podman-hpc shim creation for Docker API compatibility
#   - Container runtime verification
#
# Usage:
#   source /path/to/hpc/shell_utils/container_runtime.sh
#   setup_container_runtime "docker"    # or "podman_hpc" or "daytona"
#
# Exit codes:
#   0 - Success
#   4 - EXIT_CONNECTIVITY_FAILED (Docker daemon not responding)
# ==============================================================================

# Create a docker shim that wraps podman-hpc
# This allows Harbor's Docker backend to work with podman-hpc
install_podman_hpc_shim() {
    local workdir="${1:-$WORKDIR}"
    local bin_dir="$workdir/.bin"

    if command -v docker &>/dev/null; then
        echo "[container_runtime] docker CLI already available, skipping shim"
        return 0
    fi

    if ! command -v podman-hpc &>/dev/null; then
        echo "[container_runtime] WARNING: podman-hpc not found, cannot create shim"
        return 1
    fi

    mkdir -p "$bin_dir"
    cat > "$bin_dir/docker" <<'SHIM_EOF'
#!/usr/bin/env bash
exec podman-hpc "$@"
SHIM_EOF
    chmod +x "$bin_dir/docker"
    export PATH="$bin_dir:$PATH"
    echo "[container_runtime] Installed podman-hpc docker shim at $bin_dir/docker"
}

# Verify container runtime connectivity
# Returns 0 on success, 4 on failure
verify_container_connectivity() {
    local timeout_sec="${1:-10}"

    if [ -z "${DOCKER_HOST:-}" ]; then
        echo "[container_runtime] WARNING: DOCKER_HOST not set"
        return 0  # Not a fatal error, might work without it
    fi

    echo "[container_runtime] DOCKER_HOST=$DOCKER_HOST"
    echo "[container_runtime] CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-unknown}"

    # Try available container CLIs in order of preference
    if command -v docker &>/dev/null && timeout "$timeout_sec" docker info &>/dev/null; then
        echo "[container_runtime] Verified successfully (docker info)"
        return 0
    elif command -v podman-hpc &>/dev/null && timeout "$timeout_sec" podman-hpc info &>/dev/null; then
        echo "[container_runtime] Verified successfully (podman-hpc info)"
        return 0
    elif command -v podman &>/dev/null && timeout "$timeout_sec" podman info &>/dev/null; then
        echo "[container_runtime] Verified successfully (podman info)"
        return 0
    fi

    echo "[container_runtime] ERROR: Container daemon not responding at DOCKER_HOST=$DOCKER_HOST"
    echo "[container_runtime] HINT: Check 'podman-hpc info' or 'docker info' manually"
    return 4  # EXIT_CONNECTIVITY_FAILED
}

# Main setup function - call this from sbatch scripts
# Args:
#   $1 - harbor_env type: "docker", "podman_hpc", "daytona", "modal", etc.
#   $2 - workdir (optional, defaults to $WORKDIR)
setup_container_runtime() {
    local harbor_env="$1"
    local workdir="${2:-$WORKDIR}"

    case "$harbor_env" in
        docker)
            echo "[container_runtime] Setting up Docker runtime..."

            # Source Docker runtime setup if available
            if [ -f "$workdir/docker/setup_docker_runtime.sh" ]; then
                # shellcheck source=docker/setup_docker_runtime.sh
                source "$workdir/docker/setup_docker_runtime.sh"
            fi

            # Install podman-hpc shim if needed
            install_podman_hpc_shim "$workdir"

            # Verify connectivity
            if ! verify_container_connectivity; then
                return 4
            fi
            ;;

        podman_hpc)
            # podman-hpc is native on HPC clusters - no special setup needed
            echo "[container_runtime] Using native podman-hpc (no Docker setup needed)"
            ;;

        daytona|modal|beam)
            # Cloud backends don't need local container setup
            echo "[container_runtime] Using cloud backend: $harbor_env (no local container setup)"
            ;;

        *)
            echo "[container_runtime] Unknown harbor_env: $harbor_env (skipping container setup)"
            ;;
    esac

    return 0
}
