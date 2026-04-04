#!/bin/bash

# AstrAI Docker Script
# Build and manage Docker images

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="astrai"
IMAGE_TAG="latest"
REGISTRY=""

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker version: $(docker --version)"
}

# Build Docker image
build_image() {
    local dockerfile="${1:-Dockerfile}"
    local context="${2:-.}"

    if [ ! -f "$dockerfile" ]; then
        print_error "Dockerfile not found: $dockerfile"
        exit 1
    fi

    print_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "$dockerfile" "$context"
    print_success "Image built successfully"
}

# Run container
run_container() {
    local port="${1:-8000}"
    local gpu="${2:-false}"

    print_info "Running container on port $port..."

    if [ "$gpu" = true ]; then
        docker run --gpus all -p "${port}:8000" "${IMAGE_NAME}:${IMAGE_TAG}"
    else
        docker run -p "${port}:8000" "${IMAGE_NAME}:${IMAGE_TAG}"
    fi
}

# Push image to registry
push_image() {
    if [ -z "$REGISTRY" ]; then
        print_error "Registry not set. Use --registry option"
        exit 1
    fi

    local full_tag="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    print_info "Tagging image: ${full_tag}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "$full_tag"

    print_info "Pushing image to registry..."
    docker push "$full_tag"
    print_success "Image pushed successfully"
}

# Remove image
remove_image() {
    print_info "Removing image: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker rmi "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null || print_warning "Image not found"
    print_success "Image removed"
}

# Show image info
show_info() {
    print_info "Image information:"
    docker images "${IMAGE_NAME}"
}

# Show logs
show_logs() {
    local container_id="$1"
    if [ -z "$container_id" ]; then
        print_error "Container ID required"
        exit 1
    fi
    docker logs "$container_id"
}

# Main function
main() {
    echo "========================================"
    echo "      AstrAI Docker Management"
    echo "========================================"
    echo ""

    COMMAND=""
    DOCKERFILE="Dockerfile"
    CONTEXT="."
    PORT="8000"
    GPU=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            build)
                COMMAND="build"
                shift
                ;;
            run)
                COMMAND="run"
                shift
                ;;
            push)
                COMMAND="push"
                shift
                ;;
            remove|rm)
                COMMAND="remove"
                shift
                ;;
            info)
                COMMAND="info"
                shift
                ;;
            logs)
                COMMAND="logs"
                shift
                ;;
            --image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --dockerfile)
                DOCKERFILE="$2"
                shift 2
                ;;
            --context)
                CONTEXT="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --gpu)
                GPU=true
                shift
                ;;
            --help)
                echo "Usage: $0 <command> [options]"
                echo ""
                echo "Commands:"
                echo "  build     Build Docker image"
                echo "  run       Run container"
                echo "  push      Push image to registry"
                echo "  remove    Remove image"
                echo "  info      Show image information"
                echo "  logs      Show container logs"
                echo ""
                echo "Options:"
                echo "  --image NAME       Image name (default: astrai)"
                echo "  --tag TAG          Image tag (default: latest)"
                echo "  --registry URL     Registry URL for push"
                echo "  --dockerfile FILE  Dockerfile path (default: Dockerfile)"
                echo "  --context PATH     Build context (default: .)"
                echo "  --port PORT        Port for run (default: 8000)"
                echo "  --gpu              Enable GPU support"
                echo "  --help             Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 build"
                echo "  $0 build --tag v1.0.0"
                echo "  $0 run --port 8080"
                echo "  $0 run --gpu"
                echo "  $0 push --registry ghcr.io/username"
                exit 0
                ;;
            *)
                if [ -z "$COMMAND" ]; then
                    print_error "Unknown command: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    check_docker

    case "$COMMAND" in
        build)
            build_image "$DOCKERFILE" "$CONTEXT"
            ;;
        run)
            run_container "$PORT" "$GPU"
            ;;
        push)
            push_image
            ;;
        remove)
            remove_image
            ;;
        info)
            show_info
            ;;
        logs)
            show_logs "$2"
            ;;
        "")
            print_error "No command specified. Use --help for usage"
            exit 1
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
}

main "$@"
