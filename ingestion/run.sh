#!/bin/bash

# Healthcare Document Ingestion Pipeline - Run Script
# ===================================================
# This script provides an easy way to run the healthcare document ingestion pipeline
# with proper environment setup and error handling.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_status "Using Python $PYTHON_VERSION"
}

# Function to check if .env file exists
check_env_file() {
    ENV_FILE="../.env"
    if [ ! -f "$ENV_FILE" ]; then
        print_warning ".env file not found in parent directory"
        print_status "Creating .env file from template..."
        
        if [ -f "env_example.txt" ]; then
            cp env_example.txt "$ENV_FILE"
            print_success "Created .env file from template"
            print_warning "Please edit .env file with your actual values:"
            print_warning "  - OPENAI_API_KEY (required)"
            print_warning "  - S3_BUCKET (optional)"
            print_warning "  - AWS credentials (optional)"
            echo
        else
            print_error "env_example.txt not found"
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

# Function to check if Qdrant is running
check_qdrant() {
    print_status "Checking Qdrant connection..."
    
    if command_exists curl; then
        if curl -s http://localhost:6333/collections >/dev/null 2>&1; then
            print_success "Qdrant is running and accessible"
        else
            print_warning "Qdrant is not running or not accessible"
            print_status "To start Qdrant, run:"
            print_status "  docker run -p 6333:6333 qdrant/qdrant"
            echo
            read -p "Do you want to continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_status "Exiting. Please start Qdrant and try again."
                exit 1
            fi
        fi
    else
        print_warning "curl not available, skipping Qdrant check"
    fi
}

# Function to install requirements
install_requirements() {
    print_status "Checking requirements..."
    
    if [ -f "requirements_ingestion.txt" ]; then
        print_status "Installing ingestion requirements..."
        $PYTHON_CMD -m pip install -r requirements_ingestion.txt
        
        # Try to install spaCy model
        print_status "Installing spaCy English model..."
        if $PYTHON_CMD -m spacy download en_core_web_sm >/dev/null 2>&1; then
            print_success "SpaCy model installed successfully"
        else
            print_warning "SpaCy model installation failed (optional)"
        fi
    else
        print_error "requirements_ingestion.txt not found"
        exit 1
    fi
}



# Function to run ingestion
run_ingestion() {
    print_status "Starting healthcare document ingestion..."
    echo
    
    if [ -f "run_ingestion.py" ]; then
        $PYTHON_CMD run_ingestion.py
        print_success "Ingestion completed"
    else
        print_error "run_ingestion.py not found"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "Healthcare Document Ingestion Pipeline - Run Script"
    echo "=================================================="
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -t, --test     Run tests only"
    echo "  -s, --setup    Run setup only (install requirements)"
    echo "  -i, --ingest   Run ingestion only (skip tests)"
    echo "  -a, --all      Run full pipeline (setup + test + ingest)"
    echo
    echo "Examples:"
    echo "  $0              # Run full pipeline"
    echo "  $0 --test       # Run tests only"
    echo "  $0 --setup      # Setup only"
    echo "  $0 --ingest     # Ingestion only"
    echo
}

# Main function
main() {
    echo "Healthcare Document Ingestion Pipeline"
    echo "======================================"
    echo
    
    # Parse command line arguments
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--test)
            check_python
            run_tests
            exit 0
            ;;
        -s|--setup)
            check_python
            install_requirements
            exit 0
            ;;
        -i|--ingest)
            check_python
            check_env_file
            check_qdrant
            run_ingestion
            exit 0
            ;;
        -a|--all)
            # Run full pipeline
            ;;
        "")
            # No arguments, run full pipeline
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    
    # Run full pipeline
    print_status "Running full ingestion pipeline..."
    echo
    
    # Step 1: Check Python
    check_python
    
    # Step 2: Install requirements
    install_requirements
    
    # Step 3: Check environment
    check_env_file
    
    # Step 4: Check Qdrant
    check_qdrant
    
    
    # Step 6: Run ingestion
    run_ingestion
    
    echo
    print_success "Pipeline completed successfully!"
    echo
    print_status "You can now search your documents using:"
    print_status "  $PYTHON_CMD -c \"from ingestion import HealthcareDocumentIngestion; ingestion = HealthcareDocumentIngestion(); print(ingestion.search_documents('your query here'))\""
}

# Run main function with all arguments
main "$@"
