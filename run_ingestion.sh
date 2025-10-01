#!/bin/bash

# Healthcare Document Ingestion Pipeline - Root Launcher
# =====================================================
# This script provides easy access to the ingestion pipeline from the root directory.

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Healthcare Document Ingestion Pipeline${NC}"
echo "======================================"
echo

# Check if we're in the right directory
if [ ! -d "ingestion" ]; then
    echo "Error: ingestion directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Navigate to ingestion directory and run the main script
cd ingestion
./run.sh "$@"
