#!/usr/bin/env python3
"""
Simple script to run the healthcare document ingestion pipeline.
"""

import os
import sys
from pathlib import Path


def main():
    """Run the ingestion pipeline."""

    # Configuration
    PDF_MD_DIR = Path(__file__).parent.parent / "Insurance" / "pdf_md"

    if not PDF_MD_DIR.exists():
        print(f"Error: Directory not found: {PDF_MD_DIR}")
        print(
            "Please ensure the Insurance/pdf_md directory exists with your markdown files."
        )
        sys.exit(1)

    # Check for .env file
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        print("Warning: .env file not found.")
        print("Please create .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_key_here")
        print()

    # Import and run ingestion
    try:
        from ingestion_pipeline import HealthcareDocumentIngestion

        print("Healthcare Document Ingestion Pipeline")
        print("=" * 40)
        print(f"Processing directory: {PDF_MD_DIR}")
        print()

        # Initialize pipeline
        ingestion = HealthcareDocumentIngestion()

        # Run ingestion
        results = ingestion.ingest_directory(str(PDF_MD_DIR))

        # Print results
        print("\n" + "=" * 50)
        print("INGESTION RESULTS")
        print("=" * 50)
        print(f"Total files: {results['total_files']}")
        print(f"Processed: {results['processed_files']}")
        print(f"Failed: {results['failed_files']}")
        print(f"Total chunks: {results['total_chunks']}")

        if results["failed_file_list"]:
            print("\nFailed files:")
            for file in results["failed_file_list"]:
                print(f"  - {file}")

        print("\nIngestion completed!")

        if results["processed_files"] > 0:
            print("\nYou can now search your documents using:")
            print(
                "  python -c \"from ingestion import HealthcareDocumentIngestion; ingestion = HealthcareDocumentIngestion(); print(ingestion.search_documents('your query here'))\""
            )

    except ImportError as e:
        print(f"Error importing ingestion pipeline: {e}")
        print("Please install requirements: pip install -r ../requirements_new.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running ingestion: {e}")
        sys.exit(1)


main()
