"""
Healthcare Document Ingestion Package
====================================

This package provides comprehensive healthcare document ingestion capabilities
for insurance policy documents with optimized chunking strategies.

Main Components:
- HealthcareDocumentIngestion: Main ingestion pipeline class
- Text splitters optimized for healthcare documents
- UIN extraction and metadata enrichment
- S3 integration for PDF storage
- Qdrant vector database integration

Usage:
    from ingestion import HealthcareDocumentIngestion

    ingestion = HealthcareDocumentIngestion()
    results = ingestion.ingest_directory("/path/to/documents")
"""

from .ingestion_pipeline import HealthcareDocumentIngestion

__version__ = "1.0.0"
__author__ = "Healthcare Bot Team"
__all__ = ["HealthcareDocumentIngestion"]
