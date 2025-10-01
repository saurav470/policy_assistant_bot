"""
Healthcare Insurance Document Ingestion Pipeline
===============================================

This script handles the ingestion of insurance policy documents into Qdrant vector database
with optimal chunking strategies for healthcare documents.

Features:
- UIN number extraction and metadata enrichment
- Healthcare-optimized chunking strategy
- S3 upload for PDF files
- Qdrant vector storage with LangChain
- Text and metadata storage for efficient retrieval
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

# Core libraries
import boto3
from botocore.exceptions import ClientError
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# LangChain imports
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
)
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader

# Additional utilities
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HealthcareDocumentIngestion:
    """
    Healthcare document ingestion pipeline with optimized chunking for insurance policies.
    """

    def __init__(
        self,
        qdrant_url: str = None,
        collection_name: str = "healthcare_insurance",
        s3_bucket: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Name of the Qdrant collection
            s3_bucket: S3 bucket name for PDF storage
            openai_api_key: OpenAI API key for embeddings
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Initialize clients
        self._init_clients()

        # Initialize text splitters for healthcare documents
        self._init_text_splitters()

        # UIN pattern for insurance documents
        self.uin_pattern = re.compile(
            r"[A-Z]{5}[A-Z0-9]{2}[A-Z0-9]{2}[A-Z0-9]{2}[A-Z0-9]{2}[A-Z0-9]{2}[A-Z0-9]{2}"
        )

    def _init_clients(self):
        """Initialize Qdrant and S3 clients."""
        try:
            # Initialize Qdrant client
            print(self.qdrant_url)
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            logger.info(f"Connected to Qdrant at {self.qdrant_url}")

            # Initialize OpenAI embeddings
            if self.openai_api_key:
                self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                logger.info("OpenAI embeddings initialized")
            else:
                logger.warning("OpenAI API key not provided. Using default embeddings.")
                self.embeddings = None

            # Initialize S3 client if bucket is provided
            if self.s3_bucket:
                self.s3_client = boto3.client("s3")
                logger.info(f"S3 client initialized for bucket: {self.s3_bucket}")
            else:
                self.s3_client = None
                logger.warning("S3 bucket not provided. PDF uploads will be skipped.")

        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise

    def _init_text_splitters(self):
        """Initialize text splitters optimized for healthcare documents."""

        # Healthcare-specific separators for better chunking
        healthcare_separators = [
            "\n\n## ",  # Markdown headers
            "\n\n**",  # Bold text (often section headers)
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentence endings
            "! ",  # Exclamation
            "? ",  # Question
            "; ",  # Semicolon
            ", ",  # Comma
            " ",  # Space
            "",  # Character level
        ]

        # Recursive character splitter for healthcare documents
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            separators=healthcare_separators,
            chunk_size=1000,  # Optimal for healthcare context
            chunk_overlap=200,  # Good overlap for context preservation
            length_function=len,
            is_separator_regex=False,
        )

        # Token-based splitter for precise control
        self.token_splitter = TokenTextSplitter(
            chunk_size=800,  # Slightly smaller for token precision
            chunk_overlap=150,
            model_name="gpt-3.5-turbo",  # For token counting
        )

        # Try to initialize spaCy splitter for sentence-aware splitting
        try:
            self.spacy_splitter = None
            
            # SpacyTextSplitter(
            #     pipeline="en_core_web_sm", chunk_size=1000, chunk_overlap=200
            # )
            logger.info("SpaCy text splitter initialized")
        except Exception as e:
            logger.warning(f"SpaCy splitter not available: {e}")
            self.spacy_splitter = None

    def extract_uin_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract UIN number from filename.

        Args:
            filename: The filename to extract UIN from

        Returns:
            UIN string if found, None otherwise
        """
        # Look for UIN pattern in filename
        uin = filename.split("_")[0]
        if uin:
            return uin.strip()

        return None

    def extract_uin_from_content(self, content: str) -> List[str]:
        """
        Extract all UIN numbers from document content.

        Args:
            content: Document content to search

        Returns:
            List of UIN numbers found
        """
        uins = self.uin_pattern.findall(content)
        return list(set(uins))  # Remove duplicates

    def create_healthcare_metadata(
        self, content: str, filename: str, uin: Optional[str] = None, chunk_text: str = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata for healthcare documents.

        Args:
            content: Document content
            filename: Source filename
            uin: UIN number if available
            chunk_text: Chunk text (if creating metadata for a chunk)

        Returns:
            Metadata dictionary
        """
        # Extract UIN from filename if not provided
        print("Creating metadata...", filename)
        if not uin:
            uin = self.extract_uin_from_filename(filename)

        # Extract additional UINs from content
        content_uins = self.extract_uin_from_content(content)

        # Use chunk text if provided, otherwise use full content
        text_to_store = chunk_text if chunk_text else content

        # Create metadata
        metadata = {
            "source": filename,
            "uin": uin,
            "all_uins": content_uins,
            "document_type": "insurance_policy",
            "domain": "healthcare",
            "ingestion_date": datetime.now().isoformat(),
            "content_length": len(content),
            "chunk_index": 0,  # Will be updated per chunk
            "total_chunks": 0,  # Will be updated after chunking
            "text": text_to_store,  # Store chunk text or full content in metadata for retrieval
            "file_hash": hashlib.md5(content.encode()).hexdigest(),
        }

        # Extract policy-specific information
        policy_info = self._extract_policy_info(content)
        metadata.update(policy_info)

        return metadata

    def _extract_policy_info(self, content: str) -> Dict[str, Any]:
        """
        Extract policy-specific information from content.

        Args:
            content: Document content

        Returns:
            Dictionary with extracted policy information
        """
        policy_info = {}

        # Extract policy number patterns
        policy_number_patterns = [
            r"Policy\s*Number[:\s]+([A-Z0-9\s]+)",
            r"Policy\s*No[:\s]+([A-Z0-9\s]+)",
            r"Policy\s*#\s*([A-Z0-9\s]+)",
        ]

        for pattern in policy_number_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                policy_info["policy_number"] = match.group(1).strip()
                break

        # Extract sum insured
        sum_insured_patterns = [
            r"Sum\s*Insured[:\s]+([Rs\.\s\d,]+)",
            r"Coverage\s*Amount[:\s]+([Rs\.\s\d,]+)",
            r"SI[:\s]+([Rs\.\s\d,]+)",
        ]

        for pattern in sum_insured_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                policy_info["sum_insured"] = match.group(1).strip()
                break

        # Extract provider name
        provider_patterns = [
            r"Care\s*Health\s*Insurance",
            r"Religare\s*Health\s*Insurance",
            r"HDFC\s*ERGO",
            r"Star\s*Health",
        ]

        for pattern in provider_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                policy_info["provider"] = pattern.replace(r"\s+", " ")
                break

        return policy_info

    def chunk_healthcare_document(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Chunk healthcare document with optimal strategy.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            List of chunked documents
        """
        # Create base document
        base_doc = Document(page_content=content, metadata=metadata)

        # Try different chunking strategies and choose the best one
        chunking_strategies = []

        # Strategy 1: Recursive character splitter (best for healthcare)
        try:
            recursive_chunks = self.recursive_splitter.split_documents([base_doc])
            chunking_strategies.append(("recursive", recursive_chunks))
        except Exception as e:
            logger.warning(f"Recursive chunking failed: {e}")

        # Strategy 2: Token-based splitter
        try:
            token_chunks = self.token_splitter.split_documents([base_doc])
            chunking_strategies.append(("token", token_chunks))
        except Exception as e:
            logger.warning(f"Token chunking failed: {e}")

        # Strategy 3: SpaCy splitter (if available)
        if self.spacy_splitter:
            try:
                spacy_chunks = self.spacy_splitter.split_documents([base_doc])
                chunking_strategies.append(("spacy", spacy_chunks))
            except Exception as e:
                logger.warning(f"SpaCy chunking failed: {e}")

        # Choose the best strategy based on chunk quality
        best_strategy = self._select_best_chunking_strategy(chunking_strategies)

        if not best_strategy:
            logger.error("All chunking strategies failed")
            return []

        strategy_name, chunks = best_strategy
        logger.info(
            f"Using {strategy_name} chunking strategy with {len(chunks)} chunks"
        )

        # Update metadata for each chunk
        for i, chunk in enumerate(chunks):
            # Create chunk-specific metadata with chunk text
            chunk_metadata = self.create_healthcare_metadata(
                content=metadata.get("text", ""),  # Use original content for UIN extraction
                filename=metadata.get("source", ""),
                uin=metadata.get("uin"),
                chunk_text=chunk.page_content  # Store chunk text instead of full content
            )
            
            # Update chunk metadata with chunk-specific information
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunking_strategy": strategy_name,
                    "chunk_size": len(chunk.page_content),
                }
            )
            
            # Update the chunk's metadata
            chunk.metadata = chunk_metadata

        return chunks

    def _select_best_chunking_strategy(
        self, strategies: List[Tuple[str, List[Document]]]
    ) -> Optional[Tuple[str, List[Document]]]:
        """
        Select the best chunking strategy based on chunk quality metrics.

        Args:
            strategies: List of (strategy_name, chunks) tuples

        Returns:
            Best strategy tuple or None
        """
        if not strategies:
            return None

        best_strategy = None
        best_score = -1

        for strategy_name, chunks in strategies:
            if not chunks:
                continue

            # Calculate quality score
            score = self._calculate_chunk_quality_score(chunks)

            if score > best_score:
                best_score = score
                best_strategy = (strategy_name, chunks)

        return best_strategy

    def _calculate_chunk_quality_score(self, chunks: List[Document]) -> float:
        """
        Calculate quality score for chunks based on healthcare document characteristics.

        Args:
            chunks: List of document chunks

        Returns:
            Quality score (higher is better)
        """
        if not chunks:
            return 0.0

        scores = []

        for chunk in chunks:
            content = chunk.page_content

            # Score based on chunk size (prefer 500-1500 characters)
            size_score = 1.0
            if len(content) < 200:
                size_score = 0.3  # Too small
            elif len(content) > 2000:
                size_score = 0.7  # Too large

            # Score based on healthcare-specific content
            healthcare_keywords = [
                "policy",
                "coverage",
                "benefit",
                "claim",
                "hospitalization",
                "premium",
                "sum insured",
                "exclusion",
                "waiting period",
                "pre-existing",
                "copay",
                "deductible",
                "network",
            ]

            keyword_score = sum(
                1
                for keyword in healthcare_keywords
                if keyword.lower() in content.lower()
            ) / len(healthcare_keywords)

            # Score based on sentence completeness
            sentence_score = 1.0
            if content.count(".") < 2:  # Should have multiple sentences
                sentence_score = 0.5

            # Combined score
            chunk_score = size_score * 0.4 + keyword_score * 0.4 + sentence_score * 0.2
            scores.append(chunk_score)

        return sum(scores) / len(scores) if scores else 0.0

    def upload_pdf_to_s3(self, file_path: str, uin: str) -> Optional[str]:
        """
        Upload PDF file to S3 with organized folder structure.

        Args:
            file_path: Path to the PDF file
            uin: UIN number for naming

        Returns:
            S3 object key if successful, None otherwise
        """
        if not self.s3_client or not self.s3_bucket:
            logger.warning("S3 not configured, skipping PDF upload")
            return None

        try:
            # Create organized folder structure based on UIN
            file_extension = Path(file_path).suffix
            filename = Path(file_path).name

            # Extract provider and policy type from UIN for better organization
            provider_folder = self._get_provider_folder(uin)
            policy_type_folder = self._get_policy_type_folder(uin)

            # Create S3 key with organized folder structure
            s3_key = f"healthcare-insurance/{provider_folder}/{policy_type_folder}/{uin}/{filename}"

            # Ensure the folder structure exists by creating a placeholder
            self._ensure_s3_folder_exists(s3_key)

            # Upload file
            self.s3_client.upload_file(file_path, self.s3_bucket, s3_key)

            logger.info(f"Uploaded {file_path} to S3: s3://{self.s3_bucket}/{s3_key}")
            return s3_key

        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {e}")
            return None

    def _get_provider_folder(self, uin: str) -> str:
        """
        Extract provider folder name from UIN.

        Args:
            uin: UIN number

        Returns:
            Provider folder name
        """
        if uin.startswith("CHIHLIP") or uin.startswith("CHIHLIA"):
            return "care-health-insurance"
        elif uin.startswith("RHIHLIP"):
            return "religare-health-insurance"
        elif uin.startswith("HDFC"):
            return "hdfc-ergo"
        elif uin.startswith("STAR"):
            return "star-health"
        else:
            return "other-providers"

    def _get_policy_type_folder(self, uin: str) -> str:
        """
        Extract policy type folder name from UIN.

        Args:
            uin: UIN number

        Returns:
            Policy type folder name
        """
        if "SHIELD" in uin.upper() or uin.startswith("CHIHLIA"):
            return "add-on-policies"
        elif "ADVANTAGE" in uin.upper():
            return "advantage-policies"
        elif "CARE" in uin.upper():
            return "care-policies"
        else:
            return "standard-policies"

    def _ensure_s3_folder_exists(self, s3_key: str):
        """
        Ensure S3 folder structure exists by creating a placeholder file.

        Args:
            s3_key: S3 object key
        """
        try:
            # Extract folder path from s3_key
            folder_path = "/".join(s3_key.split("/")[:-1])

            # Create a placeholder file to ensure folder exists
            placeholder_key = f"{folder_path}/.folder_placeholder"

            # Check if placeholder already exists
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=placeholder_key)
                # Placeholder exists, no need to create
            except ClientError:
                # Placeholder doesn't exist, create it
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=placeholder_key,
                    Body="This folder contains healthcare insurance documents",
                    ContentType="text/plain",
                )
                logger.debug(f"Created S3 folder placeholder: {placeholder_key}")

        except Exception as e:
            logger.warning(f"Could not create S3 folder placeholder: {e}")
            # Don't fail the upload if folder creation fails

    def create_s3_folder_structure(self, uin: str) -> str:
        """
        Create organized S3 folder structure for a given UIN.

        Args:
            uin: UIN number

        Returns:
            S3 folder path
        """
        if not self.s3_client or not self.s3_bucket:
            logger.warning("S3 not configured, skipping folder creation")
            return None

        try:
            # Extract provider and policy type from UIN
            provider_folder = self._get_provider_folder(uin)
            policy_type_folder = self._get_policy_type_folder(uin)

            # Create folder path
            folder_path = (
                f"healthcare-insurance/{provider_folder}/{policy_type_folder}/{uin}"
            )

            # Create folder structure with placeholder
            placeholder_key = f"{folder_path}/.folder_placeholder"

            # Check if folder already exists
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=placeholder_key)
                logger.info(f"S3 folder already exists: {folder_path}")
            except ClientError:
                # Create folder with placeholder
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=placeholder_key,
                    Body=f"This folder contains documents for UIN: {uin}",
                    ContentType="text/plain",
                )
                logger.info(f"Created S3 folder structure: {folder_path}")

            return folder_path

        except Exception as e:
            logger.error(f"Error creating S3 folder structure: {e}")
            return None

    def create_qdrant_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536, distance=Distance.COSINE  # OpenAI embedding size
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            raise

    def ingest_document(self, file_path: str, pdf_path: Optional[str] = None) -> bool:
        """
        Ingest a single document into the vector database.

        Args:
            file_path: Path to the markdown file
            pdf_path: Optional path to the corresponding PDF file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the markdown file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            filename = Path(file_path).name

            # Extract UIN and create metadata
            uin = self.extract_uin_from_filename(filename)
            metadata = self.create_healthcare_metadata(content, filename, uin)

            logger.info(f"Processing document: {filename} (UIN: {uin})")

            # Create S3 folder structure and upload PDF if provided
            s3_key = None
            s3_folder = None

            if uin:
                # Create organized folder structure in S3
                s3_folder = self.create_s3_folder_structure(uin)
                if s3_folder:
                    metadata["s3_folder"] = s3_folder

            if pdf_path and os.path.exists(pdf_path):
                s3_key = self.upload_pdf_to_s3(pdf_path, uin)
                if s3_key:
                    metadata["s3_key"] = s3_key

            # Chunk the document
            chunks = self.chunk_healthcare_document(content, metadata)

            if not chunks:
                logger.error(f"No chunks created for {filename}")
                return False

            # Generate embeddings and store in Qdrant
            if self.embeddings:
                # Use LangChain Qdrant integration
                vectorstore = Qdrant(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embeddings=self.embeddings,
                )

                # Add documents to vectorstore
                vectorstore.add_documents(chunks)
                logger.info(
                    f"Successfully ingested {len(chunks)} chunks for {filename}"
                )

            else:
                logger.warning("No embeddings available, skipping vector storage")

            return True

        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return False

    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all markdown files from a directory.

        Args:
            directory_path: Path to directory containing markdown files

        Returns:
            Dictionary with ingestion results
        """
        directory = Path(directory_path)

        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return {"success": False, "error": "Directory not found"}

        # Find all markdown files
        md_files = list(directory.glob("*.md"))

        if not md_files:
            logger.warning(f"No markdown files found in {directory_path}")
            return {"success": False, "error": "No markdown files found"}

        logger.info(f"Found {len(md_files)} markdown files to ingest")

        # Create Qdrant collection
        self.create_qdrant_collection()

        # Process each file
        results = {
            "success": True,
            "total_files": len(md_files),
            "processed_files": 0,
            "failed_files": 0,
            "failed_file_list": [],
            "total_chunks": 0,
        }

        for md_file in md_files:
            # Look for corresponding PDF file
            pdf_file = md_file.with_suffix(".pdf")
            pdf_path = str(pdf_file) if pdf_file.exists() else None

            success = self.ingest_document(str(md_file), pdf_path)

            if success:
                results["processed_files"] += 1
                # Count chunks (approximate)
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = self.chunk_healthcare_document(content, {})
                results["total_chunks"] += len(chunks)
            else:
                results["failed_files"] += 1
                results["failed_file_list"].append(str(md_file))

        logger.info(
            f"Ingestion completed: {results['processed_files']}/{results['total_files']} files processed"
        )
        return results

    def search_documents(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents in the vector database.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        try:
            if not self.embeddings:
                logger.error("No embeddings available for search")
                return []

            # Create vectorstore
            vectorstore = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )

            # Perform similarity search
            if filter_metadata:
                print(f"Applying metadata filter: {filter_metadata}")
                results = vectorstore.similarity_search_with_score(
                    query, k=limit, filter=filter_metadata
                )
            else:
                results = vectorstore.similarity_search_with_score(query, k=limit)

            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                }
                formatted_results.append(result)
            print(f"Search results for query '{query}': {formatted_results}")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []


def main():
    """Main function to run the ingestion pipeline."""

    # Configuration
    PDF_MD_DIR = Path(__file__).parent.parent / "Insurance" / "pdf_md"
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "healthcare_insurance"
    S3_BUCKET = os.getenv("S3_BUCKET")  # Set this in your .env file

    # Initialize ingestion pipeline
    ingestion = HealthcareDocumentIngestion(
        qdrant_url=QDRANT_URL, collection_name=COLLECTION_NAME, s3_bucket=S3_BUCKET
    )

    # Run ingestion
    logger.info("Starting healthcare document ingestion...")
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


if __name__ == "__main__":
    main()
