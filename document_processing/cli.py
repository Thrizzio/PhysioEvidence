import argparse
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from document_processing.utils.logging import setup_logging
from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.processors.section_detector import SectionDetector
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.embeddings.gemini_embeddings import GeminiEmbeddingProvider
from document_processing.pipeline.document_pipeline import DocumentPipeline, DocumentProcessingError
from storage.db import init_db
from storage.postgres_store import PostgresVectorStore

def main():
    parser = argparse.ArgumentParser(description="PhysioEvidence Document Ingestion Pipeline.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to a single input PDF document")
    input_group.add_argument("--input-dir", help="Path to a directory of PDF documents for batch ingestion")
    
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories when using --input-dir")
    parser.add_argument("--output", help="Path to save the JSON output (for single document)")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip generating vector embeddings")
    parser.add_argument("--save-db", action="store_true", help="Persist documents and chunks to PostgreSQL + pgvector")
    parser.add_argument("--init-db", action="store_true", help="Initialize PostgreSQL schema before running ingestion")
    
    args = parser.parse_args()
    
    load_dotenv()
    setup_logging()

    # Handle schema initialization if requested
    if args.init_db:
        try:
            print("Initializing database schema...")
            init_db()
            print("Database initialized successfully.")
        except Exception as e:
            print(f"Database initialization failed: {e}")
            sys.exit(1)
            
    # Initialize components
    loader = PDFLoader()
    section_detector = SectionDetector()
    chunker = SectionAwareChunker()
    
    embedder = None
    if not args.no_embeddings:
        try:
            embedder = GeminiEmbeddingProvider()
        except ValueError as e:
            print(f"Error initializing embedder: {e}")
            print("To run without embeddings, use the --no-embeddings flag.")
            sys.exit(1)
            
    vector_store = None
    if args.save_db:
        try:
            vector_store = PostgresVectorStore()
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            sys.exit(1)

    pipeline = DocumentPipeline(
        loader=loader,
        section_detector=section_detector,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store
    )
    
    try:
        if args.input:
            chunks = pipeline.process(args.input, generate_embeddings=not args.no_embeddings)
            print(f"\nDocument: {args.input}")
            print(f"Chunks created: {len(chunks)}")
            if args.save_db:
                print(f"Saved to database: yes")
            
            if args.output:
                output_data = [c.model_dump() for c in chunks]
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"Saved processed output to {args.output}")

        elif args.input_dir:
            summary = pipeline.process_directory(
                args.input_dir,
                generate_embeddings=not args.no_embeddings,
                recursive=args.recursive
            )
            print("\n=== Batch Ingestion Summary ===")
            print(f"Total PDFs found: {summary['total_files']}")
            print(f"Successfully processed: {len(summary['successful_files'])}")
            print(f"Failed: {len(summary['failed_files'])}")
            print(f"Total chunks created: {summary['total_chunks']}")
            print(f"Duration: {summary['duration_seconds']}s")

            if summary["failed_files"]:
                print("\nFailures:")
                for fail in summary["failed_files"]:
                    print(f"  - {fail['file']}: {fail['error']}")

    except DocumentProcessingError as e:
        print(f"\nPipeline Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
