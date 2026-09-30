import argparse
import sys
import json
from dotenv import load_dotenv

from document_processing.utils.logging import setup_logging
from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.processors.section_detector import SectionDetector
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.embeddings.gemini_embeddings import GeminiEmbeddingProvider
from document_processing.pipeline.document_pipeline import DocumentPipeline, DocumentProcessingError

def main():
    parser = argparse.ArgumentParser(description="Process a research document.")
    parser.add_argument("--input", required=True, help="Path to the input PDF document")
    parser.add_argument("--output", help="Path to save the JSON output")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip generating embeddings")
    
    args = parser.parse_args()
    
    load_dotenv()
    setup_logging()
    
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
            
    pipeline = DocumentPipeline(
        loader=loader,
        section_detector=section_detector,
        chunker=chunker,
        embedder=embedder
    )
    
    try:
        chunks = pipeline.process(args.input, generate_embeddings=not args.no_embeddings)
        
        # Display summary
        print(f"\nDocument: {args.input}")
        print(f"Chunks created: {len(chunks)}")
        
        # Optionally save to JSON
        if args.output:
            # We convert pydantic models to dict, but for embeddings, they are floats.
            # We shouldn't necessarily serialize embeddings if they are huge, but user didn't say omit them from file,
            # they said "Do not include raw embedding vectors in human-readable output unless explicitly requested."
            # The stdout is human readable, the JSON output is for inspection. Let's write them if they exist.
            output_data = []
            for c in chunks:
                chunk_dict = c.model_dump()
                # To make human inspection easier, maybe truncate embedding in the json or just leave it. 
                # Let's leave it.
                output_data.append(chunk_dict)
                
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Saved processed output to {args.output}")
            
    except DocumentProcessingError as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

