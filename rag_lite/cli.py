"""
CLI interface for RAG Lite.
Features interactive REPL, similarity score inspection, chunk attribution, and abstention reporting.
"""

import argparse
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .engine import RAGPipeline
from .models import RAGResponse
from .embeddings import LocalSemanticEmbeddingProvider
from .providers import MockGenerator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def render_response(resp: RAGResponse):
    """Render RAG response, confidence score, and source chunk attributions."""
    if HAS_RICH and console:
        console.print()
        if resp.is_abstained:
            console.print(Panel(
                f"[bold red]{resp.answer}[/bold red]",
                title="[bold red]⚠️ RAG Guardrail Triggered (Abstention)[/bold red]",
                subtitle=f"Latency: {resp.latency_ms}ms | Confidence: 0.00",
                border_style="red"
            ))
            console.print()
            return

        # Grounded Answer
        score_color = "green" if resp.confidence_score >= 0.60 else "yellow"
        console.print(Panel(
            Text(resp.answer, style="bold white"),
            title=f"[bold green]💬 Grounded Answer[/bold green]",
            subtitle=f"Top Similarity: [{score_color}]{resp.confidence_score:.4f}[/{score_color}] | Latency: {resp.latency_ms}ms",
            border_style="green"
        ))
        console.print()

        # Attributed Sources Table
        if resp.sources:
            table = Table(title="📚 Retrieved Context Chunks (Source Attribution)", show_header=True, header_style="bold magenta")
            table.add_column("Rank", width=6, justify="center")
            table.add_column("Document & Chunk ID", style="cyan", width=26)
            table.add_column("Cosine Similarity", justify="center", width=18)
            table.add_column("Context Excerpt", min_width=45)

            for idx, src in enumerate(resp.sources, 1):
                s_color = "green" if src.score >= 0.60 else ("yellow" if src.score >= 0.45 else "red")
                sim_str = f"[{s_color}]{src.score:.4f}[/{s_color}]"
                doc_info = f"[bold]{src.chunk.document_title}[/bold]\n[dim]{src.chunk.chunk_id}[/dim]"
                snippet = src.chunk.text[:180].replace("\n", " ").strip() + "..."
                table.add_row(f"#{idx}", doc_info, sim_str, snippet)

            console.print(table)
            console.print()
    else:
        print("=" * 70)
        print(f"QUERY: {resp.query}")
        print(f"ANSWER: {resp.answer}")
        print(f"CONFIDENCE: {resp.confidence_score} | LATENCY: {resp.latency_ms}ms")
        print("\nSOURCES:")
        for idx, src in enumerate(resp.sources, 1):
            print(f"[{idx}] {src.chunk.document_title} ({src.chunk.chunk_id}) - Similarity: {src.score:.4f}")
            print(f"    Excerpt: {src.chunk.text[:120]}...\n")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Lite: Grounded semantic search and FAQ engine with abstention guardrails."
    )
    parser.add_argument("-q", "--query", type=str, help="Question to ask the knowledge base")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="Number of context chunks to retrieve (default: 3)")
    parser.add_argument("-t", "--threshold", type=float, default=0.40, help="Similarity score abstention threshold (default: 0.40)")
    parser.add_argument("--kb-dir", type=str, default=None, help="Directory containing .md or .txt knowledge documents")
    parser.add_argument("--mock", action="store_true", help="Force deterministic mock generation (no API keys needed)")

    args = parser.parse_args()

    # Determine knowledge base directory
    if args.kb_dir:
        kb_path = Path(args.kb_dir)
    else:
        kb_path = Path(__file__).resolve().parent.parent / "knowledge_base"

    if not kb_path.exists():
        print(f"Error: Knowledge base directory not found at {kb_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize RAG Pipeline
    generator = MockGenerator() if args.mock else None
    pipeline = RAGPipeline(
        default_score_threshold=args.threshold,
        generator=generator
    )

    # Ingest documents
    num_chunks = pipeline.ingest_directory(kb_path)
    docs = pipeline.vector_store.list_documents()

    if HAS_RICH and console:
        console.print(Panel(
            f"[bold white]Indexed Documents:[/bold white] [cyan]{', '.join(docs)}[/cyan]\n"
            f"[bold white]Total Vector Chunks:[/bold white] [green]{num_chunks}[/green]\n"
            f"[bold white]Abstention Threshold:[/bold white] [yellow]{args.threshold}[/yellow]",
            title="[bold green]⚡ RAG Lite Knowledge Engine Online[/bold green]",
            border_style="green"
        ))

    if args.query:
        resp = pipeline.query(args.query, top_k=args.top_k, score_threshold=args.threshold)
        render_response(resp)
    else:
        # Interactive REPL
        if HAS_RICH and console:
            console.print("[dim]Type your question and press Enter. Type 'exit' or 'quit' to close.[/dim]\n")
        
        while True:
            try:
                prompt_str = "Ask RAG Lite > "
                user_q = input(prompt_str).strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    break
                resp = pipeline.query(user_q, top_k=args.top_k, score_threshold=args.threshold)
                render_response(resp)
            except (KeyboardInterrupt, EOFError):
                print()
                break


if __name__ == "__main__":
    main()
