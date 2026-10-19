from rag_lite.vector_store import InMemoryVectorStore
from rag_lite.models import DocumentChunk
from rag_lite.embeddings import LocalSemanticEmbeddingProvider


def test_vector_store_add_and_search():
    store = InMemoryVectorStore(embedding_provider=LocalSemanticEmbeddingProvider(dimension=128))
    
    chunk1 = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Stripe Guide",
        text="When invoice.payment_failed occurs, trigger smart retries.",
        chunk_index=0,
        start_char=0,
        end_char=50
    )
    chunk2 = DocumentChunk(
        chunk_id="chunk-2",
        document_id="doc-2",
        document_title="Kubernetes SOP",
        text="Kubernetes deployment rollouts automatically revert if 5xx errors spike.",
        chunk_index=0,
        start_char=0,
        end_char=60
    )

    store.add_chunks([chunk1, chunk2])
    assert store.count() == 2

    # Query matching chunk 1
    results = store.search(query="invoice payment failed retries", top_k=1, score_threshold=0.2)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].score > 0.3


def test_vector_store_threshold_abstention():
    store = InMemoryVectorStore()
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        document_title="Doc",
        text="Technical documentation about databases.",
        chunk_index=0,
        start_char=0,
        end_char=30
    )
    store.add_chunks([chunk])

    # Unrelated query with high threshold should return empty (abstain)
    results = store.search(query="banana cupcake recipes with strawberries", top_k=1, score_threshold=0.85)
    assert len(results) == 0
