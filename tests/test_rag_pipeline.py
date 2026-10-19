from rag_lite.engine import RAGPipeline
from rag_lite.providers import MockGenerator
from rag_lite.models import RAGResponse


def test_rag_pipeline_end_to_end():
    pipeline = RAGPipeline(
        generator=MockGenerator(),
        default_score_threshold=0.30
    )

    doc_text = """
    # Argo Rollout Policy
    Our automated rollback protocol states that if the HTTP 5xx error rate exceeds 0.5%,
    the rollout aborts immediately and reverts traffic to the previous stable release.
    """
    pipeline.ingest_text("Argo Rollouts", doc_text, doc_id="argo_doc")

    # In-domain query
    resp: RAGResponse = pipeline.query("What is the rollback error rate threshold?", top_k=1, score_threshold=0.30)
    assert not resp.is_abstained
    assert resp.confidence_score > 0.30
    assert len(resp.sources) == 1
    assert "0.5%" in resp.answer or "0.5%" in resp.sources[0].chunk.text


def test_rag_pipeline_abstention_guardrail():
    pipeline = RAGPipeline(
        generator=MockGenerator(),
        default_score_threshold=0.60
    )

    doc_text = "Only information about Stripe billing and subscription webhooks."
    pipeline.ingest_text("Stripe Docs", doc_text, doc_id="stripe_doc")

    # Completely out-of-domain query
    resp: RAGResponse = pipeline.query(
        "How do astronauts sleep in zero gravity on the ISS?",
        score_threshold=0.75
    )
    assert resp.is_abstained
    assert resp.confidence_score == 0.0
    assert len(resp.sources) == 0
    assert "Abstained" in resp.answer
