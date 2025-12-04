"""
Vietnamese Embedding Refresh Pipeline.

This module refreshes Vietnamese Embeddings (AITeamVN/Vietnamese_Embedding) for products.
Run via: python -m automation.bert_embeddings
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    PipelineTracker,
    PipelineLock,
    setup_logging,
    send_pipeline_alert,
    get_git_commit,
)


# =============================================================================
# Configuration
# =============================================================================

BERT_CONFIG = {
    "model_name": "AITeamVN/Vietnamese_Embedding",
    "product_file": PROJECT_ROOT / "data" / "published_data" / "data_product.csv",
    "output_dir": PROJECT_ROOT / "data" / "processed" / "content_based_embeddings",
    "output_file": "product_embeddings.pt",
    "batch_size": 32,
    "max_length": 512,
    "embedding_dim": 1024,  # Vietnamese_Embedding output dimension
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# Embedding Generation
# =============================================================================

def load_product_texts(logger: logging.Logger) -> Dict[str, str]:
    """Load product texts for embedding generation."""
    import pandas as pd

    product_file = BERT_CONFIG["product_file"]
    logger.info(f"Loading products from {product_file}")

    df = pd.read_csv(product_file, encoding="utf-8")

    # Combine relevant text fields
    texts = {}
    for _, row in df.iterrows():
        product_id = str(row.get("product_id", row.name))

        # Combine: name + description + features
        text_parts = []
        if pd.notna(row.get("product_name")):
            text_parts.append(str(row["product_name"]))
        if pd.notna(row.get("processed_description")):
            text_parts.append(str(row["processed_description"]))
        if pd.notna(row.get("feature")):
            text_parts.append(str(row["feature"]))

        texts[product_id] = " [SEP] ".join(text_parts) if text_parts else ""

    logger.info(f"Loaded {len(texts)} products")
    return texts


def mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply mean pooling to model output.
    
    Args:
        model_output: Output from transformer model.
        attention_mask: Attention mask tensor.
        
    Returns:
        Mean pooled embeddings tensor.
    """
    token_embeddings = model_output[0]  # First element is last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def generate_embeddings(
    texts: Dict[str, str],
    logger: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Generate Vietnamese Embeddings for product texts using mean pooling."""
    from transformers import AutoTokenizer, AutoModel

    model_name = BERT_CONFIG["model_name"]
    device = BERT_CONFIG["device"]
    batch_size = BERT_CONFIG["batch_size"]
    max_length = BERT_CONFIG["max_length"]

    logger.info(f"Loading model {model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    product_ids = list(texts.keys())
    product_texts = [texts[pid] for pid in product_ids]

    embeddings = {}
    total_batches = (len(product_texts) + batch_size - 1) // batch_size

    logger.info(f"Generating embeddings for {len(product_texts)} products...")

    with torch.no_grad():
        for i in range(0, len(product_texts), batch_size):
            batch_texts = product_texts[i : i + batch_size]
            batch_ids = product_ids[i : i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate embeddings
            outputs = model(**inputs)

            # Apply mean pooling (not CLS token) for Vietnamese_Embedding
            attention_mask = inputs["attention_mask"]
            batch_embeddings = mean_pooling(outputs, attention_mask).cpu()

            for pid, emb in zip(batch_ids, batch_embeddings):
                embeddings[pid] = emb

            batch_num = i // batch_size + 1
            if batch_num % 10 == 0:
                logger.info(f"  Progress: {batch_num}/{total_batches} batches")

    logger.info(f"Generated {len(embeddings)} embeddings (dim={BERT_CONFIG['embedding_dim']})")
    return embeddings


def save_embeddings(
    embeddings: Dict[str, torch.Tensor],
    logger: logging.Logger,
) -> Path:
    """Save embeddings to file."""
    output_dir = BERT_CONFIG["output_dir"]
    output_file = output_dir / BERT_CONFIG["output_file"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to tensor format
    product_ids = list(embeddings.keys())
    embedding_tensor = torch.stack([embeddings[pid] for pid in product_ids])

    # Save with comprehensive metadata
    torch.save(
        {
            "product_ids": product_ids,
            "embeddings": embedding_tensor,
            "metadata": {
                "model_name": BERT_CONFIG["model_name"],
                "embedding_dim": BERT_CONFIG["embedding_dim"],
                "num_products": len(product_ids),
                "pooling_method": "mean",
                "max_length": BERT_CONFIG["max_length"],
            },
            "created_at": datetime.now().isoformat(),
            "shape": embedding_tensor.shape,
        },
        output_file,
    )

    logger.info(f"Saved embeddings to {output_file}")
    logger.info(f"  Shape: {embedding_tensor.shape}")

    return output_file


# =============================================================================
# Main Pipeline
# =============================================================================

def refresh_bert_embeddings(
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Main BERT embeddings refresh pipeline.

    Args:
        force: Force refresh even if embeddings exist
        logger: Logger instance

    Returns:
        Pipeline result
    """
    if logger is None:
        logger = setup_logging("bert_embeddings")

    tracker = PipelineTracker()
    result: Dict[str, Any] = {
        "pipeline": "bert_embeddings",
        "started_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }

    with PipelineLock("bert_embeddings") as lock:
        if not lock.acquired:
            msg = "BERT embeddings refresh already running"
            logger.warning(msg)
            result["status"] = "skipped"
            result["message"] = msg
            return result

        run_id = tracker.start_run("bert_embeddings", {"force": force})

        try:
            # Check if refresh needed
            output_file = BERT_CONFIG["output_dir"] / BERT_CONFIG["output_file"]
            if output_file.exists() and not force:
                # Check age
                mtime = datetime.fromtimestamp(output_file.stat().st_mtime)
                age_days = (datetime.now() - mtime).days

                if age_days < 7:
                    msg = f"Embeddings are fresh (age: {age_days} days), skipping"
                    logger.info(msg)
                    result["status"] = "skipped"
                    result["message"] = msg
                    tracker.complete_run(run_id, {"status": "skipped"})
                    return result

            # Load product texts
            logger.info("Loading product texts...")
            texts = load_product_texts(logger)

            # Generate embeddings
            logger.info("Generating embeddings...")
            start_time = datetime.now()
            embeddings = generate_embeddings(texts, logger)
            duration = (datetime.now() - start_time).total_seconds()

            # Save
            output_path = save_embeddings(embeddings, logger)

            # Success
            result["status"] = "success"
            result["finished_at"] = datetime.now().isoformat()
            result["duration_seconds"] = duration
            result["num_products"] = len(embeddings)
            result["output_file"] = str(output_path)

            tracker.complete_run(
                run_id,
                {
                    "status": "success",
                    "num_products": len(embeddings),
                    "duration": duration,
                },
            )

            logger.info("BERT embeddings refresh completed!")

            send_pipeline_alert(
                "bert_embeddings",
                "success",
                f"Generated {len(embeddings)} embeddings in {duration:.1f}s",
                severity="info",
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"BERT embeddings refresh failed: {error_msg}")

            result["status"] = "failed"
            result["error"] = error_msg

            tracker.fail_run(run_id, error_msg)

            send_pipeline_alert(
                "bert_embeddings",
                "failed",
                f"BERT embeddings failed: {error_msg}",
                severity="error",
            )

            raise

    return result


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Refresh PhoBERT embeddings for products",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force refresh even if embeddings are fresh",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("bert_embeddings", level=level)

    try:
        result = refresh_bert_embeddings(
            force=args.force,
            logger=logger,
        )

        print(f"\n{'=' * 60}")
        print(f"BERT Embeddings: {result['status'].upper()}")
        print(f"{'=' * 60}")

        if result["status"] == "success":
            print(f"  Products: {result['num_products']}")
            print(f"  Duration: {result['duration_seconds']:.1f}s")
            print(f"  Output: {result['output_file']}")
        elif result.get("message"):
            print(f"  Message: {result['message']}")

        sys.exit(0 if result["status"] in ("success", "skipped") else 1)

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
