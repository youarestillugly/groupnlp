import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import os

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DEPLOY_DIR = BASE_DIR / "artifacts" / "deployment_D"

MODEL_PATH = (
    BASE_DIR / "models" /
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# 2. LOAD CONFIGURATION D ARTIFACTS AND MODELS
# =========================================================

@st.cache_resource
def load_resources():

    # Load saved Configuration D artifacts
    corpus = joblib.load(
        DEPLOY_DIR / "hybrid_corpus.joblib"
    )

    bm25_index = joblib.load(
        DEPLOY_DIR / "hybrid_bm25_index.joblib"
    )

    embeddings = np.load(
        DEPLOY_DIR / "dense_embeddings.npy"
    )

    with open(
        DEPLOY_DIR / "deployment_config.json",
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    # Load embedding model
    embedding_model = SentenceTransformer(
        config["embedding_model"]
    )

    # Load CrossEncoder
    cross_encoder = CrossEncoder(
        config["cross_encoder_model"]
    )

    # Load local Qwen GGUF model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Qwen model not found: {MODEL_PATH}"
        )

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=config["context_size"],
        n_threads=4,
        verbose=False
    )

    # Check alignment
    if not (
        len(corpus)
        == bm25_index.corpus_size
        == embeddings.shape[0]
    ):
        raise ValueError(
            "Configuration D artifacts are not aligned."
        )

    return (
        corpus,
        bm25_index,
        embeddings,
        embedding_model,
        cross_encoder,
        llm,
        config
    )


# =========================================================
# 3. BM25 RETRIEVAL
# =========================================================

def hybrid_bm25_search(
    query,
    corpus,
    bm25_index,
    top_k=10
):

    tokenized_query = query.lower().split()

    scores = bm25_index.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = corpus.iloc[top_indices].copy()

    results["bm25_score"] = scores[top_indices]

    results["bm25_rank"] = range(
        1, len(results) + 1
    )

    return results


# =========================================================
# 4. DENSE RETRIEVAL
# =========================================================

def hybrid_dense_search(
    query,
    corpus,
    embeddings,
    embedding_model,
    top_k=10
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = corpus.iloc[top_indices].copy()

    results["dense_score"] = scores[top_indices]

    results["dense_rank"] = range(
        1, len(results) + 1
    )

    return results


# =========================================================
# 5. RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    corpus,
    top_k=10,
    k=60
):

    rank_scores = {}

    for _, row in bm25_results.iterrows():

        chunk_id = row["chunk_id"]

        rank_scores[chunk_id] = (
            rank_scores.get(chunk_id, 0)
            + 1 / (k + row["bm25_rank"])
        )

    for _, row in dense_results.iterrows():

        chunk_id = row["chunk_id"]

        rank_scores[chunk_id] = (
            rank_scores.get(chunk_id, 0)
            + 1 / (k + row["dense_rank"])
        )

    ranked_ids = sorted(
        rank_scores,
        key=rank_scores.get,
        reverse=True
    )[:top_k]

    results = corpus[
        corpus["chunk_id"].isin(ranked_ids)
    ].copy()

    results["rrf_score"] = (
        results["chunk_id"].map(rank_scores)
    )

    results = results.sort_values(
        "rrf_score",
        ascending=False
    ).reset_index(drop=True)

    results["fusion_rank"] = range(
        1, len(results) + 1
    )

    return results


# =========================================================
# 6. HYBRID RETRIEVAL + CROSS-ENCODER RERANKING
# =========================================================

def retrieve_hybrid_evidence(
    query,
    corpus,
    bm25_index,
    embeddings,
    embedding_model,
    cross_encoder,
    top_k=5,
    retrieval_k=10
):

    # BM25 retrieval
    bm25_results = hybrid_bm25_search(
        query,
        corpus,
        bm25_index,
        top_k=retrieval_k
    )

    # Dense retrieval
    dense_results = hybrid_dense_search(
        query,
        corpus,
        embeddings,
        embedding_model,
        top_k=retrieval_k
    )

    # Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        corpus,
        top_k=retrieval_k
    )

    # CrossEncoder reranking
    pairs = [
        (query, passage)
        for passage in fused_results["text"].tolist()
    ]

    scores = cross_encoder.predict(pairs)

    fused_results = fused_results.copy()

    fused_results["rerank_score"] = scores

    reranked = fused_results.sort_values(
        "rerank_score",
        ascending=False
    ).reset_index(drop=True)

    reranked["rerank_rank"] = range(
        1, len(reranked) + 1
    )

    return reranked.head(top_k)


# =========================================================
# 7. GROUNDED ANSWER GENERATION
# =========================================================

def generate_hybrid_answer(
    question,
    selected_evidence,
    llm,
    max_tokens=220
):

    if selected_evidence is None or selected_evidence.empty:

        return {
            "answer": (
                "Insufficient evidence in the provided collection."
            ),
            "sources": {},
            "generation_time": 0,
            "finish_reason": "no_evidence",
            "prompt_tokens": 0
        }

    evidence_blocks = []

    source_mapping = {}

    for index, (_, row) in enumerate(
        selected_evidence.iterrows(),
        start=1
    ):

        source_label = f"S{index}"

        source_mapping[source_label] = {
            "article_id": row["article_id"],
            "title": row["title"],
            "page_number": row["page_number"],
            "chunk_id": row["chunk_id"]
        }

        passage = " ".join(
            str(row["text"]).split()[:100]
        )

        evidence_blocks.append(
            f"[{source_label}]\n"
            f"Article: {row['title']}\n"
            f"Page: {row['page_number']}\n"
            f"Passage: {passage}"
        )

    evidence_text = "\n\n".join(
        evidence_blocks
    )

    valid_labels = ", ".join(
        f"[{label}]" for label in source_mapping
    )

    prompt = f"""
You are an academic research assistant.

Answer the question using only the supplied research evidence.

Rules:
1. Give a direct answer in 2-4 sentences.
2. Use only information supported by the evidence.
3. Cite supporting evidence using only these labels: {valid_labels}.
4. Never invent or use citation labels that are not supplied.
5. Do not create a reference list.
6. If the evidence cannot answer the question, state:
   Insufficient evidence in the provided collection.
7. Do not guess or invent information.

QUESTION:
{question}

RESEARCH EVIDENCE:
{evidence_text}

ANSWER:
"""

    prompt_tokens = len(
        llm.tokenize(prompt.encode("utf-8"))
    )

    if prompt_tokens + max_tokens > llm.n_ctx():

        raise ValueError(
            f"Prompt requires {prompt_tokens} tokens. "
            f"Available space for prompt: "
            f"{llm.n_ctx() - max_tokens}."
        )

    start_time = time.time()

    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.9,
        stop=["</s>"]
    )

    generation_time = time.time() - start_time

    answer = response["choices"][0]["text"].strip()

    finish_reason = response["choices"][0]["finish_reason"]

    return {
        "answer": answer,
        "sources": source_mapping,
        "generation_time": generation_time,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens
    }


# =========================================================
# 8. USER INTERFACE
# =========================================================

st.title("📚 Research Assistant")

st.markdown(
    """
    ### Evidence-Grounded Academic Research Assistant

    Ask questions about the research papers in the collection.
    Answers are generated using retrieved research evidence
    and include source citations.
    """
)

st.divider()

# Load application resources
try:

    with st.spinner(
        "Loading Configuration D models and retrieval resources..."
    ):

        (
            corpus,
            bm25_index,
            embeddings,
            embedding_model,
            cross_encoder,
            llm,
            config
        ) = load_resources()

    st.success(
        "Research Assistant is ready."
    )

except Exception as error:

    st.error(
        f"Unable to load application resources: {error}"
    )

    st.stop()


# Application information
with st.expander("About this system"):

    st.write(
        f"**Research papers:** {config['unique_articles']}"
    )

    st.write(
        f"**Retrieval passages:** {config['total_chunks']}"
    )

    st.write(
        "**Retrieval:** BM25 + Dense Retrieval"
    )

    st.write(
        "**Fusion:** Reciprocal Rank Fusion"
    )

    st.write(
        "**Reranking:** CrossEncoder"
    )

    st.write(
        "**Language model:** Qwen 2.5 0.5B Instruct"
    )


# Question input
question = st.text_area(
    "Enter your research question:",
    placeholder=(
        "Example: What machine learning methods "
        "are used for brain tumour classification?"
    ),
    height=120
)

ask_button = st.button(
    "🔍 Ask Research Assistant",
    type="primary",
    use_container_width=True
)


# =========================================================
# 9. PROCESS QUESTION
# =========================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        try:

            with st.spinner(
                "Searching research papers and generating your answer..."
            ):

                # Retrieve and rerank evidence
                evidence = retrieve_hybrid_evidence(
                    question,
                    corpus,
                    bm25_index,
                    embeddings,
                    embedding_model,
                    cross_encoder,
                    top_k=5,
                    retrieval_k=10
                )

                # Generate grounded answer
                result = generate_hybrid_answer(
                    question,
                    evidence,
                    llm,
                    max_tokens=220
                )

            st.divider()

            st.subheader("Answer")

            st.write(result["answer"])

            # Display citations
            st.subheader("Supporting Sources")

            for label, source in result["sources"].items():

                st.markdown(
                    f"**[{label}] {source['title']}**"
                )

                st.caption(
                    f"Article ID: {source['article_id']} | "
                    f"Page: {source['page_number']} | "
                    f"Chunk: {source['chunk_id']}"
                )

            # Display retrieved passages
            st.subheader("Retrieved Evidence")

            for index, (_, row) in enumerate(
                evidence.iterrows(),
                start=1
            ):

                with st.expander(
                    f"Passage {index}: {row['title']}"
                ):

                    st.write(
                        f"**Article:** {row['title']}"
                    )

                    st.write(
                        f"**Page:** {row['page_number']}"
                    )

                    st.write(
                        f"**Section:** {row['section_heading']}"
                    )

                    st.write(
                        f"**Chunk ID:** {row['chunk_id']}"
                    )

                    st.write(
                        row["text"]
                    )

                    st.caption(
                        f"Reranking score: "
                        f"{row['rerank_score']:.4f}"
                    )

            # Performance information
            st.divider()

            st.subheader("Response Information")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Generation time",
                f"{result['generation_time']:.2f} sec"
            )

            col2.metric(
                "Prompt tokens",
                result["prompt_tokens"]
            )

            col3.metric(
                "Retrieved passages",
                len(evidence)
            )

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )


# Footer
st.divider()

st.caption(
    "Powered by Configuration D | "
    "BM25 + Dense Retrieval + RRF + CrossEncoder + Qwen"
)
