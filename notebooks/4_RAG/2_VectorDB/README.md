# VectorDB

## VectorDBs

| Framework | Website | Open Source | Deployment | Notes |
|----------|---------|-------------|------------|-------|
| [FAISS](https://faiss.ai/) | faiss.ai | MIT | Local (CPU/GPU) | Optimized for dense vector search; from Meta. |
| [ChromaDB](https://www.trychroma.com/) | trychroma.com | Yes | Local | Built for LLM workflows; fast and Pythonic. |
| [Milvus](https://milvus.io/) | milvus.io | Apache 2.0 | Distributed, Cloud | Scalable and production-ready. |
| [Qdrant](https://qdrant.tech/) | qdrant.tech | Apache 2.0 | Local, Cloud | Neural search optimized; REST & gRPC APIs. |
| [Weaviate](https://weaviate.io/) | weaviate.io | BSD-3 | Local, Cloud | GraphQL interface; semantic capabilities. |
| [LanceDB](https://lancedb.com/) | lancedb.com | Yes | Local | Native Python, fast I/O, columnar format. |
| [vectordb (Jina)](https://github.com/jina-ai/vectordb) | GitHub | Apache 2.0 | Local, Cloud | Minimalistic & scalable; powered by DocArray. |
| [Bhakti](https://arxiv.org/abs/2504.01553) | arxiv.org | Yes | Local | Lightweight; for small/medium vector search. |

## Chunking Techniques

| Chunking Technique         | Description                                                                                             | Advantages                                                                                  | Disadvantages                                                                                 | Best Use Cases                                                                                 |
|----------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **Fixed-Length Chunking**  | Splits text into chunks of a specified number of tokens or characters.                                  | Simple to implement; consistent chunk sizes.                                                | May split sentences or paragraphs; can disrupt semantic coherence.                            | Structured documents where uniform chunk size is preferred.                                    |
| **Sentence-Based Chunking**| Divides text at sentence boundaries using NLP tools like spaCy or NLTK.                                 | Preserves grammatical structure; maintains semantic integrity.                              | Variable chunk sizes; may not capture broader context.                                        | Documents where sentence-level context is sufficient.                                          |
| **Semantic Chunking**      | Groups semantically similar sentences using embeddings and clustering algorithms.                       | Maintains contextual relevance; enhances retrieval accuracy.                                | Computationally intensive; complex implementation.                                            | Applications requiring high semantic coherence, like semantic search or Q&A systems.           |
| **Recursive Chunking**     | Hierarchically splits text using multiple delimiters (e.g., paragraphs, sentences) with overlap.        | Balances chunk size and context; preserves hierarchical structure.                          | Implementation complexity; may produce variable chunk sizes.                                  | Documents with nested structures, such as technical manuals or legal texts.                    |
| **Format-Aware Chunking**  | Tailors chunking based on document format (e.g., splitting code by functions, HTML by tags).            | Leverages inherent document structure; maintains logical divisions.                         | Requires format-specific parsers; less generalizable.                                         | Code repositories, HTML documents, or any content with a well-defined structural format.       |
| **Heading-Based Chunking** | Splits text based on headings (e.g., Markdown headers) to preserve section semantics.                   | Maintains document hierarchy; facilitates targeted retrieval.                               | Dependent on consistent use of headings; may not be applicable to all document types.         | Structured documents like reports, articles, or books with clear heading delineations.         |


Embeddings are numerical representations of data—such as words, sentences, or documents—that capture their semantic meaning. They enable machines to process and understand human language by placing similar concepts close together in a vector space.

---

## Understanding Embeddings

Embeddings are numerical representations of data—such as words, sentences, or documents—that capture their semantic meaning. They enable machines to process and understand human language by placing similar concepts close together in a vector space.

---

### How Document Embedding Works: Step-by-Step

1. **Input Document**  
   Example:  
   `"The cat sat on the mat."`

2. **Tokenization**  
   Split the text into tokens:  
   `["The", "cat", "sat", "on", "the", "mat"]`

3. **Embedding Generation**  
   Each token is converted into a vector using an embedding model.  
   For instance:  
   - `"The"` → `[0.1, 0.3, ...]`  
   - `"cat"` → `[0.2, 0.4, ...]`  
   - ...

4. **Aggregation**  
   Combine the individual token vectors into a single document vector, often by averaging.  
   Document Vector = Mean(`["The", "cat", "sat", "on", "the", "mat"]`)

5. **Output**  
   A fixed-size vector representing the entire document, which can be used for tasks like semantic search, clustering, or classification.

---

### 🧠 Common Embedding Models

| Model                   | Type          | Dimensions | Contextual | Typical Use Cases                         |
|-------------------------|---------------|------------|------------|-------------------------------------------|
| Word2Vec                | Static        | 100–300    | No         | Word similarity, analogy tasks            |
| GloVe                   | Static        | 50–300     | No         | Semantic analysis, information retrieval  |
| FastText                | Static        | 100–300    | No         | Handling rare words, subword information  |
| ELMo                    | Contextual    | 1024       | Yes        | Sentiment analysis, question answering    |
| BERT                    | Contextual    | 768        | Yes        | Text classification, NER, QA systems      |
| Sentence-BERT (SBERT)   | Contextual    | 384–768    | Yes        | Semantic search, clustering, STS tasks    |
| OpenAI's Ada            | Contextual    | 1536       | Yes        | Semantic search, document clustering      |

---

### API-Based vs. Self-Hosted Embedding Models

#### API-Based Models (e.g., OpenAI's Ada)

**Pros:**
- Easy to integrate; no need for infrastructure setup.
- Access to state-of-the-art models maintained by providers.
- Scalable without managing hardware resources.

**Cons:**
- Data is sent to external servers, raising privacy concerns.
- Dependent on third-party service availability and pricing.
- Limited customization of the model to specific needs.

#### Self-Hosted Models (e.g., BERT, SBERT)

**Pros:**
- Full control over data privacy and security.
- Ability to fine-tune models for specific tasks.
- No ongoing costs associated with API usage.

**Cons:**
- Requires significant computational resources and expertise.
- Longer setup time and maintenance overhead.
- May not match the performance of cutting-edge API models without extensive tuning.

---

Choosing between API-based and self-hosted embedding models depends on factors like data sensitivity, resource availability, and specific application requirements.