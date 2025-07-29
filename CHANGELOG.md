# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features

#### Keyword Search Support
- [ ] BM25 implementation
- [ ] Fuzzy matching support

#### Enhanced Data Handling
- [ ] Functions to allow for easy conversion of pandas dataframe to a Document object using `@classmethod`
- [ ] JSON/YAML import/export for datasets and results
- [ ] Support for built-in preprocessing of texts
- [ ] Support for checking and adding normalization to embeddings

#### Search and Retrieval Improvements
- [ ] Support for other similarity measures
- [ ] Built-in support for filtering query results by metadata fields

#### Performance Optimizations
- [ ] Batching support for large embedding tasks
- [ ] Caching of embeddings to avoid re-computation across sessions
- [ ] Caching for larger embedding tasks

## v0.2.0 [Current] - Initial Release

### Added
- Core semantic search functionality for small text datasets
- Multiple embedding methods support:
  - [x] scikit-learn's TFIDF
  - [x] Ollama embedding models
  - [x] Sentence Transformers models
  - [x] API-based embedding workflows
- Natural language querying with cosine similarity
- Multiple result display formats (f-string, pprint, tabulate)
- Document class for managing texts and metadata
- Support for embedding models other than "all-MiniLM-L6-v2"
- Lightweight alternative that doesn't require loading sentence-transformers

### Features
- Easily embed and index collections of texts
- Query with natural language to find semantically similar texts
- Display results with multiple formatting options
- Minimal dependencies by default
- Educational and prototype-friendly design

### Scaling Considerations
- Intentionally designed for lightweight semantic search on small to moderately sized text datasets
- In-memory storage using NumPy arrays
- Not optimized for large-scale corpora or long-term persistence

---

**Note**: This project is designed for small to moderate datasets. For large-scale or production use cases, consider dedicated vector databases like Chroma, FAISS, Pinecone, Weaviate, or Qdrant.