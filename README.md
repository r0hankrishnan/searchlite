# searchlite

![searchlite logo](assets/searchlite-no-bg.svg)

A simple Python package that allows semantic search with simple syntax on **small** text data sets.

## Features

- Easily embed and index collections of texts with a variety of embedding methods:
  - scikit-learn's TFIDF
  - Any embedding model from Ollama
  - Any embedding model from the sentence_transformers package
  - Any API-based embedding workflow
- Query with natural language to find the most semantically similar texts
- Display results with multiple formatting options (`f-string`, `pprint`, `tabulate`)
- Lightweight and minimal dependencies by default

## Scaling Considerations

`searchlite` is intentionally designed for **lightweight semantic search** on **small to moderately sized** text datasets that fit in memory. It excels in use cases like:

* Prototyping search workflows
* Performing quick, local experiments
* Building educational or notebook-based tools
* Evaluating embedding strategies

Because embeddings are stored in memory using NumPy arrays, `searchlite` is **not optimized for large-scale corpora or long-term persistence**.

## Working with Large Datasets?

If you're working with large text datasets or need production-grade semantic search with features like:

* Efficient vector indexing
* Disk-based or cloud-based storage
* Scalable nearest neighbor search
* Metadata filtering and advanced retrieval

You should consider using a dedicated vector database such as:

* [Chroma](https://www.trychroma.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Pinecone](https://www.pinecone.io/)
* [Weaviate](https://weaviate.io/)
* [Qdrant](https://qdrant.tech/)

These libraries offer Python SDKs and built-in support for embedding and querying at scale.

## Installation

Currently, **searchlite** is not published on PyPI.

To get started:

1. Clone the repository:

```bash
git clone https://github.com/r0hankrishnan/searchlite.git
cd searchlite
````

2. Install dependencies (recommend creating a virtual environment):

```bash
pip install -r requirements.txt
```

> **Note:** The `sentence-transformers` library currently requires `numpy` version less than 2. Please ensure you have `numpy<2` installed to avoid compatibility issues.

## Usage Example

```python
from searchlite.document import Document
import pandas as pd
import os

current_file_path = os.path.dirname(__file__)
data_path = os.path.join(current_file_path, "../data/synthetic_data.csv")

sample_df = pd.read_csv(data_path, index_col=0)
sample_texts = sample_df["text"]
sample_metadata = sample_df[["id", "category"]].to_dict(orient="records")

doc = Document(texts=sample_texts, metadata=sample_metadata)

print(doc)

doc.embed()

res = doc.query(query_text="wireless earbuds with good battery life")

print(res)

doc.display_results(res, options="tabulate")
```

```
+------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
|   id | category            | text                                                                                                                                                                          |   similarity score |
+======+=====================+===============================================================================================================================================================================+====================+
|    1 | Product Description | Experience unparalleled sound quality with the EchoSphere wireless earbuds, featuring noise cancellation, 12-hour battery life, and an ergonomic design perfect for workouts. |          0.492049  |
+------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
|   11 | Customer Review     | The blender exceeded my expectations with its powerful motor and easy-to-clean design. Perfect for smoothies and soups!                                                       |          0.0741458 |
+------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
|   14 | E-commerce FAQ      | Q: Does this jacket have waterproof capabilities? A: Yes, it is made with breathable waterproof fabric suitable for heavy rain.                                               |          0.0657988 |
+------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
```

## Requirements

* Python 3.7 or higher
* Core dependencies:

  * `numpy < 2`
  * `scikit-learn >= 1.0`
  * `tabulate >= 0.8.0`
  * `pandas >= 1.0`

* Optional dependencies:

  * Sentence Transformers
    * `sentence-transformers >= 2.2.0`
  * Ollama
    * `ollama`
  * Dev
    * `pytest`

You can find the full list in [setup.py](setup.py) or [pyproject.toml](pyproject.toml).

**Note:** 

If you want to use SentenceTransformer-based embeddings, you’ll need to install the optional dependency:

```bash
pip install searchlite[sentence_transformers]
```

Or, install manually:

```bash
pip install -r requirements_st.txt
```

If you want to use Ollama-based embeddings, you'll need to install the optional dependency:

```bash
pip install searchlite[ollama]
```

Or, install manually:

```bash
pip install -r requirements_ollama.txt
```

## Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or pull requests, feel free to open an issue or submit PRs.

This is my first Python package project, built as a learning experience, so I truly appreciate constructive feedback and ideas for improvement.

## License

This project is licensed under the MIT License.

## A Note on AI

This README was written with the help of Chat GPT. I also used Chat GPT to generate the synthetic data used in the tests and in the notebook demo. **All code, however, was written by me!** I strongly believe in transparency when it comes to the use of generative AI. In that spirit, you can visit [chat-gpt-prompt.md](chat-gpt-prompt.md) to see the **exact** prompt I used to help me reason through problems and think critically while designing seachlite. :)

## Future Plans

Here are some features I'm working on adding for future versions of `searchlite`:

- [x] Support for embedding models other than "all-MiniLM-L6-v2"
- [x] Create "lightweight" alternative that doesn't require loading sentence-transformers
- [ ] Support for other similarity measures
    - [ ] Implement similarity metrics from scratch and remove scikit-learn dependency
    - [ ] TFIDF (from scratch)
- [x] Other embedding type supports
    - [x] TFIDF (sklearn)
    - [x] Ollama
    - [x] Sentence Transformers
    - [x] API
- [ ] Batching
- [ ] Support for adding normalization to models that do not automatically normalize their embeddings
- [ ] Functions to allow for easy conversion of a pandas dataframe to a Document object
- [ ] Support for keyword, TF-IDF, BM25, and fuzzy matching search
- [ ] Support for weighted hybrid search
- [ ] Support for reranking search
- [ ] Support for built-in preprocessing of texts
- [ ] Built-in support for filtering query results by metadata fields
- [ ] Caching of embeddings to avoid re-computation across sessions
- [ ] CLI interface for querying datasets without writing code
- [ ] Integration with Jupyter widgets for interactive exploration
- [ ] JSON/YAML import/export for datasets and results
- [ ] Caching for larger embedding tasks

## Contact

The repository is open on GitHub:
[https://github.com/r0hankrishnan/searchlite](https://github.com/r0hankrishnan/searchlite)

Feel free to open issues or reach out with questions or suggestions!


Thank you for checking out **searchlite**!
