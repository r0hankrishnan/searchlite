# searchlite

![searchlite logo](assets/searchlite-no-bg.svg)

A simple Python package that allows semantic search on text data sets with simple syntax.

---

## Features

- Easily embed and index collections of texts with sentence-transformers
- Query with natural language to find the most semantically similar texts
- Display results with multiple formatting options (`f-string`, `pprint`, `tabulate`)
- Lightweight and minimal dependencies

---

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

---

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
+------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|   id | category            | text                                                                                                                                                                          |
+======+=====================+===============================================================================================================================================================================+
|    1 | Product Description | Experience unparalleled sound quality with the EchoSphere wireless earbuds, featuring noise cancellation, 12-hour battery life, and an ergonomic design perfect for workouts. |
+------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|    5 | Travel Guide        | Discover the hidden gems of Kyoto, from tranquil temples to bustling markets, and experience authentic Japanese culture like never before.                                    |
+------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|    4 | Recipe              | Preheat the oven to 375°F. Mix flour, sugar, and eggs in a bowl, then fold in fresh blueberries. Bake for 25 minutes or until golden brown.                                   |
+------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
---

## Requirements

* Python 3.7+
* `sentence-transformers`
* `numpy < 2`
* `scikit-learn`
* `tabulate`

You can find the full list in `requirements.txt`.

---

## Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or pull requests, feel free to open an issue or submit PRs.

This is my first Python package project, built as a learning experience, so I truly appreciate constructive feedback and ideas for improvement.

---

## License

This project is licensed under the MIT License.

---

## A Note on AI

This README was written with the help of Chat GPT. I also used Chat GPT to generate the synthetic data used in the tests and in the notebook demo. **All code was written by me!** You can visit `chat-gpt-prompt.md` to see the **exact** prompt I used to help me reason through problems and think critically while designing seachlite. :)

---

## Future Plans

Here are some features I'm working on adding for future versions of `searchlite`:

- [ ] Support for embedding models other than "all-MiniLM-L6-v2"
- [ ] Create "lightweight" alternative that doesn't require loading sentence-transformers
- [ ] Support for other similarity measures
- [ ] Implement similarity metrics from scratch and remove scikit-learn dependency
- [ ] Support for adding normalization to models that do not automatically normalize their embeddings
- [ ] Functions to allow for easy conversion of a pandas dataframe to a Document object
- [ ] Support for keyword, TF-IDF, BM25, and fuzzy matching search
- [ ] Support for weighted hybrid search
- [ ] Support for reranking search
- [ ] Support for built-in preprocessing of texts
- [ ] Built-in support for filtering query results by metadata fields
- [ ] Caching of embeddings to avoid re-computation across sessions
- [ ] CLI interface for querying datasets without writing code
- [ ] Support for indexing larger datasets with Faiss or Annoy
- [ ] Integration with Jupyter widgets for interactive exploration
- [ ] JSON/YAML import/export for datasets and results
- [ ] Caching for larger embedding tasks


---

## Contact

The repository is open on GitHub:
[https://github.com/r0hankrishnan/searchlite](https://github.com/r0hankrishnan/searchlite)

Feel free to open issues or reach out with questions or suggestions!

---

Thank you for checking out **searchlite**!
