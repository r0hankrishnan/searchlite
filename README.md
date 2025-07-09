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

## Contact

The repository is open on GitHub:
[https://github.com/r0hankrishnan/searchlite](https://github.com/r0hankrishnan/searchlite)

Feel free to open issues or reach out with questions or suggestions!

---

Thank you for checking out **searchlite**!