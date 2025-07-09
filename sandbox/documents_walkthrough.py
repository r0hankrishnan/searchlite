from searchlite.document import Document
import pandas as pd
import os

current_file_path = os.path.dirname(__file__)
data_path = os.path.join(current_file_path, "../data/synthetic_data.csv")

sample_df = pd.read_csv(data_path, index_col=0)
sample_texts = sample_df["text"]
sample_metadata = sample_df[["id", "category"]].to_dict(orient = "records")

doc = Document(texts = sample_texts, metadata = sample_metadata)

doc

doc.embed()

res = doc.query(query_text = "wireless earbuds with good battery life")

res

doc.display_results(res, style = "tabulate")