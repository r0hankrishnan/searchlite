from searchlite.document import Document
import pandas as pd
import pytest
import os

current_file_path = os.path.dirname(__file__)
data_path = os.path.join(current_file_path, "../data/synthetic_data.csv")

sample_df = pd.read_csv(data_path, index_col=0)
sample_texts = sample_df["text"]
sample_metadata = sample_df[["id", "category"]].to_dict(orient = "records")

def test_initialization():
    doc = Document(texts = sample_texts, metadata = sample_metadata)
    assert(doc.texts == sample_texts.tolist())
    assert(doc.metadata == sample_metadata)
    assert(doc.embeddings == None)
    
    
def test_embedding():
    doc = Document(texts = sample_texts, metadata = sample_metadata)
    doc.embed()
    assert(doc.embeddings is not None)
    assert(len(doc.embeddings) == len(sample_texts))
    
def test_query():
    doc = Document(texts = sample_texts, metadata = sample_metadata)
    doc.embed()
    res = doc.query(query_text = "wireless earbuds with good battery life")
    assert(isinstance(res, list))
    assert(len(res) == 3)
    assert all("text" in r for r in res)
    assert all("id" in r for r in res)
    
def test_query_error_if_not_embedded():
    doc = Document(texts = sample_texts, metadata = sample_metadata)
    with pytest.raises(ValueError):
        doc.query(query_text = "wireless earbuds with good battery life")

def test_display_options(capsys):
    doc = Document(texts = sample_texts, metadata = sample_metadata)
    doc.embed()
    res = doc.query(query_text = "wireless earbuds with good battery life")
    
    doc.display_results(res, "f-string")
    captured = capsys.readouterr()
    assert "text:" in captured.out
    
    doc.display_results(res, "pprint")
    captured = capsys.readouterr()
    assert "[" in captured.out
    
    doc.display_results(res, "tabulate")
    captured = capsys.readouterr()
    assert "text" in captured.out
    
def test_display_invalid_option():
    doc = Document(texts = sample_texts, metadata = sample_metadata)
    doc.embed()
    res = doc.query(query_text = "wireless earbuds with good battery life")
    with pytest.raises(ValueError):
        doc.display_results(res, "other")