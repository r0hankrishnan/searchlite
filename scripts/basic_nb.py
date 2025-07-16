import nbformat as nbf

def create_basic_nb(title:str, version:float, intro:str):
    BASIC_NB = nbf.v4.new_notebook()
    BASIC_NB.cells = [
        nbf.v4.new_markdown_cell(f"# {title} v{version} \n{intro}"),
        nbf.v4.new_markdown_cell("In this notebook we'll load a sample text data set with some metadata, split the \
            dataframe into the text and its metadata, load it into `searchlite`, and perform/display a semantic search."
            ),
        nbf.v4.new_markdown_cell("First, import your dependencies to load your data. For this example, \
            you'll only need pandas (for loading in our example data) and os (for defining the file path to our example data)."),
        nbf.v4.new_code_cell(
            """
import pandas as pd
import os
            """
            ),
        nbf.v4.new_markdown_cell("## Import and look at data"),
        nbf.v4.new_markdown_cell("Next, define the path to the sample data. In this case it is in the data folder.\
            After defining the path, use pandas to load in the csv file as a data frame."),
        nbf.v4.new_code_cell(
            "sample_df = pd.read_csv(\n"
            "   os.path.join(os.getcwd(), '../data/synthetic_data.csv'),\n"
            "   index_col=0\n)"),
        nbf.v4.new_markdown_cell("Let's take a look at our sample data below. The data consists of 15 distinct pieces \
            of text with corresponding id and category values. Each text topic is quite different so you can test the \
                semantic search with different queries to see if the results makes sense."),
        nbf.v4.new_code_cell("sample_df"),
        nbf.v4.new_markdown_cell("Before initializing the `Document` class, you need to split the dataframe into the \
            text you want to embed and it's corresponding metadata (shown below). You can accomplish this by simply \
                isolating the text column and by using the .to_dict() method to convert the metadata columns into a \
                    list of dictionaries, with each entry corresponding to a row in the dataframe."),
        nbf.v4.new_code_cell(
            """
sample_texts = sample_df["text"]
sample_metadata = sample_df[["id", "category"]].to_dict(orient = "records")
            """),
        nbf.v4.new_code_cell("sample_texts[0:3]"),
        nbf.v4.new_code_cell("sample_metadata[0:3]"),
        nbf.v4.new_markdown_cell("## Use searchlite to embed text and run semantic search"),
        nbf.v4.new_markdown_cell("Now, you can instantiate the `Document` class. As shown below, both the text and metadata \
            are saved as attributes. Before performing search, you must generate embeddings for the texts stored within the \
                `Document` instance. For the basic demo, you only need to import `Document` from `searchlite.document`.\nNote\
                    that if you do not specify a model, `searchlite` automatically imports the `SkTFIDFEmbedder` class which \
                        implements scikit-learn's TFIDF Vectorizer. Upon initializing your `Document`, the SkTFIDFEmbedder will \
                            automatically be fit to your texts."),
        nbf.v4.new_code_cell("from searchlite.document import Document"),
        nbf.v4.new_code_cell("doc = Document(texts = sample_texts, metadata = sample_metadata)"),
        nbf.v4.new_code_cell("doc"),
        nbf.v4.new_markdown_cell("Run the .embed() method to run scikit-learn's TFIDF Vectorizer. If you want to use a different \
            source for your embedding model, check out the other example notebooks to see how to initialize an embedder and pass it to your `Document`."),
        nbf.v4.new_code_cell("doc.embed()"),
        nbf.v4.new_markdown_cell("After generating your text embeddings, you can run semantic search on your text corpus by using the \
            .query() method. Your query will be embedded into a vector and compared against your text corpus using cosine similarity. \
                By default, .query() returns the top 3 matches but this can be changed by modifying the **top_k** parameter.\nAs you can \
                    see from the cell below, .query() returns a list of dictionaries with each dictionary containing the metadata and text \
                        of the identified matches."),
        nbf.v4.new_code_cell(
            """
res = doc.query(query_text = 'wireless earbuds with good battery life')
res
            """),
        nbf.v4.new_markdown_cell(
            """
The `Document` class has three options to nicely display the results of your semantic search in the terminal: f-string, pprint, and tabulate.

- "f-string" outputs a custom f-string (defined in document.py)

- "pprint" leverages the pprint package to display a list of dictionaries of the top k results

- "tabulate" leverages the tabulate library to display a table of the top k results.
            """
        ),
        nbf.v4.new_code_cell("doc.display_results(output_list_dict = res, style = 'f-string')"),
        nbf.v4.new_code_cell("doc.display_results(output_list_dict = res, style = 'pprint')"),
        nbf.v4.new_code_cell("doc.display_results(output_list_dict = res, style = 'tabulate')")
        ]
    
    return BASIC_NB