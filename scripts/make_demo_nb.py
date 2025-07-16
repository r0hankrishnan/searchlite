import nbformat as nbf
import os

def create_demo_notebook():

    # Get basic notebook info
    file_name = input("Name the notebook file: ").strip()
    clean_file_name = file_name.strip().replace(".ipynb", "").lower().replace(" ", "_")
    final_file_name = f"{clean_file_name}.ipynb"
           
    title = input("Title the notebook: ").strip()
            
    while True:
        try:
            version = float(input("What version are you demoing? "))
            break
        except Exception as e:
            print("Must be a valid float. Try again.")
            
    intro = input("Write a brief intro for the notebook: ").strip()

    os.makedirs("examples", exist_ok=True) # Make examples folder if DNE
    
    # Create notebook scaffold   
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title} v{version} \n{intro}"),
        nbf.v4.new_markdown_cell("First, import your dependencies. For this example, we need searchlite, pandas \
            (for loading in our example data), and os (for defining the file path to our example data)."),
        nbf.v4.new_code_cell(
            """
            from searchlite.document import Document
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
        nbf.v4.new_markdown_cell("## Use searchlite to embed text an run semantic search"),
        nbf.v4.new_markdown_cell("Now, you can initialize our `Document` class. As shown below, both the text and metadata \
            are saved as attributes. Before performing search, you must generate embeddings for the texts stored within the `Document` instance.")
        ]

    with open(f"./examples/{final_file_name}", "w") as f:
        nbf.write(nb, f)
        
    print(f"Demo notebook created: {final_file_name}")

if __name__ == "__main__":
    create_demo_notebook()
