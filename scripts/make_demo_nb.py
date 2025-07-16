import nbformat as nbf
import os
from basic_nb import create_basic_nb
from embedder_nb import create_embedder_nb

def create_demo_notebook():

    # Get basic notebook info
    nb_type = input("Basic or Embedder NB: ").strip().lower().replace("NB", "").replace(" ", "") # Choose between two demo NB templates
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
    if nb_type == "basic":
        notebook = create_basic_nb(title = title, version = version, intro = intro)
    else:
        notebook = create_embedder_nb(title = title, version = version, intro = intro)

    with open(f"./examples/{final_file_name}", "w") as f:
        nbf.write(notebook, f)
        
    print(f"Demo notebook created: {final_file_name}")

if __name__ == "__main__":
    create_demo_notebook()
