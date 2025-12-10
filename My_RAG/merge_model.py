import os

def merge_files(directory, output_filename, chunk_prefix):
    output_path = os.path.join(directory, output_filename)
    
    if os.path.exists(output_path):
        return

    
    parts = sorted([f for f in os.listdir(directory) if f.startswith(chunk_prefix)])
    if not parts:
        return

    with open(output_path, 'wb') as outfile:
        for part in parts:
            part_path = os.path.join(directory, part)
            with open(part_path, 'rb') as infile:
                outfile.write(infile.read())
    
    print(f"mering complete")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models", "bge-reranker-v2-m3")
    
    merge_files(
        directory=model_dir,
        output_filename="model.safetensors",
        chunk_prefix="model.safetensors.part_"
    )