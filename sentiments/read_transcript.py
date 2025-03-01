def read_transcript(input_file):
    full_text = None
    with open(input_file, "r") as f:
        full_text = f.read().strip()

    if not full_text:
        print("Input file is empty. Nothing to adjust.")
    return full_text