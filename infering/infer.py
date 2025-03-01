from transformers import pipeline
import os

def adjust_transcription(input_file="transcription.txt", output_file="adjusted_transcription.txt"):
    """
    Adjusts the entire contents of a transcription file for better clarity using a Hugging Face model.
    Processes the file as a whole to retain context across chunks.

    Args:
        input_file (str): Path to the input transcription file.
        output_file (str): Path to save the adjusted transcription.
    """
    # Load the Hugging Face summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    # Read the entire transcription file as one string
    with open(input_file, "r") as f:
        full_text = f.read().strip()

    if not full_text:
        print("Input file is empty. Nothing to adjust.")
        return

    # Calculate dynamic length constraints based on the full text
    input_length = len(full_text.split())
    max_length = min(max(20, input_length + 20), 300)  # Cap at 300 words for coherence
    min_length = min(max(10, input_length - 10), max_length - 10)

    try:
        # Process the entire text at once
        result = summarizer(full_text, max_length=max_length, min_length=min_length, do_sample=False)
        adjusted_text = result[0]["summary_text"]

        # Print original vs adjusted for comparison
        # print("Original Transcription:")
        # print(full_text)
        # print("\nAdjusted Transcription:")
        # print(adjusted_text)

        # Write the adjusted transcription to a new file
        with open(output_file, "w") as f:
            f.write(adjusted_text)
        print(f"\nAdjusted transcription saved to {output_file}")

    except Exception as e:
        print(f"Error processing transcription: {str(e)}")
        with open(output_file, "w") as f:
            f.write(full_text)  # Fallback to original if error occurs
        print(f"Fallback: Original transcription saved to {output_file}")

if __name__ == "__main__":
    adjust_transcription()