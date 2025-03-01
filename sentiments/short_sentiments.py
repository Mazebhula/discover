from transformers import pipeline

def adjust_short_transcription(input_text):
    try:
        # Load summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Summarize to a short length
        input_length = len(input_text.split())
        min_length = min(max(10, input_length - 10), 10)
        summary = summarizer(input_text, max_length=15, min_length=min_length, do_sample=False)
        summary_text = summary[0]["summary_text"]
        
        # Adjust to 50 characters
        final_text = summary_text[:62]  # Truncate if longer
        if len(final_text) < 62:
            final_text = final_text.ljust(62)  # Pad with spaces if shorter
        
        return final_text
    
    except Exception as e:
        return f"Error: {str(e)}"[:50].ljust(50)
