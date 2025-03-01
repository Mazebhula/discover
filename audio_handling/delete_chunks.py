import os

def delete_chunks(chunk_files):
    """
    Deletes the specified chunk files from the filesystem.
    
    Args:
        chunk_files (list): List of file paths to delete.
    """
    for chunk_file in chunk_files:
        try:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
                print(f"Deleted: {chunk_file}")
            else:
                print(f"File not found: {chunk_file}")
        except Exception as e:
            print(f"Error deleting {chunk_file}: {str(e)}")