from pathlib import Path


def load_document(file_path: str) -> str:
    """
    Load source document content.
    """

    path = Path(file_path)

    return path.read_text(
        encoding="utf-8"
    )