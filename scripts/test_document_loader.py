from ai_orchestrator.loaders.document_loader import load_document


content = load_document(
    "sample_data/documents/sick_leave.md"
)

print(content)