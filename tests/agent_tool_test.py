# tests/agent_tool_test.py
from agent_tools import aws_docs_search

def main():
    q = "How do I enable default encryption on an S3 bucket with a KMS key? Provide steps."
    out = aws_docs_search.invoke({"question": q, "n_items": 2})  # LC tool interface
    print("RAW OUTPUT (trimmed):\n", out[:1200])
    assert "Context:" in out, "No Context block returned"
    assert "\nSources:\n" in out, "No Sources block returned"
    print("\nPASS: agent tool returned context + sources.")
if __name__ == "__main__":
    main()
