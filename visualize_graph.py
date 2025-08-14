# visualize_graph.py
"""
This script builds the LangGraph application from graph.py and saves a
visualization of its structure to an image file in the project's root directory.

This is designed for a graph built with StateGraph.

To run this script from your terminal:
  python visualize_graph.py
"""

from pathlib import Path
from graph import build_app

def generate_graph_visualization():
    """
    Builds the LangGraph application and saves a PNG image of its structure.
    """
    # --- 1. Define the output path for the image ---
    # This will save the image in the current directory where the script is run.
    output_path = Path("graph_visualization.png")

    # --- 2. Build the LangGraph app ---
    # This imports and calls the build_app function from your graph.py file.
    # build_app() compiles your StateGraph into a runnable 'app' object.
    app = build_app()

    print("Successfully built the LangGraph application.")
    print("Generating visualization...")

    try:
        # --- 3. Generate the graph image as bytes ---
        # The .get_graph() method provides access to the underlying graph structure.
        # .draw_mermaid_png() renders this structure into a PNG image.
        image_bytes = app.get_graph().draw_mermaid_png()

        # --- 4. Save the image bytes to the specified file ---
        with open(output_path, "wb") as f:
            f.write(image_bytes)
            
        print(f"✅ Success! Graph visualization saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ Error generating visualization.")
        print("Please ensure you have installed graphviz and the required Python packages:")
        print("  - System: `brew install graphviz` (macOS) or `sudo apt-get install graphviz` (Linux)")
        print("  - Python: `pip install pygraphviz pillow`")
        print(f"  - Underlying error: {e}")

if __name__ == "__main__":
    generate_graph_visualization()