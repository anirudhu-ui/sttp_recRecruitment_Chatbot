"""
RecruitFlow AI - Graph Visualization (Member 4)
Run: python generate_graph.py
Saves workflow/recruitment_graph.png
"""
import os

from langgraph.checkpoint.memory import MemorySaver

from graph import build_graph

os.makedirs("workflow", exist_ok=True)

# A throwaway in-memory checkpointer is enough just to compile+draw the
# graph shape - visualization doesn't need real persistence.
graph = build_graph(MemorySaver())
graph.get_graph().draw_mermaid_png(output_file_path="workflow/recruitment_graph.png")
print("Saved workflow/recruitment_graph.png")
