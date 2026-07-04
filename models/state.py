"""
Compat shim - DO NOT put field definitions here.

Member 1's agents/jd_parser.py, agents/screening_agent.py and
nodes/data_loader.py all import `from models.state import
RecruitmentState`. Rather than editing three working files just to
change an import path, this shim re-exports the ONE canonical state
definition that Member 4 owns at the project root (`state.py`), so
there is never a second, divergent copy of RecruitmentState floating
around.
"""
from state import RecruitmentState  # noqa: F401
