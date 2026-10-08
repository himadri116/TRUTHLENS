"""
Script to generate a publication-grade editable Microsoft Word (.docx) Work-Plan Report
for a 1-Week Viable Multi-Agent AI Project:
Title: DevTeam AI: Autonomous Multi-Agent System for Collaborative Software Development, Peer Review, and Self-Healing Test Verification
Designed specifically for a realistic 7-day implementation schedule.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_callout(doc, text, title="1-WEEK FEASIBILITY GUARANTEE", hex_border="0F766E", hex_bg="F0FDFA"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, hex_bg)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{hex_border}"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(4)
    run_title = p.add_run(f"⚡ {title}: ")
    run_title.bold = True
    run_title.font.size = Pt(10.5)
    run_title.font.color.rgb = RGBColor(0x0F, 0x76, 0x6E)
    
    run_text = p.add_run(text)
    run_text.font.size = Pt(10)
    run_text.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_table(tbl, header_bg="0F766E"):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(tbl.rows):
        for c_idx, cell in enumerate(row.cells):
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            if r_idx == 0:
                set_cell_background(cell, header_bg)
                if cell.paragraphs and cell.paragraphs[0].runs:
                    cell.paragraphs[0].runs[0].font.bold = True
                    cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                set_cell_background(cell, "F8FAFC" if r_idx % 2 == 1 else "FFFFFF")

def build_agentic_workplan_docx(output_filename="Agentic_AI_WorkPlan_Report.docx"):
    doc = Document()
    
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    primary_color = RGBColor(0x0F, 0x76, 0x6E)   # Deep Teal
    secondary_color = RGBColor(0x02, 0x84, 0xC7) # Sky / Ocean Blue
    success_color = RGBColor(0x16, 0x65, 0x34)   # Forest Green
    dark_neutral = RGBColor(0x1F, 0x29, 0x37)
    
    # Title Block
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(14)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("PROJECT WORK-PLAN REPORT")
    r_title.bold = True
    r_title.font.size = Pt(22)
    r_title.font.name = "Calibri"
    r_title.font.color.rgb = primary_color
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(8)
    r_sub = p_sub.add_run("DevTeam AI: Autonomous Multi-Agent System for Collaborative Software Development, Peer Review, and Self-Healing Code Verification")
    r_sub.font.size = Pt(13.5)
    r_sub.bold = True
    r_sub.font.color.rgb = secondary_color
    
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(14)
    r_meta = p_meta.add_run("Specialization: Agentic AI, Autonomous LLM Orchestration & Self-Healing Code Workflows\nProject Duration: 1-Week Rapid Implementation Schedule (7-Day Phased Plan)")
    r_meta.font.size = Pt(10)
    r_meta.font.color.rgb = primary_color
    r_meta.bold = True
    
    p_line = doc.add_paragraph()
    p_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_line.paragraph_format.space_after = Pt(14)
    r_line = p_line.add_run("―" * 55)
    r_line.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    
    # Section 1: Problem Statement
    h_prob = doc.add_heading("1. Problem Statement", level=1)
    h_prob.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "Modern software engineering requires far more than writing raw syntax. High-quality code demands clear architectural design, "
        "comprehensive unit testing for edge cases, critical peer review, and runtime debugging. "
        "While modern Large Language Models (such as GPT-4 and Gemini) can generate code snippets from text prompts, monolithic single-prompt LLMs "
        "fail at building working software for three fundamental reasons:\n\n"
        "1. No Grounded Execution: A single LLM cannot execute its own code. It frequently outputs code with subtle syntax errors, incorrect import names, "
        "or off-by-one algorithmic bugs without ever realizing it.\n"
        "2. Absence of Adversarial Testing: When the same prompt writes both code and tests, it creates a self-fulfilling bias—writing tests that avoid "
        "its own blind spots rather than rigorously probing edge cases.\n"
        "3. Inability to Self-Heal: When a human runs LLM-generated code and encounters a traceback error, the user must manually copy and paste the error back "
        "and forth in a tedious manual loop.\n\n"
        "This project proposes DevTeam AI, a lightweight, highly viable multi-agent system where specialized software agents communicate autonomously: "
        "an Architect designs the blueprint, a Developer writes the code, a Test Engineer writes adversarial unit tests, an Execution Engine runs the tests locally, "
        "and a Reviewer diagnoses runtime failures and guides the Developer through an automated self-healing loop until 100% of tests pass."
    )
    
    # Section 2: Why Agentic AI & 1-Week Viability
    h_why = doc.add_heading("2. Motivation: Why Agentic AI & Why 1-Week Viability?", level=1)
    h_why.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "Agentic AI transforms AI from a passive question-answering assistant into an autonomous team of goal-driven workers that communicate, critique, "
        "and execute real-world actions:\n"
        "• True Multi-Agent Communication: Agents don't just speak to the user; they communicate with each other. The Test Engineer passes failing assertions "
        "to the Reviewer, who provides actionable debugging advice to the Developer.\n"
        "• Self-Healing Feedback Loop: If tests fail, the Developer modifies the code automatically based on the compiler traceback. Human intervention is only needed at the start.\n"
        "• Realistic 1-Week Scope: Unlike complex projects requiring heavy Docker clusters or multi-week dataset training, DevTeam AI uses Python's built-in "
        "tempfile and subprocess modules to execute Pytest locally. This makes the entire pipeline 100% buildable, testable, and demonstrable in exactly 7 days."
    )
    
    add_callout(
        doc,
        "DevTeam AI is specifically scoped for a 7-day implementation. It leverages Python's built-in subprocess and tempfile execution engines, "
        "eliminating heavy infrastructure dependencies while delivering authentic multi-agent communication, automated Pytest execution, and self-healing code repair.",
        title="7-DAY IMPLEMENTATION GUARANTEE"
    )
    
    # Section 3: Project Objectives
    h_obj = doc.add_heading("3. Project Objectives", level=1)
    h_obj.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "The objective of DevTeam AI is to build a fully functional, autonomous software development team in Python that takes a natural language task description "
        "(e.g., 'Build an LRU Cache with TTL expiration') and outputs production-ready, verified code with passing unit tests and documentation within 60 seconds.\n\n"
        "Specific Objectives for the 1-Week Plan:\n"
        "1. Build a multi-agent state graph using LangGraph / CrewAI (or a clean Python state-machine controller).\n"
        "2. Implement an Architect Agent that extracts technical requirements, class interfaces, and method signatures.\n"
        "3. Implement a Developer Agent that writes clean, idiomatic Python code based on the architecture spec.\n"
        "4. Implement a QA Test Engineer Agent that independently writes comprehensive Pytest test suites probing edge cases.\n"
        "5. Implement a Local Execution Engine that runs the generated tests in a temporary directory via Python subprocess and captures stdout/stderr.\n"
        "6. Implement an Automated Self-Healing Loop: If tests fail, the Reviewer diagnoses the error, and the Developer refines the code (up to 3 cycles).\n"
        "7. Build a responsive Streamlit Dashboard displaying live agent message logs, state transitions, code viewers, and terminal test passes."
    )
    
    # Section 4: Multi-Agent System Architecture
    h_arch = doc.add_heading("4. Multi-Agent Architecture & Communication Protocol", level=1)
    h_arch.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "DevTeam AI is structured around a centralized Typed Blackboard State (`DevTeamState`). "
        "Each agent receives the shared state, performs its specialized task using LLM reasoning or local tool execution, and posts its output back to the state."
    )
    
    # Table of Agents
    tbl_agents = doc.add_table(rows=6, cols=4)
    format_table(tbl_agents, header_bg="0F766E")
    agents_data = [
        ("Agent Name", "Role & Personality", "Tools & Method", "Primary Deliverable"),
        ("1. Architect Agent", "System Designer & Spec Writer", "LLM Prompting, Interface Design", "Technical Specification (Classes, methods, invariants)"),
        ("2. Developer Agent", "Software Engineer & Implementer", "LLM Code Generation, Refactoring", "Implementation Code (`solution.py`)"),
        ("3. QA Test Agent", "Adversarial Test Engineer", "Pytest Generator, Edge-case probing", "Unit Test Suite (`test_solution.py`)"),
        ("4. Execution Engine", "Local Runtime & Test Judge", "Python `subprocess`, `tempfile`, `pytest`", "Terminal execution logs, Exit code, Traceback"),
        ("5. Reviewer & Healer", "Senior Debugger & Reviewer", "Traceback Parser, Root-cause analysis", "Debug guidance and patch suggestions for Developer")
    ]
    for r_idx, row in enumerate(agents_data):
        for c_idx, val in enumerate(row):
            cell = tbl_agents.cell(r_idx, c_idx)
            cell.text = val
            if r_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            elif c_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # Communication Workflow
    h_comm = doc.add_heading("4.1 Step-by-Step Inter-Agent Communication Loop", level=2)
    h_comm.style.font.color.rgb = secondary_color
    doc.add_paragraph(
        "Step 1: User Input -> Architect Agent:\n"
        "The user enters a task (e.g., 'Implement a Token Bucket Rate Limiter'). The Architect breaks down the problem, specifies class structures, "
        "method signatures, parameter types, and edge conditions, and posts `tech_spec` to the state.\n\n"
        "Step 2: Architect -> Developer & QA Test Agent (Parallel Dispatch):\n"
        "• The Developer Agent reads `tech_spec` and writes `solution.py`.\n"
        "• Simultaneously, the QA Test Agent reads `tech_spec` and writes adversarial tests in `test_solution.py` (e.g., testing burst capacity, negative tokens, multi-threading).\n\n"
        "Step 3: Developer & QA -> Execution Engine:\n"
        "The Execution Engine writes both files into a temporary directory and executes `pytest test_solution.py -v`. It captures exit code, execution time, and any error tracebacks.\n\n"
        "Step 4: Execution Engine -> Decision Router:\n"
        "• If Exit Code == 0 (All tests pass): Pipeline marks status as `VERIFIED` and passes code to Documenter.\n"
        "• If Exit Code != 0 (Failures or errors): Pipeline triggers the Self-Healing Loop.\n\n"
        "Step 5: Self-Healing Feedback Loop (Reviewer -> Developer):\n"
        "The Reviewer Agent analyzes the exact traceback (e.g., `AssertionError: expected 5 tokens, got 4`). It diagnoses the logical error and passes specific "
        "fix instructions to the Developer Agent. The Developer rewrites the buggy logic, and the Execution Engine re-runs the tests. (Capped at 3 automatic retry cycles).\n\n"
        "Step 6: Final Presentation:\n"
        "The final verified code, passing Pytest output, and complexity analysis are rendered cleanly in the Streamlit UI."
    )
    
    # Section 5: Technical Feasibility & Software Stack
    h_tech = doc.add_heading("5. Technical Feasibility & Software Stack", level=1)
    h_tech.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "DevTeam AI is engineered for rapid 1-week delivery using lightweight, standard Python libraries with zero heavy infrastructure overhead:\n"
        "• Multi-Agent Framework: LangGraph / Custom StateGraph controller (Python) with typed state dictionary.\n"
        "• LLM Engine: Dual-mode support using Google Gemini 1.5 Flash (via free API tier for fast sub-second generation) or local Ollama (Llama 3.1 / Qwen 2.5 Coder).\n"
        "• Local Test Sandbox: Standard Python `tempfile` and `subprocess.run(['pytest', ...], capture_output=True, timeout=10)`.\n"
        "• User Interface: Streamlit web dashboard with interactive tabs (Task Input, Live Agent Chat, Code Viewer, Pytest Terminal Output)."
    )
    
    # Section 6: Expected Final Deliverables
    h_out = doc.add_heading("6. Expected Final Outputs & Deliverables", level=1)
    h_out.style.font.color.rgb = primary_color
    
    tbl_out = doc.add_table(rows=7, cols=3)
    format_table(tbl_out, header_bg="0F766E")
    out_data = [
        ("Promised Final Output", "Implementation Detail", "Verification Target"),
        ("1. Multi-Agent Orchestration Core", "LangGraph state machine linking Architect, Developer, QA, and Reviewer", "Smooth cyclic state transitions with state persistence"),
        ("2. Automated Code Generation", "Developer Agent outputting clean Python code adhering to spec", "Syntactically valid Python code with type hints"),
        ("3. Adversarial Pytest Synthesis", "QA Test Agent generating 5-10 targeted unit tests per task", "Thorough coverage of edge cases and boundary conditions"),
        ("4. Safe Local Subprocess Runner", "Execution Engine running Pytest with timeout and isolation", "Clean capture of exit codes, execution times, and tracebacks"),
        ("5. Automated Self-Healing Loop", "Closed-loop error parsing and patch regeneration", "Successful auto-recovery from test failures within <= 2 retries"),
        ("6. Streamlit Web Dashboard", "Interactive visual console showing agent avatars and live logs", "Real-time step-by-step visibility into agent reasoning")
    ]
    for r_idx, row in enumerate(out_data):
        for c_idx, val in enumerate(row):
            cell = tbl_out.cell(r_idx, c_idx)
            cell.text = val
            if r_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            elif c_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # Section 7: 7-Day Day-by-Day Implementation Schedule
    h_sched = doc.add_heading("7. Proposed 7-Day Implementation Schedule", level=1)
    h_sched.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "To ensure successful completion within one week, the project is organized into 7 daily milestone sprints matching the development pace of our previous project:"
    )
    
    tbl_sched = doc.add_table(rows=8, cols=4)
    format_table(tbl_sched, header_bg="0F766E")
    sched_data = [
        ("Schedule", "Promised Milestone", "Key Daily Engineering Tasks", "Deliverable Output"),
        ("Day 1", "Architecture & State Schema Design", "Design `DevTeamState` schema; setup project structure and LangGraph / controller harness", "`state_schema.py` & Project Scaffold"),
        ("Day 2", "Architect & Developer Agent Setup", "Implement system prompts for Architect and Developer; parse structured markdown code blocks", "Architect & Developer Agent modules"),
        ("Day 3", "QA Test Agent & Subprocess Runner", "Build QA prompt generating Pytest scripts; implement `tempfile` execution runner with timeouts", "QA Agent & `test_runner.py` module"),
        ("Day 4", "Reviewer & Self-Healing Feedback Loop", "Wire execution tracebacks back to Reviewer; build automatic retry loop (Developer <-> QA)", "Working Closed-Loop Self-Healing Pipeline"),
        ("Day 5", "Edge-Case Hardening & Code Documenter", "Add prompt for complexity analysis ($O(n)$ space/time) and docstrings; handle syntax errors", "Documenter Agent & Hardened Logic"),
        ("Day 6", "Interactive Streamlit Web Dashboard", "Build multi-agent UI with agent message bubbles, live status spinners, code editor, and terminal output", "`app.py` Streamlit Dashboard"),
        ("Day 7", "Testing on 10 Benchmark Tasks & Report", "Run evaluation across 10 coding benchmarks (LRU Cache, Rate Limiter, Pathfinding, etc.); compile report", "Final Working System & Benchmark Report")
    ]
    for r_idx, row in enumerate(sched_data):
        for c_idx, val in enumerate(row):
            cell = tbl_sched.cell(r_idx, c_idx)
            cell.text = val
            if r_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            elif c_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Section 8: Evaluation Plan (10 Benchmark Tasks)
    h_eval = doc.add_heading("8. Evaluation & Verification Benchmark Plan", level=1)
    h_eval.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "On Day 7, the system will be evaluated across 10 diverse Python coding benchmarks to measure autonomous repair capabilities:\n"
        "1. LRU Cache with O(1) get/put operations.\n"
        "2. Token Bucket Rate Limiter with burst capacity.\n"
        "3. Dijkstra Shortest Path algorithm on weighted graphs.\n"
        "4. Thread-Safe In-Memory Key-Value Store.\n"
        "5. Markdown to HTML AST Parser.\n"
        "6. Binary Search Tree with AVL balancing.\n"
        "7. JWT Token Validator with expiry checking.\n"
        "8. Concurrent Task Queue with worker thread pool.\n"
        "9. Matrix Multiplication with Strassen's algorithm.\n"
        "10. Trie (Prefix Tree) with fuzzy prefix autocomplete.\n\n"
        "Target Metrics: We aim to achieve > 90% Pass@1 or Pass@2 (passing on first try or after one self-healing revision) with zero human intervention."
    )
    
    # Section 9: Conclusion
    h_conc = doc.add_heading("9. Conclusion", level=1)
    h_conc.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "DevTeam AI offers a practical, engaging, and demonstrably viable demonstration of Agentic AI. "
        "By structuring the problem into clear, specialized roles that communicate and hold each other accountable through executable unit tests, "
        "the project avoids the pitfalls of ungrounded LLM hallucinations. Most importantly, its lightweight Python-native architecture guarantees that "
        "it can be completely developed, tested, and demonstrated within a 1-week timeline."
    )
    
    output_path = os.path.abspath(output_filename)
    try:
        doc.save(output_path)
        print(f"1-Week Agentic Work-Plan Report generated successfully: {output_path}")
        return output_path
    except PermissionError:
        fallback_path = os.path.abspath("Agentic_AI_1Week_WorkPlan_Report.docx")
        doc.save(fallback_path)
        print(f"Notice: '{output_filename}' is open in Word. Saved to fallback: {fallback_path}")
        return fallback_path

if __name__ == "__main__":
    build_agentic_workplan_docx()
