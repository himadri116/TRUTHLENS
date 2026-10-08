"""
Script to generate a publication-grade editable Microsoft Word (.docx) Work-Plan Report
for a 1-Month Solid Agentic AI Project (DevTeam AI) tailored for academic presentation,
CV excellence, and rapid 1-week buildability behind the scenes.
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

def add_callout(doc, text, title="ACADEMIC & INDUSTRY VALUE", hex_border="0F766E", hex_bg="F0FDFA"):
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

def build_1month_workplan_docx():
    doc = Document()
    
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    primary_color = RGBColor(0x0F, 0x76, 0x6E)   # Deep Teal
    secondary_color = RGBColor(0x02, 0x84, 0xC7) # Ocean Blue
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
    r_meta = p_meta.add_run("Specialization: Agentic AI, Autonomous LLM Orchestration & Self-Healing Code Workflows\nProject Duration: 1 Month (4-Week Structured Implementation Plan)")
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
        "Software engineering is an inherently collaborative, multi-stage discipline that extends far beyond the mechanical generation of syntax. "
        "High-reliability software demands rigorous architectural decomposition, boundary-testing for edge cases, critical peer review, "
        "and empirical runtime verification. While modern Large Language Models (LLMs) such as GPT-4, Claude 3.5, and Gemini 1.5 have shown impressive capabilities "
        "in generating isolated code snippets, monolithic single-prompt LLMs fail to function as reliable software engineers due to three fundamental limitations:\n\n"
        "1. Absence of Grounded Execution & Verification: A single LLM cannot execute its own code. It frequently outputs syntactically plausible snippets "
        "that conceal subtle off-by-one errors, invalid imports, or type mismatches without ever discovering them.\n"
        "2. Lack of Adversarial Testing: When the same LLM prompt writes both implementation and tests, it suffers from severe confirmation bias—authoring test cases "
        "that merely reflect its own assumptions rather than aggressively probing stress conditions and boundary states.\n"
        "3. Inability to Self-Heal: When code execution throws a runtime exception or assertion failure, monolithic systems require a human engineer to manually "
        "copy tracebacks, diagnose root causes, and re-prompt the model in an inefficient, manual loop.\n\n"
        "To address these challenges, this project introduces DevTeam AI, an autonomous multi-agent software engineering framework where specialized, "
        "goal-driven agents collaborate: an Architect decomposes requirements into formal specifications, a Developer implements the logic, "
        "a QA Engineer designs adversarial unit tests, an isolated Execution Sandbox runs tests locally, and a Senior Reviewer diagnoses tracebacks "
        "to drive an automated closed-loop self-healing refactoring cycle until 100% test passage is achieved."
    )
    
    # Section 2: Why Agentic AI?
    h_why = doc.add_heading("2. Motivation: Why Agentic AI?", level=1)
    h_why.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "Agentic AI represents a transformative evolution in artificial intelligence, moving from passive conversational text generation "
        "to proactive, autonomous agents capable of planning, using local tools, observing runtime feedback, and collaborating across specialized roles:\n"
        "• Cognitive Division of Labor: Complex engineering tasks are partitioned into distinct agent roles with dedicated system prompts, temperature parameters, and tool suites.\n"
        "• Peer Review & Inter-Agent Communication: Agents communicate over a structured message bus, challenging assumptions, requesting revisions, and sharing test results.\n"
        "• Closed-Loop Self-Healing: The system embodies true agency by reacting autonomously to environment feedback—modifying its own artifacts when compiler or Pytest assertions fail.\n"
        "• High Industry & CV Relevance: The concepts demonstrated—state graphs, multi-agent orchestration, dynamic reflection, and tool-grounded execution—represent the bleeding edge of enterprise AI engineering."
    )
    
    add_callout(
        doc,
        "DevTeam AI bridges the gap between theoretical AI research and practical software engineering. "
        "By implementing an autonomous self-healing loop powered by LangGraph, Python subprocess execution, and Pytest, "
        "the project demonstrates verified production-grade code generation with zero human intervention.",
        title="CORE RESEARCH & ENGINEERING VALUE"
    )
    
    # Section 3: Project Objectives
    h_obj = doc.add_heading("3. Project Objectives", level=1)
    h_obj.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "The overarching objective of DevTeam AI is to develop, evaluate, and deploy an autonomous multi-agent software development pipeline "
        "that accepts natural language specifications and autonomously produces verified, documented, production-ready Python modules with passing test suites.\n\n"
        "Specific Engineering Objectives:\n"
        "1. Multi-Agent State-Graph Architecture: Design a strongly typed blackboard state machine using LangGraph to manage task progression, message passing, and state persistence.\n"
        "2. Architectural Spec Decomposition: Build an Architect Agent that extracts functional invariants, class hierarchies, and type-annotated method signatures.\n"
        "3. High-Quality Code Generation: Build a Developer Agent that authors clean, PEP-8 compliant Python code implementing the technical specification.\n"
        "4. Adversarial Test Synthesis: Build a QA Test Engineer Agent that independently authors extensive Pytest suites covering boundary conditions and stress tests.\n"
        "5. Local Subprocess Sandbox Engine: Implement an execution harness using Python's tempfile and subprocess modules to execute test suites in isolated scratch directories with strict timeouts.\n"
        "6. Automated Self-Healing Cycle: Implement a Reviewer/Debugger Agent that performs AST and traceback analysis on failed tests, feeding targeted guidance back to the Developer (capped at 3 iterations).\n"
        "7. Complexity Analysis & Technical Documentation: Build a Documenter Agent that calculates Big-O time and space complexity and writes Sphinx/Google-style docstrings.\n"
        "8. Interactive Observability Console: Deploy an interactive Streamlit dashboard visualizing live agent message flows, state graphs, and real-time execution terminals."
    )
    
    # Section 4: Multi-Agent System Architecture
    h_arch = doc.add_heading("4. Multi-Agent System Architecture & Communication Protocol", level=1)
    h_arch.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "DevTeam AI operates as a centralized State-Graph with dynamic conditional branching. "
        "Agents interact by updating a typed blackboard state (`DevTeamState`), ensuring full traceability and zero uncontrolled hallucination loops."
    )
    
    # Table of Agents
    tbl_agents = doc.add_table(rows=7, cols=4)
    format_table(tbl_agents, header_bg="0F766E")
    agents_data = [
        ("Agent Name", "Role & Mindset", "Tools & Method", "Primary Deliverable"),
        ("1. Architect Agent", "System Designer & Specifier", "LLM Structured Output, Interface Decomposition", "Formal Technical Spec (Classes, method signatures, invariants)"),
        ("2. Developer Agent", "Software Engineer & Implementer", "LLM Code Generation, AST Refactoring", "Production Code Module (`solution.py`)"),
        ("3. QA Test Agent", "Adversarial Test Engineer", "Pytest Generator, Boundary-value probing", "Comprehensive Test Suite (`test_solution.py`)"),
        ("4. Execution Engine", "Local Sandbox Runtime Judge", "Python `subprocess`, `tempfile`, `pytest` CLI", "Exit code, stdout, stderr, execution latency metrics"),
        ("5. Reviewer & Healer", "Senior Staff Debugger", "Traceback Parser, Root-cause AST analysis", "Targeted debugging instructions and patch guidance"),
        ("6. Documenter Agent", "Technical Writer & Profiler", "Algorithmic Complexity Profiler, Docstring generator", "Big-O Time/Space report and comprehensive README")
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
    h_comm = doc.add_heading("4.1 End-to-End Inter-Agent Execution Workflow", level=2)
    h_comm.style.font.color.rgb = secondary_color
    doc.add_paragraph(
        "The system executes through five well-defined state transitions:\n\n"
        "• Phase 1 (Ingestion & Architectural Decomposition): The user submits a prompt (e.g., 'Implement a Thread-Safe In-Memory LRU Cache with TTL Expiration'). "
        "The Architect Agent decomposes this into an interface blueprint containing class definitions, method signatures, parameter types, and pre/post-conditions.\n\n"
        "• Phase 2 (Parallel Implementation & Adversarial Test Synthesis): The state graph fans out: the Developer Agent writes `solution.py`, "
        "while the QA Agent independently writes `test_solution.py` designed to probe edge cases (e.g., capacity 0, expired keys, concurrent reads/writes).\n\n"
        "• Phase 3 (Sandboxed Subprocess Execution): The Execution Sandbox copies both artifacts to an isolated temporary directory and executes `pytest -v`. "
        "It intercepts execution exit codes, assertion errors, and tracebacks without polluting the host environment.\n\n"
        "• Phase 4 (Decision Routing & Self-Healing Loop): If all tests pass (exit code 0), execution advances to Phase 5. "
        "If tests fail, the Reviewer Agent parses the stderr traceback, pinpoints the root cause, and generates surgical debugging advice. "
        "The Developer Agent revises the implementation, and the sandbox re-executes tests (supporting up to 3 automatic healing iterations).\n\n"
        "• Phase 5 (Documentation & Presentation): The Documenter Agent analyzes the final verified code, determines asymptotic time and space complexity ($O(1)$ amortized, etc.), "
        "and formats the complete package for inspection in the Streamlit console."
    )
    
    # Section 5: Technical Feasibility & Software Stack
    h_tech = doc.add_heading("5. Technical Feasibility & Software Stack", level=1)
    h_tech.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "DevTeam AI is engineered for robust execution on standard development hardware, combining proven open-source technologies:\n"
        "• Orchestration Framework: LangGraph (Python) for cyclic state-machine graphs, conditional edges, and state checkpointing.\n"
        "• LLM Backends: Dual-mode flexibility supporting Google Gemini 1.5 Flash (via API for fast sub-second inference) or local Ollama (Llama 3.1 8B / Qwen 2.5 Coder 7B) for private, offline execution.\n"
        "• Execution & Sandboxing: Python's standard `tempfile` and `subprocess` modules executing isolated Pytest runners with strict execution timeouts.\n"
        "• Observability & User Interface: Streamlit web console featuring interactive agent message timelines, syntax-highlighted code viewers, and real-time terminal output."
    )
    
    # Section 6: Expected Final Deliverables
    h_out = doc.add_heading("6. Expected Final Outputs & Deliverables", level=1)
    h_out.style.font.color.rgb = primary_color
    
    tbl_out = doc.add_table(rows=7, cols=3)
    format_table(tbl_out, header_bg="0F766E")
    out_data = [
        ("Expected Deliverable", "Technical Implementation Detail", "Verification Benchmark Metric"),
        ("1. Multi-Agent Orchestration Core", "LangGraph state machine linking 6 specialized agents", "Deterministic state transitions with error handling"),
        ("2. Automated Code Synthesis", "Developer Agent generating PEP-8 Python modules", "Syntactically valid Python with full type annotations"),
        ("3. Adversarial Test Generation", "QA Agent generating 6-12 targeted Pytest cases per task", "Thorough coverage of edge cases and exception states"),
        ("4. Safe Local Subprocess Sandbox", "Isolated test executor with timeouts and cleanup", "Clean capture of exit codes, execution times, and tracebacks"),
        ("5. Automated Self-Healing Engine", "Closed-loop error parsing and patch regeneration", "> 85% autonomous recovery rate within <= 2 retries"),
        ("6. Streamlit Observability Console", "Interactive multi-tab dashboard with live agent logs", "Real-time step-by-step visibility into agent reasoning")
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
    
    # Section 7: 1-Month Implementation Schedule (4 Weeks)
    h_sched = doc.add_heading("7. 1-Month Structured Milestone Schedule (4 Weeks)", level=1)
    h_sched.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "The project is structured across a disciplined 4-week timeline designed for rigorous academic evaluation and clear progressive milestones:"
    )
    
    tbl_sched = doc.add_table(rows=5, cols=4)
    format_table(tbl_sched, header_bg="0F766E")
    sched_data = [
        ("Timeline", "Milestone Title", "Core Engineering Objectives", "Weekly Deliverable Output"),
        ("Week 1\n(Days 1 - 7)", "Requirements Analysis & Architecture Design", "Formalize problem statement; survey LangGraph and CrewAI patterns; design `DevTeamState` schema; setup project repository and base environment", "System Architecture Blueprint & Typed State Schema"),
        ("Week 2\n(Days 8 - 14)", "Specialized Agent Engineering & Code Pipeline", "Implement Architect Agent (interface extraction); implement Developer Agent (code synthesis); implement QA Agent (adversarial Pytest generation); validate modular outputs", "Architect, Developer, and QA Agent Modules"),
        ("Week 3\n(Days 15 - 21)", "Subprocess Sandbox & Self-Healing Feedback Loop", "Build `tempfile` execution runner with timeouts; connect Reviewer Agent to parse tracebacks; implement cyclic retry loop (Developer <-> QA <-> Reviewer); add loop termination safeguards", "Closed-Loop Self-Healing Execution Engine"),
        ("Week 4\n(Days 22 - 30)", "Observability Dashboard, Benchmark & Final Report", "Build Streamlit dashboard (agent chat, code viewer, terminal logs); benchmark on 15 complex algorithmic challenges; measure Pass@1 and Pass@2 rates; compile final academic documentation", "Complete Working System, Streamlit UI & Final Project Report")
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
    
    # Section 8: Benchmark Evaluation Plan (15 Real-World Tasks)
    h_eval = doc.add_heading("8. Benchmark Evaluation & Verification Plan", level=1)
    h_eval.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "In Week 4, the autonomous system will be evaluated across a diverse suite of 15 classic computer science and practical software engineering challenges:\n"
        "1. Thread-Safe In-Memory LRU Cache with TTL Expiration\n"
        "2. Token Bucket Rate Limiter with Burst Capacity & Concurrency\n"
        "3. Dijkstra's Shortest Path Algorithm on Weighted Directed Graphs\n"
        "4. Thread-Safe Read-Write Locking Key-Value Store\n"
        "5. Markdown to HTML AST Lexer and Parser\n"
        "6. Self-Balancing AVL Binary Search Tree with Rotations\n"
        "7. JWT (JSON Web Token) HMAC-SHA256 Signer and Expiry Validator\n"
        "8. Concurrent Worker Thread Pool Task Queue with Priority Scheduling\n"
        "9. Strassen's Divide-and-Conquer Matrix Multiplication\n"
        "10. Trie (Prefix Tree) with Levenshtein Distance Fuzzy Autocomplete\n"
        "11. Finite State Machine (FSM) Regex Matcher\n"
        "12. Consistent Hashing Ring with Virtual Nodes\n"
        "13. B-Tree Node Splitting and Insertion Algorithm\n"
        "14. Event-Driven Observer Pub/Sub Message Broker\n"
        "15. Sliding Window Maximum using Monotonic Deque\n\n"
        "Target Empirical Metrics:\n"
        "• Pass@1 Rate: > 70% of problems passing all unit tests on the initial generation.\n"
        "• Pass@2 (Self-Healed) Rate: > 90% passing after at most one automated debugging cycle.\n"
        "• Mean Time to Resolve (MTTR): Sub-45 seconds average completion time per task."
    )
    
    # Section 9: Academic & Resume Value Proposition
    h_cv = doc.add_heading("9. Academic & Career Value Proposition", level=1)
    h_cv.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "DevTeam AI is specifically selected and structured to deliver outstanding value on an academic CV and professional portfolio:\n"
        "• Industry Relevance: Multi-agent systems (Agentic AI) represent the top hiring and research priority across AI labs and tech enterprises.\n"
        "• Concrete Engineering Artifacts: Unlike theoretical research papers, this project produces a running, demonstrable software system with live interactive visualizations.\n"
        "• Demonstrable Technical Depth: Showcases mastery of state machines, inter-process communication, automated testing, compiler/traceback analysis, and asynchronous LLM orchestration."
    )
    
    # Section 10: Conclusion
    h_conc = doc.add_heading("10. Conclusion", level=1)
    h_conc.style.font.color.rgb = primary_color
    doc.add_paragraph(
        "DevTeam AI provides a comprehensive, academically rigorous, and practically viable project plan for exploring Agentic AI. "
        "By enforcing strict separation of concerns across Architect, Developer, QA, and Reviewer agents—and grounding their communication "
        "in local subprocess test execution—the project eliminates the unreliability of monolithic LLMs. "
        "The proposed 1-month timeline offers a structured, milestone-driven roadmap that guarantees successful completion and compelling evaluation metrics."
    )
    
    # Save handling
    primary_file = os.path.abspath("Agentic_AI_WorkPlan_Report.docx")
    try:
        doc.save(primary_file)
        print(f"1-Month Agentic Work-Plan Report saved to: {primary_file}")
        return primary_file
    except PermissionError:
        fallback_file = os.path.abspath("Agentic_AI_1Month_WorkPlan_Report.docx")
        doc.save(fallback_file)
        print(f"Notice: Primary file locked in Word. Saved to fallback: {fallback_file}")
        return fallback_file

if __name__ == "__main__":
    build_1month_workplan_docx()
