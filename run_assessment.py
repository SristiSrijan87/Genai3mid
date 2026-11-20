#!/usr/bin/env python3
"""
run_assessment.py - CLI Evaluation Suite for Project 7: AI Research Paper Assistant
Executes the mandatory 5 assessment questions (Q1–Q5), validates strict grounding,
formats terminal output tables, and exports results to assessment_results.json.
"""

import sys
import time
import json
from pathlib import Path

# Add workspace directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import answer, get_status, get_gemini_api_key

ASSESSMENT_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What exact learning rate, number of warmup steps, and Adam optimizer beta values were used in the primary pre-training run described in Section 4?",
        "type": "In-Domain Technical Parameter",
        "expected": "Learning rate 3e-4 (0.0003), 4,000 linear warmup steps, Adam optimizer beta_1 = 0.9, beta_2 = 0.95 (Section 4.2, Page 16)",
        "expected_grounding": "Supported"
    },
    {
        "id": "Q2",
        "question": "Who is the current Chief Executive Officer of OpenAI?",
        "type": "Out-of-Domain Grounding Check",
        "expected": "[not found]",
        "expected_grounding": "[not found]"
    },
    {
        "id": "Q3",
        "question": "By what percentage did the proposed hybrid retrieval architecture improve Exact Match (EM) accuracy on the benchmark dataset compared with the dense-only baseline?",
        "type": "In-Domain Experimental Metric",
        "expected": "14.6% improvement (from 61.2% to 75.8% in Section 5.3, Page 31)",
        "expected_grounding": "Supported"
    },
    {
        "id": "Q4",
        "question": "What GPU cluster configuration (number and model of GPUs) and total training duration, in days, were required to train the 13-billion-parameter model?",
        "type": "In-Domain Compute Infrastructure",
        "expected": "128 NVIDIA H100 (80GB SXM5) GPUs and 18.5 days total training duration (Section 7.1, Page 45)",
        "expected_grounding": "Supported"
    },
    {
        "id": "Q5",
        "question": "What was the worldwide box-office revenue of the film Oppenheimer in 2023?",
        "type": "Out-of-Domain Grounding Check",
        "expected": "[not found]",
        "expected_grounding": "[not found]"
    }
]


def print_banner():
    banner = """
================================================================================
          PROJECT 7: AI RESEARCH PAPER LITERATURE ASSISTANT
                 Q1–Q5 COMPREHENSIVE ASSESSMENT SUITE
================================================================================
Corpus: transformer_rag_research_paper_50p.pdf (50 Pages)
Vector DB: Local ChromaDB (./chroma_db) | Model: gemini-2.5-flash (temp=0.0)
Embedding: text-embedding-004 | Top-K: 5
--------------------------------------------------------------------------------
"""
    print(banner)


def run_all_assessments(output_file: str = "assessment_results.json"):
    print_banner()

    status = get_status()
    print(f"[*] Corpus PDF Status: {'Found (50 Pages)' if status['pdf_exists'] else 'MISSING'}")
    print(f"[*] ChromaDB Persistence: {status['chroma_dir']} (Indexed Chunks: {status['indexed_chunks']})")
    
    api_key = get_gemini_api_key()
    if not api_key:
        print("\n[!] ERROR: GEMINI_API_KEY is not configured in your environment or .env file.")
        print("    Please set GEMINI_API_KEY=your_key in .env to run live model queries.")
        sys.exit(1)

    print("\nExecuting evaluation questions...\n")

    results = []
    total_passed = 0

    for idx, item in enumerate(ASSESSMENT_QUESTIONS, start=1):
        qid = item["id"]
        q_text = item["question"]
        q_type = item["type"]
        expected = item["expected"]
        exp_grounding = item["expected_grounding"]

        print(f"[{idx}/5] Running {qid} ({q_type})...")
        t0 = time.time()
        res = answer(q_text)
        latency = round(time.time() - t0, 2)

        actual_ans = res.get("answer", "")
        sources = res.get("sources", [])

        # Check grounding compliance
        is_not_found = actual_ans.strip().lower() == "[not found]"
        if exp_grounding == "[not found]":
            passed = is_not_found and len(sources) == 0
            compliance_status = "PASSED (Strict [not found])" if passed else "FAILED (Hallucinated/Extra text)"
        else:
            passed = not is_not_found and len(sources) > 0
            compliance_status = "PASSED (Supported with sources)" if passed else "FAILED (Expected answer but got [not found])"

        if passed:
            total_passed += 1

        result_entry = {
            "id": qid,
            "question": q_text,
            "type": q_type,
            "expected": expected,
            "actual_answer": actual_ans,
            "sources": sources,
            "latency_seconds": latency,
            "grounding_status": compliance_status,
            "passed": passed
        }
        results.append(result_entry)

        # Print per-question card
        print(f"    - Query: \"{q_text}\"")
        print(f"    - Status: {compliance_status} ({latency}s)")
        print(f"    - Answer: {actual_ans}")
        print(f"    - Sources: {sources}\n")

    # Print Summary Table
    print("=" * 80)
    print("                        ASSESSMENT SUMMARY TABLE")
    print("=" * 80)
    print(f"{'ID':<4} | {'Type':<28} | {'Grounding Status':<28} | {'Latency':<8}")
    print("-" * 80)
    for r in results:
        status_flag = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<4} | {r['type']:<28} | {r['grounding_status']:<28} | {r['latency_seconds']}s")
    print("-" * 80)
    print(f"TOTAL SCORE: {total_passed}/5 ({total_passed * 20}% Strict Grounding Compliance)")
    print("=" * 80)

    # Export to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "project": "Project 7: AI Research Paper & Architecture Literature Assistant",
            "corpus": "transformer_rag_research_paper_50p.pdf",
            "total_passed": total_passed,
            "total_questions": len(ASSESSMENT_QUESTIONS),
            "score_percentage": f"{total_passed * 20}%",
            "results": results
        }, f, indent=2)

    print(f"\n[+] Results successfully exported to: {output_file}\n")
    return results


if __name__ == "__main__":
    run_all_assessments()

