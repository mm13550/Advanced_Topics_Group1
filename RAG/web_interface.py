"""
Web Interface for RAG Legal Assistant
Interactive Gradio interface with side-by-side comparison of baseline vs RAG.
"""

import gradio as gr
from typing import Tuple
import json
from datetime import datetime


def create_interface():
    """Create Gradio web interface for RAG legal assistant."""
    
    from rag_system import RAGSystem
    from baseline_llm import BaseLLM
    
    rag_system = RAGSystem()
    baseline_llm = BaseLLM()
    
    def process_query(
        query: str,
        mode: str,
        top_k: int,
        temperature: float
    ) -> Tuple[str, str]:
        """
        Process user query and return response with metadata.
        
        Args:
            query: User's legal question
            mode: Response mode (RAG, IRAC-only, or Baseline)
            top_k: Number of cases to retrieve
            temperature: Sampling temperature
            
        Returns:
            Tuple of (response_text, metadata_text)
        """
        if not query.strip():
            return "Please enter a question.", ""
        
        mode_map = {
            "RAG (Retrieval + IRAC)": "rag",
            "IRAC Only (No Retrieval)": "irac_only",
            "Baseline (No IRAC, No Retrieval)": "baseline"
        }
        
        selected_mode = mode_map[mode]
        
        result = rag_system.generate_response(
            query,
            mode=selected_mode,
            top_k=top_k,
            temperature=temperature
        )
        
        response = result["response"]
        
        metadata_parts = [
            f"Mode: {selected_mode.upper()}",
            f"Model: {result.get('model', 'N/A')}",
            f"Timestamp: {result['timestamp']}"
        ]
        
        if selected_mode == "rag":
            metadata_parts.append(f"Cases Retrieved: {result['num_cases_retrieved']}")
            
            if result['retrieved_cases']:
                metadata_parts.append("\nRetrieved Cases:")
                for i, case in enumerate(result['retrieved_cases'], 1):
                    metadata_parts.append(
                        f"\n{i}. Similarity: {case['similarity']:.4f}\n"
                        f"   Preview: {case['preview'][:150]}..."
                    )
        
        metadata = "\n".join(metadata_parts)
        
        return response, metadata
    
    def compare_modes(
        query: str,
        top_k: int,
        temperature: float
    ) -> Tuple[str, str, str]:
        """
        Compare responses across all three modes.
        
        Returns:
            Tuple of (baseline_response, irac_response, rag_response)
        """
        if not query.strip():
            return "Please enter a question.", "", ""
        
        baseline_result = baseline_llm.generate_response(
            query,
            temperature=temperature
        )
        
        irac_result = rag_system.generate_response(
            query,
            mode="irac_only",
            temperature=temperature
        )
        
        rag_result = rag_system.generate_response(
            query,
            mode="rag",
            top_k=top_k,
            temperature=temperature
        )
        
        baseline_text = f"**BASELINE**\n\n{baseline_result}"
        
        irac_text = f"**IRAC (No Retrieval)**\n\n{irac_result['response']}"
        
        rag_info = f"Retrieved {rag_result['num_cases_retrieved']} cases\n\n"
        rag_text = f"**RAG (Retrieval + IRAC)**\n\n{rag_info}{rag_result['response']}"
        
        return baseline_text, irac_text, rag_text
    
    css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .response-box {
        border-left: 3px solid #2563eb;
        padding-left: 1rem;
    }
    """
    
    with gr.Blocks(css=css, title="RAG Legal Assistant") as interface:
        gr.Markdown(
            """
            # RAG Legal Assistant
            Ask legal questions and get responses using retrieval-augmented generation.
            """
        )
        
        with gr.Tab("Single Query"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Legal Question",
                        placeholder="e.g., What constitutes probable cause for a search?",
                        lines=3
                    )
                    
                    with gr.Row():
                        mode_select = gr.Radio(
                            choices=[
                                "RAG (Retrieval + IRAC)",
                                "IRAC Only (No Retrieval)",
                                "Baseline (No IRAC, No Retrieval)"
                            ],
                            value="RAG (Retrieval + IRAC)",
                            label="Response Mode"
                        )
                    
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Number of Cases to Retrieve (RAG mode only)"
                        )
                        
                        temp_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Temperature"
                        )
                    
                    submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column(scale=3):
                    response_output = gr.Textbox(
                        label="Response",
                        lines=15,
                        elem_classes=["response-box"]
                    )
                    
                    metadata_output = gr.Textbox(
                        label="Metadata & Retrieved Cases",
                        lines=8
                    )
            
            submit_btn.click(
                fn=process_query,
                inputs=[query_input, mode_select, top_k_slider, temp_slider],
                outputs=[response_output, metadata_output]
            )
        
        with gr.Tab("Compare Modes"):
            gr.Markdown(
                """
                Compare responses across all three modes side-by-side.
                """
            )
            
            compare_query = gr.Textbox(
                label="Legal Question",
                placeholder="e.g., What are the elements of negligence?",
                lines=3
            )
            
            with gr.Row():
                compare_top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Cases to Retrieve (RAG)"
                )
                
                compare_temp = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Temperature"
                )
            
            compare_btn = gr.Button("Compare All Modes", variant="primary")
            
            with gr.Row():
                baseline_output = gr.Textbox(
                    label="Baseline",
                    lines=12
                )
                
                irac_output = gr.Textbox(
                    label="IRAC Only",
                    lines=12
                )
                
                rag_output = gr.Textbox(
                    label="RAG",
                    lines=12
                )
            
            compare_btn.click(
                fn=compare_modes,
                inputs=[compare_query, compare_top_k, compare_temp],
                outputs=[baseline_output, irac_output, rag_output]
            )
        
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## RAG Legal Assistant
                
                This system uses Retrieval-Augmented Generation to answer legal questions.
                
                ### Modes:
                
                **Baseline**: Standard LLM without retrieval or structured prompting.
                
                **IRAC Only**: Uses IRAC framework (Issue, Rule, Application, Conclusion) 
                without retrieval from case database.
                
                **RAG**: Retrieves relevant cases from database and generates IRAC-structured 
                responses grounded in legal precedent.
                
                ### Dataset:
                
                Caselaw Access Project - 2M+ U.S. court opinions with 768-D embeddings.
                
                ### Model:
                
                Llama 3.3 70B via Groq API
                """
            )
    
    return interface


if __name__ == "__main__":
    print("Starting RAG Legal Assistant web interface...")
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
