"""
Web Interface for RAG Legal Assistant
"""

import sys
import io
import wave
import base64
import re
import numpy as np
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CSS = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }

/* Shrink the mic recorder to just the waveform + record button */
.mic-wrap .icon-buttons          { display: none !important; }
.mic-wrap .download-button       { display: none !important; }
.mic-wrap .upload-container      { display: none !important; }
.mic-wrap [class*="source"]      { display: none !important; }
.mic-wrap [class*="device"]      { display: none !important; }
.mic-wrap select                  { display: none !important; }
.mic-wrap .record-button         { margin: 0 !important; }
.mic-wrap                        { min-height: unset !important; padding: 4px !important; overflow: hidden !important; }
.mic-wrap > .component-wrapper   { gap: 4px !important; }

/* Orange send arrow button */
#send-btn {
    background: #f97316 !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    width: 84px !important;
    min-width: 84px !important;
    height: 34px !important;
    padding: 0 10px !important;
    line-height: 1 !important;
}
#send-btn:hover { background: #ea6c0a !important; }

/* Minimal TTS player */
audio { height: 36px; outline: none; }
.tts-wrap { padding: 4px 0; }
"""


# ---------------------------------------------------------------------------
# Speech helpers
# ---------------------------------------------------------------------------

def transcribe_audio(audio_tuple):
    """Convert mic audio (sample_rate, np.ndarray) → transcript string."""
    if audio_tuple is None:
        return ""
    try:
        import speech_recognition as sr
    except ImportError:
        return "[Install SpeechRecognition: pip install SpeechRecognition]"

    sample_rate, audio_data = audio_tuple
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    buf.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(buf) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return "[Speech recognition unavailable]"


def speak_text_html(text):
    """Convert text → HTML snippet with a minimal autoplay audio player."""
    if not text or not text.strip():
        return ""
    try:
        from gtts import gTTS
    except ImportError:
        return "<p style='color:red'>Install gTTS: pip install gTTS</p>"

    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<div class="tts-wrap">'
        f'<audio controls autoplay style="width:100%;height:36px">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        f'</audio></div>'
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def create_interface():
    from RAG.rag_system import RAGSystem

    rag_system = RAGSystem()

    def _format_response_for_chat(raw_response: str) -> str:
        """Render model output as Final Answer first, then IRAC reasoning."""
        text = (raw_response or "").strip()
        if not text:
            return ""

        labels = ["final answer", "issue", "rule", "application", "conclusion"]
        sections = {label: "" for label in labels}

        # Parse both inline and multiline labels, e.g.
        # "Issue: ... Rule: ... Application: ... Conclusion: ..."
        section_pattern = re.compile(
            r"(?is)(final answer|issue|rule|application|conclusion)\s*:\s*(.*?)"
            r"(?=(?:\s|\n)*(?:final answer|issue|rule|application|conclusion)\s*:|$)"
        )
        for match in section_pattern.finditer(text):
            label = match.group(1).lower()
            content = re.sub(r"\s+", " ", match.group(2)).strip()
            # Remove accidental nested label prefixes like "Issue: Issue: ..."
            content = re.sub(
                r"(?is)^(final answer|issue|rule|application|conclusion)\s*:\s*",
                "",
                content,
            ).strip()
            if content:
                sections[label] = content

        final_answer = sections["final answer"]
        issue = sections["issue"]
        rule = sections["rule"]
        application = sections["application"]
        conclusion = sections["conclusion"]

        # Fallback when the model misses labels.
        if not any([final_answer, issue, rule, application, conclusion]):
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            final_answer = paragraphs[0] if paragraphs else text
            rest = paragraphs[1:] if len(paragraphs) > 1 else []
            issue = rest[0] if len(rest) > 0 else ""
            rule = rest[1] if len(rest) > 1 else ""
            application = rest[2] if len(rest) > 2 else ""
            conclusion = rest[3] if len(rest) > 3 else ""

        if not final_answer:
            final_answer = conclusion or "No direct answer found from retrieved cases."

        out = [
            "### Final Answer",
            f"> {final_answer}",
            "\n### IRAC Reasoning",
            "#### Issue",
            issue or "Not clearly stated.",
            "\n#### Rule",
            rule or "Not clearly stated.",
            "\n#### Application",
            application or "Not clearly stated.",
            "\n#### Conclusion",
            conclusion or final_answer,
        ]
        return "\n\n".join(out)

    def _build_metadata(result: dict) -> str:
        if not result.get("retrieved_cases"):
            return "No cases retrieved."
        blocks = []
        sorted_cases = sorted(result["retrieved_cases"], key=lambda c: c.get("similarity", 0), reverse=True)
        for i, case in enumerate(sorted_cases, 1):
            citation = case.get("citation", case.get("case_name", "Unknown case"))
            sim = f"{case['similarity']:.3f}"
            doc_excerpt = case.get("text", "")[:450].strip()
            if len(case.get("text", "")) > 450:
                doc_excerpt += "..."
            blocks.append(
                f"Case {i}\n"
                f"{citation}\n"
                f"Similarity: {sim}\n\n"
                f"{doc_excerpt}"
            )
        return "\n\n─────────────────────────────────────\n\n".join(blocks)

    def chat(query, history, llm_history, top_k, temperature):
        if not query.strip():
            return "", history, llm_history, "", ""

        result = rag_system.generate_response(
            query,
            mode="rag",
            top_k=top_k,
            temperature=temperature,
            chat_history=llm_history if llm_history else None,
        )

        response = result["response"]
        display_response = _format_response_for_chat(response)
        new_turns = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": display_response},
        ]
        history = history + new_turns
        llm_history = llm_history + new_turns
        return "", history, llm_history, _build_metadata(result), response

    # ------------------------------------------------------------------ UI --
    with gr.Blocks(title="RAG Legal Assistant") as interface:
        gr.Markdown("# RAG Legal Assistant")

        llm_history_state = gr.State([])
        last_response_state = gr.State("")

        with gr.Tab("Chat"):
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=5, value=3, step=1,
                        label="Cases to Retrieve"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.3, step=0.1,
                        label="Creativity (Temperature)"
                    )

            # NOTE: Gradio 6 removed `type` and `render_markdown` from gr.Chatbot — do not add them back
            chatbot = gr.Chatbot(height=440, show_label=False)

            # Input row: mic | text | send
            with gr.Row(equal_height=True):
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    show_label=False,
                    elem_classes=["mic-wrap"],
                    scale=1,
                    min_width=118,
                    container=False,
                )
                msg_input = gr.Textbox(
                    placeholder="Ask a legal question...",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                send_btn = gr.Button("Send", size="sm", scale=1, min_width=84, elem_id="send-btn")

            # Controls row
            with gr.Row():
                clear_btn = gr.Button(
                    "Clear conversation", size="sm", variant="secondary", scale=3
                )
                tts_btn = gr.Button("🔊 Hear response", size="sm", scale=2)

            # Minimal TTS player — hidden until used
            tts_html = gr.HTML(value="", visible=True)

            with gr.Accordion("Retrieved Cases (last turn)", open=False):
                metadata_output = gr.Markdown(
                    value="",
                    show_label=False,
                    container=False
                )

            # -------- event wiring --------
            chat_inputs  = [msg_input, chatbot, llm_history_state, top_k_slider, temp_slider]
            chat_outputs = [msg_input, chatbot, llm_history_state, metadata_output, last_response_state]

            # Mic: transcribe on stop → populate text box for review before sending
            mic_input.stop_recording(
                fn=transcribe_audio,
                inputs=[mic_input],
                outputs=[msg_input],
            )

            send_btn.click(fn=chat, inputs=chat_inputs, outputs=chat_outputs)
            msg_input.submit(fn=chat, inputs=chat_inputs, outputs=chat_outputs)

            tts_btn.click(
                fn=speak_text_html,
                inputs=[last_response_state],
                outputs=[tts_html],
            )

            clear_btn.click(
                fn=lambda: ("", [], [], "", "", ""),
                outputs=[msg_input, chatbot, llm_history_state,
                         metadata_output, last_response_state, tts_html],
            )

        with gr.Tab("About"):
            gr.Markdown(
                """
                ## RAG Legal Assistant

                A conversational legal research tool that answers questions strictly from real U.S. court opinions — no hallucinated citations, no general knowledge fill-in.

                ---

                ### How a query works (end to end)

                1. **Embedding** — your question is encoded into a 768-dim vector using [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), a state-of-the-art bi-encoder fine-tuned for semantic search.
                2. **Retrieval** — ChromaDB performs approximate nearest-neighbour search over 1,349 chunks of U.S. court opinions (Caselaw Access Project), returning up to 20 candidates ranked by cosine similarity.
                3. **Reranking** — a cross-encoder ([ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)) re-scores every candidate pair (query, chunk) for precise relevance, and the top *k* are kept.
                4. **Generation** — the top chunks are injected into a strict system prompt sent to **Llama 3.3 70B** (via Groq). The prompt enforces the **IRAC framework** (Issue → Rule → Application → Conclusion): the model must identify the legal issue, state the rule drawn from the retrieved cases, apply it to the question, and deliver a conclusion — quoting verbatim from the retrieved text at each step. If the documents don't contain a clear answer, the model is required to say so rather than speculate.
                5. **Conversation memory** — prior turns are included in each API call so you can ask follow-up questions and the model has full context.

                ---

                ### Retrieved Cases panel
                After each answer, expand **Retrieved Cases (last turn)** to see which court opinions were used, their similarity scores (sorted highest → lowest), and the chunk of text the model actually read. Similarity < 0.60 triggers a relevance warning.

                ---

                ### Speech features
                - **Mic** — click the microphone, speak, stop recording. Google Speech Recognition transcribes your question into the text box; review it and hit Send when ready.
                - **🔊 Hear response** — generates a natural-sounding audio reading of the last answer via Google TTS and plays it inline with a minimal play/pause bar.

                ---

                ### Settings
                | Setting | Effect |
                |---|---|
                | Cases to Retrieve (1–5) | How many reranked court opinion chunks are fed to the LLM per query. More = more context but slower. |
                | Creativity (Temperature) | 0 = highly deterministic answers; 1 = more varied phrasing. Keep low (0.2–0.4) for legal research. |

                ---

                ### Dataset & stack
                - **Corpus:** Caselaw Access Project — 2M+ U.S. court opinions; 1,349 chunks indexed here.
                - **Embeddings:** BAAI/bge-base-en-v1.5 (768-dim, BGE instruction prefix at query time).
                - **Vector store:** ChromaDB (persistent, cosine distance).
                - **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2.
                - **LLM:** Llama 3.3 70B Versatile via Groq API.
                """
            )

    return interface


if __name__ == "__main__":
    print("Starting RAG Legal Assistant...")
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CSS,
    )
