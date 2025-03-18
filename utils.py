from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

class EdTechAssistant:
    def __init__(self, 
                 custom_rag_prompt: PromptTemplate = None, 
                 model: ChatGoogleGenerativeAI = None, 
                 embeddings: GoogleGenerativeAIEmbeddings = None, 
                 vector_store: Chroma = None, 
                 text_splitter: RecursiveCharacterTextSplitter = None):
        
        self.template = """
        You are an intelligent assistant designed to provide accurate and helpful answers based on the context provided. Follow these guidelines:
        1. Use only the information from the context to answer the question.
        2. If the context does not contain enough information to answer the question, say "I don't know" and do not make up an answer.
        3. Be concise and specific in your response.
        4. Always end your answer with "Thanks for asking!" to maintain a friendly tone.

        Context: {context}

        Question: {question}

        Answer:
        """
        self.custom_rag_prompt = custom_rag_prompt or PromptTemplate.from_template(self.template)
        self.model = model or ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.vector_store = vector_store or Chroma(embedding_function=self.embeddings)
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

    class State:
        def __init__(self, question: str):
            self.question = question
            self.context: List[Document] = []
            self.answer: str = ""

    def retrieve(self, state: 'EdTechAssistant.State'):
        state.context = self.vector_store.similarity_search(state.question)
        return state

    def generate(self, state: 'EdTechAssistant.State'):
        docs_content = "\n\n".join(doc.page_content for doc in state.context)
        messages = self.custom_rag_prompt.invoke({"question": state.question, "context": docs_content})
        response = self.model.invoke(messages)
        state.answer = response
        return state

    def workflow(self, state_input: Dict[str, Any]) -> Dict[str, Any]:
        state = self.State(state_input["question"])
        state = self.retrieve(state)
        state = self.generate(state)
        return {"context": state.context, "answer": state.answer}

    def process_pdfs(self, files):
        status_messages = []
        for file in files:
            loader = PyPDFLoader(file.name)
            pages = loader.load()
            all_splits = self.text_splitter.split_documents(pages)
            self.vector_store.add_documents(documents=all_splits)
            status_messages.append(f"Processed {file.name} and added to database!")
        return "\n".join(status_messages)

    def ask_question(self, question):
        result = self.workflow({"question": question})
        context = "\n\n".join(doc.page_content for doc in result["context"])
        answer = result["answer"].content
        return context, answer

    def main(self):
        with gr.Blocks() as demo:
            gr.Markdown("# PDF Query Interface")
            with gr.Tab("Upload PDFs"):
                pdf_input = gr.File(label="Upload PDFs", file_count="multiple")
                upload_status = gr.Textbox(label="Status")
                upload_button = gr.Button("Upload and Process")
            with gr.Tab("Ask Question"):
                question_input = gr.Textbox(label="Your Question", elem_id="question_input")
                context_output = gr.Textbox(label="Context", lines=10)
                answer_output = gr.Textbox(label="Answer", lines=5, elem_id="answer_output")
                ask_button = gr.Button("Ask")

                # Add STT and TTS buttons
                stt_button = gr.Button("Speak Your Question")
                tts_button = gr.Button("Hear the Answer")

                # Add custom HTML and JavaScript
                gr.HTML("""
                <script>
                function startSpeechRecognition() {
                    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                    recognition.lang = 'en-US';
                    recognition.interimResults = false;
                    recognition.maxAlternatives = 1;

                    recognition.start();

                    recognition.onresult = (event) => {
                        const transcript = event.results[0][0].transcript;
                        document.getElementById("question_input").value = transcript;
                    };

                    recognition.onerror = (event) => {
                        console.error("Speech recognition error:", event.error);
                    };
                }

                function speakText() {
                    const text = document.getElementById("answer_output").value;
                    const utterance = new SpeechSynthesisUtterance(text);
                    window.speechSynthesis.speak(utterance);
                }
                </script>
                """)

                # Attach JavaScript to buttons using Gradio's `js` parameter
                stt_button.click(fn=None, js="startSpeechRecognition()")
                tts_button.click(fn=None, js="speakText()")

            upload_button.click(self.process_pdfs, inputs=pdf_input, outputs=upload_status)
            ask_button.click(self.ask_question, inputs=question_input, outputs=[context_output, answer_output])

        demo.launch()

if __name__ == "__main__":
    assistant = EdTechAssistant()
    assistant.main()