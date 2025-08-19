"""
RAG Web Application
===================

A Streamlit web interface for Document Question Answering using RAG (Retrieval-Augmented Generation).
Users can configure PDF directory, embedding model, LLM model, and ask questions about their documents.
"""

import streamlit as st
from pathlib import Path
import traceback
import time
from typing import Optional, List, Dict, Any
import pandas as pd
import json

# Import DLLMForge modules
from dllmforge.rag_preprocess_documents import PDFLoader, TextChunker
from dllmforge.rag_embedding_open_source import LangchainHFEmbeddingModel
from dllmforge.LLMs.Deltares_LLMs import DeltaresOllamaLLM
from dllmforge.langchain_api import LangchainAPI
from dllmforge.rag_evaluation import RAGEvaluator
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss


class RAGApp:
    """Main RAG Application class"""

    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.embedding_model = None
        self.documents_loaded = False

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'rag_app' not in st.session_state:
            st.session_state.rag_app = RAGApp()
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'vector_store_ready' not in st.session_state:
            st.session_state.vector_store_ready = False
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "main"
        if 'evaluator_results' not in st.session_state:
            st.session_state.evaluator_results = []

    def load_embedding_model(self, model_name: str):
        """Load the selected embedding model"""
        try:
            with st.spinner(f"Loading embedding model: {model_name}..."):
                self.embedding_model = LangchainHFEmbeddingModel(model_name=model_name)
            st.success(f"✅ Embedding model loaded: {model_name}")
            return True
        except Exception as e:
            st.error(f"❌ Error loading embedding model: {str(e)}")
            return False

    def load_llm_model(self, provider: str, **kwargs):
        """Load the selected LLM model"""
        try:
            with st.spinner(f"Loading LLM: {provider}..."):
                if provider == "Deltares Ollama":
                    self.llm = DeltaresOllamaLLM(base_url=kwargs.get('base_url'),
                                                 model_name=kwargs.get('model_name'),
                                                 temperature=kwargs.get('temperature', 0.7))
                elif provider == "OpenAI":
                    self.llm = LangchainAPI(model_provider="openai",
                                            model_name=kwargs.get('model_name', 'gpt-4'),
                                            temperature=kwargs.get('temperature', 0.7)).llm
                elif provider == "Azure OpenAI":
                    self.llm = LangchainAPI(model_provider="azure-openai",
                                            deployment_name=kwargs.get('deployment_name'),
                                            api_base=kwargs.get('api_base'),
                                            api_version=kwargs.get('api_version'),
                                            temperature=kwargs.get('temperature', 0.7)).llm
                elif provider == "Mistral":
                    self.llm = LangchainAPI(model_provider="mistral",
                                            model_name=kwargs.get('model_name', 'mistral-large-latest'),
                                            temperature=kwargs.get('temperature', 0.7)).llm
                else:
                    raise ValueError(f"Unsupported LLM provider: {provider}")

            st.success(f"✅ LLM loaded: {provider}")
            return True
        except Exception as e:
            st.error(f"❌ Error loading LLM: {str(e)}")
            return False

    def process_documents(self, pdf_directory: str, chunk_size: int = 1000, overlap_size: int = 200):
        """Process PDF documents from the specified directory"""
        try:
            data_dir = Path(pdf_directory)
            if not data_dir.exists():
                st.error(f"Directory does not exist: {pdf_directory}")
                return False

            # Find all PDF files
            pdfs = list(data_dir.glob("*.pdf"))
            if not pdfs:
                st.error(f"No PDF files found in: {pdf_directory}")
                return False

            st.info(f"Found {len(pdfs)} PDF files")

            # Initialize components
            loader = PDFLoader()
            chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)

            global_embeddings = []
            metadatas = []

            # Process each PDF
            progress_bar = st.progress(0)
            for i, pdf_path in enumerate(pdfs):
                st.write(f"Processing: {pdf_path.name}")

                # Load PDF
                pages, file_name, metadata = loader.load(pdf_path)

                # Create chunks
                chunks = chunker.chunk_text(pages, file_name, metadata)

                # Embed chunks
                chunk_embeddings = self.embedding_model.embed(chunks)
                global_embeddings.extend(chunk_embeddings)
                metadatas.extend([chunk["metadata"] for chunk in chunks])

                progress_bar.progress((i + 1) / len(pdfs))
                st.write(f"✅ Processed {len(chunk_embeddings)} chunks from {file_name}")

            # Create vector store
            st.write("Creating vector database...")
            index = faiss.IndexFlatL2(len(global_embeddings[0]["text_vector"]))

            self.vector_store = FAISS(
                embedding_function=self.embedding_model.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            # Add embeddings to vector store
            for chunk, meta in zip(global_embeddings, metadatas):
                self.vector_store.add_texts(texts=[chunk["chunk"]],
                                            metadatas=[meta],
                                            ids=[chunk["chunk_id"]],
                                            embeddings=[chunk["text_vector"]])

            st.success(f"✅ Successfully processed {len(global_embeddings)} document chunks")
            self.documents_loaded = True
            st.session_state.vector_store_ready = True
            return True

        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            st.error(traceback.format_exc())
            return False

    def ask_question(self, question: str, k: int = 5, score_threshold: float = 0.5):
        """Ask a question using the RAG system"""
        try:
            if not self.vector_store or not self.llm:
                st.error("Please load documents and configure LLM first")
                return None

            # Create retriever
            retriever = self.vector_store.as_retriever(search_type="similarity_score_threshold",
                                                       search_kwargs={
                                                           "score_threshold": score_threshold,
                                                           "k": k
                                                       })

            # Get answer
            with st.spinner("Generating answer..."):
                # Check if this is a Deltares Ollama LLM (has ask_with_retriever method)
                if hasattr(self.llm, 'ask_with_retriever'):
                    chat_result = self.llm.ask_with_retriever(question, retriever)
                    answer = chat_result.generations[0].message.content
                else:
                    # For other LLMs, manually create the prompt with context
                    contexts = retriever.invoke(question)
                    context_text = "\n\n".join([doc.page_content for doc in contexts])

                    prompt = f"""Based on the following context, please answer the question.

Context:
{context_text}

Question: {question}

Please provide a clear and concise answer based only on the information provided in the context."""

                    # Use the LLM directly
                    response = self.llm.invoke([{"role": "user", "content": prompt}])
                    answer = response.content if hasattr(response, 'content') else str(response)
                    contexts = retriever.invoke(question)

                # Clean up answer if needed
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()

                # Get relevant contexts if not already retrieved
                if 'contexts' not in locals():
                    contexts = retriever.invoke(question)

            return {
                'answer': answer,
                'contexts': contexts,
                'question': question,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            st.error(traceback.format_exc())
            return None


def evaluator_page():
    """RAG Evaluation Page - Hidden page for evaluation purposes"""
    st.title("🔬 RAG System Evaluator")
    st.markdown("Comprehensive evaluation of RAG system performance using RAGAS metrics")

    # Check if system is ready
    if not st.session_state.vector_store_ready:
        st.error("❌ System not initialized. Please configure and load your system first.")
        if st.button("← Go Back to Main Page"):
            st.session_state.current_page = "main"
            st.rerun()
        return

    # Evaluation configuration
    st.header("⚙️ Evaluation Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Evaluator LLM")
        eval_llm_provider = st.selectbox("Choose LLM for Evaluation", ["OpenAI", "Anthropic", "Deltares Ollama"],
                                         help="Select the LLM to use for evaluation metrics")

        # Provider-specific configurations for evaluator
        if eval_llm_provider == "OpenAI":
            st.info("Using OpenAI API key from environment")
        elif eval_llm_provider == "Anthropic":
            st.info("Using Anthropic API key from environment")
        elif eval_llm_provider == "Deltares Ollama":
            eval_base_url = st.text_input("Evaluator Ollama Base URL",
                                          value="https://chat-api.directory.intra",
                                          key="eval_base_url")
            eval_model_name = st.text_input("Evaluator Model Name", value="qwen3:latest", key="eval_model_name")

    with col2:
        st.subheader("🎯 Test Configuration")

        # Option to use predefined questions or custom questions
        test_mode = st.radio("Test Mode", ["Predefined Questions", "Custom Questions"],
                             help="Choose how to generate test questions")

        if test_mode == "Custom Questions":
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)

    # Test Questions Section
    st.header("❓ Test Questions")

    if test_mode == "Predefined Questions":
        st.info("Using built-in test questions for evaluation")
        predefined_questions = [{
            "question": "What is the main topic of these documents?",
            "ground_truth": "The documents cover various topics as specified in the document collection."
        }, {
            "question": "What are the key findings?",
            "ground_truth": "Key findings are detailed in the conclusions section of the documents."
        }, {
            "question": "What methodology was used?",
            "ground_truth": "The methodology section describes the approaches and methods used."
        }, {
            "question": "What are the main conclusions?",
            "ground_truth": "Main conclusions are summarized at the end of each document."
        }, {
            "question": "What recommendations are provided?",
            "ground_truth": "Recommendations are provided based on the analysis and findings."
        }]

        # Display questions
        for i, q in enumerate(predefined_questions, 1):
            with st.expander(f"Question {i}: {q['question']}"):
                st.write(f"**Ground Truth:** {q['ground_truth']}")

        test_questions = predefined_questions

    elif test_mode == "Custom Questions":
        test_questions = []
        st.write("Enter your custom test questions:")

        for i in range(num_questions):
            with st.expander(f"Question {i+1}"):
                question = st.text_input(f"Question {i+1}", key=f"q_{i}")
                ground_truth = st.text_area(f"Expected Answer/Ground Truth {i+1}",
                                            key=f"gt_{i}",
                                            help="Provide the ideal answer for evaluation")

                if question and ground_truth:
                    test_questions.append({"question": question, "ground_truth": ground_truth})

    # Run Evaluation
    st.header("🚀 Run Evaluation")

    if st.button("▶️ Start Evaluation", type="primary", disabled=len(test_questions) == 0):
        if len(test_questions) == 0:
            st.error("Please provide at least one test question.")
            return

        try:
            # Initialize evaluator
            with st.spinner("Initializing evaluator..."):
                if eval_llm_provider.lower() == "deltares ollama":
                    evaluator = RAGEvaluator(llm_provider="deltares",
                                             deltares_llm=DeltaresOllamaLLM(base_url=eval_base_url,
                                                                            model_name=eval_model_name))
                else:
                    evaluator = RAGEvaluator(llm_provider=eval_llm_provider.lower())

            st.success("✅ Evaluator initialized")

            # Get current RAG app
            current_app = st.session_state.rag_app

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            # Process each question
            for i, q_data in enumerate(test_questions):
                status_text.text(f"Evaluating question {i+1}/{len(test_questions)}: {q_data['question'][:50]}...")

                # Get RAG response
                rag_result = current_app.ask_question(q_data["question"])

                if rag_result:
                    # Extract contexts as strings
                    retrieved_contexts = [doc.page_content for doc in rag_result['contexts']]

                    # Run evaluation
                    evaluation = evaluator.evaluate_rag_pipeline(question=q_data["question"],
                                                                 generated_answer=rag_result['answer'],
                                                                 retrieved_contexts=retrieved_contexts,
                                                                 ground_truth_answer=q_data.get("ground_truth", ""))

                    result = {
                        'question': q_data["question"],
                        'generated_answer': rag_result['answer'],
                        'ground_truth': q_data.get("ground_truth", ""),
                        'retrieved_contexts': retrieved_contexts,
                        'evaluation': evaluation,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    results.append(result)

                progress_bar.progress((i + 1) / len(test_questions))

            # Store results
            st.session_state.evaluator_results = results

            status_text.text("Evaluation completed!")
            st.success(f"✅ Evaluation completed for {len(results)} questions")

        except Exception as e:
            st.error(f"❌ Evaluation failed: {str(e)}")
            st.error(traceback.format_exc())

    # Display Results
    if st.session_state.evaluator_results:
        st.header("📊 Evaluation Results")

        results = st.session_state.evaluator_results

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        avg_ragas = sum(r['evaluation'].ragas_score for r in results) / len(results)
        avg_relevancy = sum(r['evaluation'].answer_relevancy.score for r in results) / len(results)
        avg_faithfulness = sum(r['evaluation'].faithfulness.score for r in results) / len(results)
        avg_context_relevancy = sum(r['evaluation'].context_relevancy.score for r in results) / len(results)

        with col1:
            st.metric("Average RAGAS Score", f"{avg_ragas:.3f}")
        with col2:
            st.metric("Answer Relevancy", f"{avg_relevancy:.3f}")
        with col3:
            st.metric("Faithfulness", f"{avg_faithfulness:.3f}")
        with col4:
            st.metric("Context Relevancy", f"{avg_context_relevancy:.3f}")

        # Detailed results
        st.subheader("🔍 Detailed Results")

        for i, result in enumerate(results, 1):
            with st.expander(f"Question {i}: {result['question'][:100]}..."):
                eval_res = result['evaluation']

                # Question and answers
                st.markdown("**📝 Question:**")
                st.write(result['question'])

                st.markdown("**🤖 Generated Answer:**")
                st.write(result['generated_answer'])

                if result['ground_truth']:
                    st.markdown("**✅ Ground Truth:**")
                    st.write(result['ground_truth'])

                # Metrics
                st.markdown("**📊 Evaluation Metrics:**")
                metric_cols = st.columns(4)

                with metric_cols[0]:
                    st.metric("RAGAS Score", f"{eval_res.ragas_score:.3f}")
                with metric_cols[1]:
                    st.metric("Answer Relevancy", f"{eval_res.answer_relevancy.score:.3f}")
                with metric_cols[2]:
                    st.metric("Faithfulness", f"{eval_res.faithfulness.score:.3f}")
                with metric_cols[3]:
                    st.metric("Context Relevancy", f"{eval_res.context_relevancy.score:.3f}")

                # Explanations
                with st.expander("📋 Detailed Explanations"):
                    st.markdown("**Answer Relevancy Explanation:**")
                    st.write(eval_res.answer_relevancy.explanation)

                    st.markdown("**Faithfulness Explanation:**")
                    st.write(eval_res.faithfulness.explanation)

                    st.markdown("**Context Relevancy Explanation:**")
                    st.write(eval_res.context_relevancy.explanation)

        # Export functionality
        st.subheader("💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download as JSON"):
                # Prepare data for JSON export
                export_data = []
                for result in results:
                    export_item = {
                        'question': result['question'],
                        'generated_answer': result['generated_answer'],
                        'ground_truth': result['ground_truth'],
                        'timestamp': result['timestamp'],
                        'ragas_score': result['evaluation'].ragas_score,
                        'answer_relevancy_score': result['evaluation'].answer_relevancy.score,
                        'faithfulness_score': result['evaluation'].faithfulness.score,
                        'context_relevancy_score': result['evaluation'].context_relevancy.score,
                        'answer_relevancy_explanation': result['evaluation'].answer_relevancy.explanation,
                        'faithfulness_explanation': result['evaluation'].faithfulness.explanation,
                        'context_relevancy_explanation': result['evaluation'].context_relevancy.explanation
                    }
                    export_data.append(export_item)

                json_str = json.dumps(export_data, indent=2)
                st.download_button(label="Download JSON",
                                   data=json_str,
                                   file_name=f"rag_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json",
                                   mime="application/json")

        with col2:
            if st.button("📊 Download as CSV"):
                # Create DataFrame for CSV export
                csv_data = []
                for result in results:
                    csv_data.append({
                        'Question': result['question'],
                        'Generated Answer': result['generated_answer'],
                        'Ground Truth': result['ground_truth'],
                        'RAGAS Score': result['evaluation'].ragas_score,
                        'Answer Relevancy': result['evaluation'].answer_relevancy.score,
                        'Faithfulness': result['evaluation'].faithfulness.score,
                        'Context Relevancy': result['evaluation'].context_relevancy.score,
                        'Timestamp': result['timestamp']
                    })

                df = pd.DataFrame(csv_data)
                csv_str = df.to_csv(index=False)
                st.download_button(label="Download CSV",
                                   data=csv_str,
                                   file_name=f"rag_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")

        # Clear results
        if st.button("🗑️ Clear Results"):
            st.session_state.evaluator_results = []
            st.rerun()

    # Navigation
    st.markdown("---")
    if st.button("← Back to Main Page"):
        st.session_state.current_page = "main"
        st.rerun()


def main():
    """Main Streamlit application"""

    # Page configuration
    st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

    # Initialize app
    app = RAGApp()
    app.initialize_session_state()

    # Check for hidden evaluator page access
    query_params = st.query_params
    if "evaluator" in query_params:
        st.session_state.current_page = "evaluator"

    # Page routing
    if st.session_state.current_page == "evaluator":
        evaluator_page()
        return

    # Main page content (existing functionality)
    # Main title
    st.title("📚 RAG Document Question & Answer System")
    st.markdown("Upload PDFs, configure your models, and ask questions about your documents!")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Add hidden access to evaluator page
        st.markdown("---")
        with st.expander("🔬 Advanced Tools"):
            st.markdown("**Developer Mode:**")
            if st.button("🧪 Open Evaluator", help="Access the RAG evaluation suite"):
                st.session_state.current_page = "evaluator"
                st.rerun()
            st.markdown("*Or add `?evaluator=true` to the URL*")
        st.markdown("---")

        # PDF Directory Configuration
        st.subheader("📁 Document Directory")
        pdf_directory = st.text_input("PDF Directory Path",
                                      placeholder="Enter path to folder containing PDF files",
                                      help="Specify the directory containing your PDF documents")

        # Embedding Model Configuration
        st.subheader("🧠 Embedding Model")
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5", "thenlper/gte-small", "thenlper/gte-base"
        ]

        selected_embedding = st.selectbox("Choose Embedding Model",
                                          embedding_models,
                                          help="Select the embedding model for document vectorization")

        # LLM Configuration
        st.subheader("🤖 Language Model")
        llm_provider = st.selectbox("Choose LLM Provider", ["Deltares Ollama", "OpenAI", "Azure OpenAI", "Mistral"],
                                    help="Select your preferred language model provider")

        # Provider-specific configurations
        if llm_provider == "Deltares Ollama":
            base_url = st.text_input("Ollama Base URL",
                                     value="https://chat-api.directory.intra",
                                     help="Base URL for Ollama API")
            model_name = st.text_input("Model Name", value="qwen3:latest", help="Ollama model name")
        elif llm_provider == "OpenAI":
            model_name = st.selectbox("OpenAI Model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                                      help="Select OpenAI model")
            st.info("Make sure to set your OPENAI_API_KEY environment variable")
        elif llm_provider == "Azure OpenAI":
            deployment_name = st.text_input("Deployment Name",
                                            placeholder="your-gpt-4-deployment",
                                            help="Azure OpenAI deployment name")
            api_base = st.text_input("API Base URL",
                                     placeholder="https://your-resource.openai.azure.com/",
                                     help="Azure OpenAI endpoint URL")
            api_version = st.text_input("API Version", value="2023-12-01-preview", help="Azure OpenAI API version")
            st.info("Set AZURE_OPENAI_API_KEY environment variable")
        elif llm_provider == "Mistral":
            model_name = st.selectbox("Mistral Model",
                                      ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
                                      help="Select Mistral model")
            st.info("Make sure to set your MISTRAL_API_KEY environment variable")

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            temperature = st.slider("Temperature",
                                    0.0,
                                    1.0,
                                    0.7,
                                    0.1,
                                    help="""**Temperature**: Controls randomness in the LLM's responses.
                
**Values:**
- **0.0**: Deterministic, always picks the most likely response (best for factual Q&A)
- **0.1-0.3**: Low randomness, very consistent and focused responses
- **0.4-0.6**: Moderate randomness, balanced between consistency and creativity
- **0.7-0.8**: Higher creativity, more varied responses (good for general conversation)
- **0.9-1.0**: High randomness, very creative but potentially less coherent

**Recommendation**: Use 0.1-0.3 for factual document analysis, 0.7 for general Q&A""")

            chunk_size = st.slider("Chunk Size",
                                   500,
                                   2000,
                                   1000,
                                   100,
                                   help="""**Chunk Size**: Number of characters per document chunk for processing.
                
**Values:**
- **500-800**: Small chunks, good for precise answers to specific questions
- **900-1200**: Medium chunks, balanced approach for most use cases
- **1300-1600**: Large chunks, better for comprehensive context and summaries
- **1700-2000**: Very large chunks, preserves more context but may dilute relevance

**Impact**: 
- Smaller chunks = More precise, focused answers but may miss broader context
- Larger chunks = More comprehensive context but potentially less focused answers

**Recommendation**: 1000 for general use, 800 for technical documents, 1500 for narrative texts""")

            overlap_size = st.slider("Overlap Size",
                                     50,
                                     500,
                                     200,
                                     50,
                                     help="""**Overlap Size**: Number of characters that consecutive chunks share.
                
**Values:**
- **50-100**: Minimal overlap, more independent chunks
- **150-250**: Standard overlap, ensures continuity between chunks
- **300-400**: High overlap, preserves more context across chunk boundaries
- **450-500**: Maximum overlap, significant redundancy but best context preservation

**Purpose**: Prevents information loss at chunk boundaries, especially important for:
- Multi-sentence concepts
- Tables and lists
- Technical specifications
- Step-by-step procedures

**Recommendation**: 200 (20% of chunk size) for most documents, 300+ for complex technical content""")

        with st.expander("🔍 RAG Retrieval Settings"):
            k_documents = st.slider(
                "Documents to Retrieve",
                1,
                20,
                5,
                1,
                help="""**Documents to Retrieve (k)**: Number of most relevant document chunks to provide as context.
                
**Values:**
- **1-3**: Minimal context, fast processing, highly focused answers
- **4-6**: Standard retrieval, good balance of relevance and coverage
- **7-10**: Comprehensive context, slower processing, broader perspective
- **11-20**: Maximum context, may include less relevant information

**Trade-offs:**
- **Higher k**: More comprehensive answers, better coverage of complex topics, slower processing
- **Lower k**: Faster responses, more focused answers, may miss relevant context

**Use Cases:**
- **k=3**: Simple factual questions, quick lookups
- **k=5**: General Q&A, balanced approach
- **k=8-10**: Complex analysis, multi-faceted questions
- **k=15+**: Comprehensive overviews, research summaries

**Recommendation**: Start with 5, increase for complex questions, decrease for simple lookups""")

            score_threshold = st.slider(
                "Similarity Threshold",
                0.0,
                1.0,
                0.5,
                0.1,
                help="""**Similarity Threshold**: Minimum similarity score for including documents in context.
                
**Values:**
- **0.0-0.2**: Very permissive, includes loosely related content
- **0.3-0.5**: Standard filtering, good balance of relevance and coverage  
- **0.6-0.7**: Strict filtering, only highly relevant content
- **0.8-1.0**: Very strict, only extremely similar content (may exclude relevant info)

**How it works**: Documents with similarity scores below this threshold are excluded, even if within the top-k results.

**Impact:**
- **Lower threshold**: More context but potentially less relevant information
- **Higher threshold**: Highly relevant context but may exclude useful information

**Use Cases:**
- **0.3-0.4**: Exploratory questions, broad topics
- **0.5-0.6**: Standard Q&A, balanced approach
- **0.7-0.8**: Specific technical queries, precise answers needed
- **0.9+**: Exact matches only (rarely recommended)

**Recommendation**: 0.5 for general use, 0.3 for broad exploration, 0.7 for precise technical queries""")

        # Load Models Button
        st.subheader("🚀 Initialize System")
        if st.button("Load Models & Process Documents", type="primary"):
            success = True

            # Load embedding model
            if app.load_embedding_model(selected_embedding):
                st.session_state.rag_app.embedding_model = app.embedding_model
            else:
                success = False

            # Load LLM
            llm_kwargs = {'temperature': temperature}
            if llm_provider == "Deltares Ollama":
                llm_kwargs.update({'base_url': base_url, 'model_name': model_name})
            elif llm_provider == "Azure OpenAI":
                llm_kwargs.update({
                    'deployment_name': deployment_name,
                    'api_base': api_base,
                    'api_version': api_version
                })
            else:
                llm_kwargs.update({'model_name': model_name})

            if app.load_llm_model(llm_provider, **llm_kwargs):
                st.session_state.rag_app.llm = app.llm
            else:
                success = False

            # Process documents
            if success and pdf_directory:
                if app.process_documents(pdf_directory, chunk_size, overlap_size):
                    st.session_state.rag_app.vector_store = app.vector_store
                    st.session_state.rag_app.documents_loaded = True
                else:
                    success = False

            if success:
                st.success("🎉 System ready for questions!")
            else:
                st.error("❌ Failed to initialize system")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Questions")

        # Question input
        question = st.text_input("Your Question:",
                                 placeholder="Ask anything about your documents...",
                                 help="Enter your question about the loaded documents")

        # Ask button
        if st.button("🔍 Get Answer", disabled=not st.session_state.vector_store_ready):
            if question:
                # Get the current app instance
                current_app = st.session_state.rag_app

                result = current_app.ask_question(question, k=k_documents, score_threshold=score_threshold)

                if result:
                    # Add to chat history
                    st.session_state.chat_history.append(result)

                    # Display answer
                    st.success("✅ Answer generated!")
                    st.markdown("### 💡 Answer:")
                    st.markdown(result['answer'])

                    # Display relevant contexts
                    with st.expander("📄 Source Documents"):
                        for i, context in enumerate(result['contexts']):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"- **File:** {context.metadata.get('file_name', 'Unknown')}")
                            st.markdown(f"- **Page:** {context.metadata.get('page_number', 'Unknown')}")
                            st.markdown(f"- **Content:** {context.page_content[:300]}...")
                            st.markdown("---")
            else:
                st.warning("Please enter a question")

        # Chat History
        if st.session_state.chat_history:
            st.header("📜 Chat History")
            for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {item['question'][:50]}... ({item['timestamp']})"):
                    st.markdown(f"**Question:** {item['question']}")
                    st.markdown(f"**Answer:** {item['answer']}")

    with col2:
        st.header("ℹ️ System Status")

        # Status indicators
        if st.session_state.get('rag_app'):
            app_state = st.session_state.rag_app

            st.markdown("**📊 Current Configuration:**")
            st.write(f"🧠 Embedding Model: {selected_embedding.split('/')[-1]}")
            st.write(f"🤖 LLM Provider: {llm_provider}")
            st.write(f"📁 Documents Loaded: {'✅ Yes' if app_state.documents_loaded else '❌ No'}")
            st.write(f"💾 Vector Store Ready: {'✅ Yes' if st.session_state.vector_store_ready else '❌ No'}")

            st.markdown("---")

            # Usage tips
            st.markdown("**💡 Usage Tips:**")
            st.markdown("""
            1. **Set PDF Directory**: Point to a folder with PDF files
            2. **Choose Models**: Select embedding and LLM models
            3. **Load System**: Click the load button to initialize
            4. **Ask Questions**: Type questions about your documents
            5. **Review Sources**: Check the source documents for each answer
            
            **🎯 Parameter Optimization Guide:**
            
            **For Technical Documents:**
            - Temperature: 0.1-0.3 (precise, factual responses)
            - Chunk Size: 800-1000 (preserve technical details)
            - Overlap: 250-300 (maintain technical continuity)
            - k: 3-5 (focused technical context)
            - Score Threshold: 0.6-0.7 (high precision)
            
            **For General Knowledge/Research:**
            - Temperature: 0.5-0.7 (balanced creativity)
            - Chunk Size: 1000-1500 (comprehensive context)
            - Overlap: 200-250 (standard continuity)
            - k: 5-8 (broader perspective)
            - Score Threshold: 0.4-0.6 (balanced relevance)
            
            **For Exploratory Analysis:**
            - Temperature: 0.6-0.8 (creative insights)
            - Chunk Size: 1200-1600 (rich context)
            - Overlap: 200-300 (context preservation)
            - k: 8-12 (comprehensive coverage)
            - Score Threshold: 0.3-0.5 (inclusive search)
            
            **For Quick Fact-Finding:**
            - Temperature: 0.1-0.2 (deterministic answers)
            - Chunk Size: 600-800 (focused chunks)
            - Overlap: 150-200 (minimal redundancy)
            - k: 3-4 (targeted retrieval)
            - Score Threshold: 0.6-0.8 (high precision)
            """)

            st.markdown("---")

            # Sample questions
            if st.session_state.vector_store_ready:
                st.markdown("**🔍 Try These Sample Questions:**")
                sample_questions = [
                    "What is the main topic of these documents?", "Can you summarize the key findings?",
                    "What are the main conclusions?", "What methodology was used?", "What are the recommendations?"
                ]

                for sq in sample_questions:
                    if st.button(sq, key=f"sample_{sq[:10]}"):
                        st.session_state.sample_question = sq

        # Clear history button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


if __name__ == "__main__":
    main()
