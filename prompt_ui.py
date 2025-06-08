import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Disable Streamlit watcher for torch classes to prevent errors
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER'] = 'false'

# Page configuration with disable_code_echo
st.set_page_config(
    page_title="Simple Research Paper Summarizer",
    page_icon="📝",
    layout="centered"
)

# Hide the source code display
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Research Paper Summarizer")
st.write("Get concise summaries of AI research papers")

# Input fields
paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need", "BERT", "GPT-3", "Diffusion Models"]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short", "Medium", "Long"]
)

# Function to generate summary
def generate_summary(paper, style, length, model_storage):
    with st.spinner("Generating summary..."):
        try:
            # Create the directory if it doesn't exist
            os.makedirs(model_storage, exist_ok=True)
            
            # Initialize the Hugging Face pipeline with minimal requirements
            llm = HuggingFacePipeline.from_model_id(
                model_id="microsoft/phi-1_5",
                task="text-generation",
                pipeline_kwargs={
                    "temperature": 0.5,
                    "max_new_tokens": 500,
                    "do_sample": True,  # Added to fix the warning
                    # Use CPU only to avoid GPU memory issues
                    "device_map": "cpu"
                },
                # Specify where to download and store the model
                model_kwargs={
                    "cache_dir": model_storage
                }
            )
            
            model = ChatHuggingFace(llm=llm)
            
            # Define a simple prompt template
            template = PromptTemplate.from_template("""
                Summarize the paper '{paper_input}' in a {style_input} style.
                Length: {length_input}.
                Provide key insights and methodologies clearly.
            """)
            
            # Create and run the chain
            chain = template | model
            result = chain.invoke({
                "paper_input": paper,
                "style_input": style,
                "length_input": length
            })
            
            return result.content
            
        except Exception as e:
            st.error(f"Error: {e}")
            return None

# Generate button with progress indicator
if st.button("Summarize"):
    # Get model storage location from sidebar
    model_storage = st.session_state.get('model_storage', "D:/huggingface_models")
    
    # Show a warning about processing time
    st.warning("Using CPU-only mode. This may take a few minutes on limited hardware.")
    
    # Progress bar for user feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate progress while actually generating
    import time
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 10:
            status_text.text("Initializing the model...")
        elif i < 30:
            status_text.text(f"Loading model from {model_storage}...")
        elif i < 80:
            status_text.text("Generating summary...")
        else:
            status_text.text("Finalizing output...")
        time.sleep(0.1)
    
    # Generate the actual summary
    summary = generate_summary(paper_input, style_input, length_input, model_storage)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display the summary
    if summary:
        st.subheader(f"Summary of {paper_input}")
        st.write(summary)
        
        # Add download option
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name=f"{paper_input.replace(' ', '_')}_summary.txt",
            mime="text/plain"
        )

# Add a section for tips and info
with st.expander("Tips for using this app"):
    st.markdown("""
    ### Tips to improve performance on limited hardware:
    
    - Close other applications before running the app
    - Be patient during the first run as the model needs to load
    - Start with 'Short' summaries if having memory issues
    - If you encounter persistent errors, try restarting the application
    
    ### Paper information:
    
    - **Attention Is All You Need**: Introduces the Transformer architecture
    - **BERT**: Bidirectional Encoder Representations from Transformers
    - **GPT-3**: Generative Pre-trained Transformer 3 by OpenAI
    - **Diffusion Models**: Probabilistic models for image generation
    """)

# Sidebar for configuration and information
with st.sidebar:
    st.header("Configuration")
    
    # Option to change model storage location
    model_storage = st.text_input(
        "Model Storage Location",
        value="D:/huggingface_models",
        help="Directory where models will be stored"
    )
    
    # Option to clear cache
    if st.button("Clear Cache (if facing memory issues)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared successfully!")
    
    # Display model storage information
    st.info(f"Models will be stored in: {model_storage}")
    st.caption("Make sure this drive has enough free space (~2GB)")
    
    # Add information about storage
    st.markdown("""
    ### Storage Information
    
    The model files will be downloaded to the specified directory 
    on first run. This helps:
    
    - Prevent filling up your system drive
    - Allow reuse of models between sessions
    - Improve loading time for subsequent runs
    """)