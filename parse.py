from llama_cloud_services import LlamaParse
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI  # Correct import
# imports
from llama_index.embeddings.mistralai import MistralAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLAMA Cloud
os.environ["LLAMA_CLOUD_API_KEY"] = "ENCRYPTED

# Initialize Embeddings
api_key = "ENCRYPTED"
model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)
# Initialize LLM - CORRECTED
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Set settings
Settings.llm = llm
Settings.embed_model = embed_model

# Parse document
documents = LlamaParse(result_type="markdown").load_data("referral_package.pdf")


# Node parsing
node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)  # Reuse the same llm instance
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

# Create index and query engine
recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
query_engine = recursive_index.as_query_engine(similarity_top_k=25)

# Query processing
medical_queries = [
            "What is the patient's full name, medical record number, and date of birth?",
            "What insurance information is provided including provider, subscriber details, and policy numbers?",
            "What is the primary diagnosis and all related medical codes?",
            "What is the complete medical history and list of symptoms mentioned?",
            "What medications is the patient currently taking and what treatments are scheduled?",
            "Who are all the healthcare providers mentioned including names, specialties, and contact information?",
            "What language preferences or special considerations are mentioned for this patient?",
            "What are all the dates, appointments, and scheduled events mentioned?",
            "What allergies or adverse reactions are documented?",
            "What hospital admissions or previous treatments are referenced?"
        ]



document2 = LlamaParse(result_type="markdown").load_data("PA.pdf")

node_parser2 = MarkdownElementNodeParser(llm=llm, num_workers=8)  # Reuse the same llm instance
nodes2 = node_parser2.get_nodes_from_documents(document2)
base_nodes2, objects2 = node_parser2.get_nodes_and_objects(nodes2)

recursive_index2 = VectorStoreIndex(nodes=base_nodes2 + objects2)
query_engine2 = recursive_index2.as_query_engine(similarity_top_k=25)

fields = "what are the patient information fields that you can extract from this document and try to answer them with info you extracted from first pdf file"

response = query_engine.query(fields)
print(f"Response: {response}\n{'-'*50}")

#for q in medical_queries:
    #response = query_engine.query(q)

    #print(f"Query: {q}\nResponse: {response}\n{'-'*50}")



    
