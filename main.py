import os
import json
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

def extract_resume_info(file_path):
    # Step 1: Read the PDF and extract text
    reader = PdfReader(file_path)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    
    # Step 2: Split the document into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100  # Slight overlap for better context retention
    )
    docs = text_splitter.create_documents(formatted_document)
    
    # Step 3: Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(docs, embeddings)
    
    # Step 4: Initialize the GroqChat model
    llm = ChatGroq(
        temperature=0.2,
        model="llama-3.1-70b-versatile",
        api_key="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"  # Ensure the API key is set as an environment variable
    )
    
    # Step 5: Create the retrieval chain
    retriever = store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 matches for better context
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # Step 6: Define extraction queries
    queries = {
        "relevant_title": "What is the main job title or role of the person?",
        "years_of_experience": "How many years of experience does this person have?",
        "techstack": "List the technologies, tools, and frameworks this person is proficient in.",
        "current_location": "Where is the current location of the person?",
        "certifications": "List all certifications the person has obtained.",
        "native_languages_known": "What languages does this person speak natively?",
        "computer_languages_known": "What programming or computer languages does this person know?"
    }
    
    # Step 7: Extract information
    resume_info = {}
    for key, query in queries.items():
        try:
            response = retrieval_chain.run(query)
            resume_info[key] = response
        except Exception as e:
            resume_info[key] = f"Error extracting data: {e}"
    
    return resume_info

# Main execution block
if __name__ == "__main__":
    # Specify the PDF file path
    file_path = "Resume_SDE.pdf"  # Replace with the actual file path

    if not os.path.exists(file_path):
        print("Error: File not found. Please provide a valid PDF file path.")
    else:
        try:
            # Extract resume information
            extracted_info = extract_resume_info(file_path)
            
            # Print and save the output in JSON format
            json_output = json.dumps(extracted_info, indent=4)
            print(json_output)
            
            # Save to a file
            output_file = "resume_info.json"
            with open(output_file, "w") as f:
                f.write(json_output)
            print(f"Resume information saved to {output_file}.")
        except Exception as e:
            print(f"An error occurred: {e}")
