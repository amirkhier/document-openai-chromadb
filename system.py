import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI 

# Load environment variables from .env file
load_dotenv()

# Get API Key from .env file
openai_key = os.getenv("OPENAI_API_KEY")

# Activating the embedding functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key, model_name="text-embedding-3-small")

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

# Use the OpenAI API directly for queries
client = OpenAI(api_key=openai_key)

# # Example OpenAI API usage
# response = client.chat.completions.create(
#     model='gpt-3.5-turbo',
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What is human life expectancy in the United States?",
#         },
       
#     ]
# )
# print(response.choices[0].message.content)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

#load documents from directroy
directory_path = './news_articles'
documents = load_documents_from_directory(directory_path)

print(f"Loaded No.Documents : {len(documents)}")

chunked_documents = []
for doc in documents:
    chunks = split_text(doc['text'])
    print('==== Spliting chunks from document ====')
    for i,chunk in enumerate(chunks):
        chunked_documents.append({'id' : f"{doc['id']}_chunk{i+1}" , 'text' : chunk})

# print(f"Loaded Chunked documents : {len(chunked_documents)}")

#function to generate embedding with openai client:
def get_openai_embedding(text):
    resp = client.embeddings.create(input=text,model='text-embedding-3-small')
    embedding = resp.data[0].embedding
    print('==== Generating Embedding ====')
    return embedding

#adding embedding to the chunked documents list
for doc in chunked_documents:
    doc['embedding'] = get_openai_embedding(doc['text'])


#adding the chunked embeddded procced to vector database chroma:

for doc in chunked_documents:
    print('==== Inserting Chunks into db====')
    collection.upsert(ids=[doc['id']],documents=[doc['text']],embeddings=[doc['embedding']])

# Function to query documents
def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")

# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message.content
    return answer

while True:
    #getting relevant chunks/documents from input question

    question = input("Enter a question :  , press -1 to stop : ")
    if question == '-1':
        break
    relevant_chunks = query_documents(question)
    #getting the result:
    result = generate_response(question,relevant_chunks)

    print(result)



