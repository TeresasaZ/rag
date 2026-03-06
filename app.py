import ollama
from qdrant_client import models, QdrantClient
import openai
import os

openai_client = openai.OpenAI(base_url= "https://models.github.ai/inference", api_key = os.environ['GITHUB_TOKEN'])
 
qd_client = QdrantClient("http://localhost:6333")
# qd_client = 'jina-small'
ollama_client = ollama.Client(host="http://localhost:11434")
 
def vector_search(question):
    query_points = qd_client.query_points(
        collection_name='met-museum-artworks', # database
        # sql query
        query=models.Document(
            text=question,
            model="jinaai/jina-embeddings-v2-small-en"
        ),
        #schema
        using="jina-small",
        #top 100
        limit=5,
        # select
        with_payload=True
    )
 
    results = []

    print(query_points)

 
    for point in query_points.points:
        results.append(point.payload)

    
    return results
 
def rag_query(question):
    # Get the chunks from Qdrant
    search_results = vector_search(question)
 
    context = ""
    # format the results
    for results in (search_results):
        print("\n")
        print(results)
        print("\n")
        context += str(results) +"\n\n"
 
    prompt = f"""Use the context below to answer the question. If the answer isn't in the context, say so.
 
Context:
{context}
 
Question: {question}
 
Answer:"""
 
    # response = ollama_client.generate(
    #     model="tinyllama",
    #     prompt=prompt
    # )

    response = openai_client.chat.completions.create(
        model = os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"),
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
 
    #return response['response']
 
def build_prompt(query, search_results):
   
    context = ""
 
    for doc in search_results:
        #print(doc['artwork_text'])
        #print('\n')
        context = context + f"artwork_description: {doc['artwork_text']}\nimage_url: {doc['artwork_image_url']}\ngallery_link: {doc['artwork_gallery_link']}\n"
 
    prompt = f"""
        You are an AI assistant that answers user questions about artworks in the European Paintings collection at the Metropolitan Museum of Art.
        You will be given context information from the museum's knowledge base. Use ONLY this context to answer the user's question.
        If the answer cannot be found in the context, say "I could not find that information in the collection. Could you elaborate further on your question?" and then explain what is confusing you about the question.
        Do not hallucinate or make up facts.
 
        Guardrails:
        - Answer in clear, concise, and user-friendly language.
        - If available, include the title, artist, date, medium, and link to the official museum page.
        - If images are provided in the context, return their URLs so they can be displayed.
        - Keep the response factual and grounded in the provided context.
 
       
        User Question:
        {query}
 
        Context:
        {context}
 
 
""".strip()
   
   
    return prompt
 
def rag(query):
    print("Looking for relevant documents!")
    search_results = vector_search(query)
    print("Found relevant documents!")
    prompt = build_prompt(query, search_results)
    print("Answering Question\n")
    # answer = ollama_client.generate(
    #     model="tinyllama",
    #     prompt=prompt
    # )
    answer = openai_client.chat.completions.create(
        model=os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"),
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
 
    #return answer["response"]
    return answer.choices[0].message.content.strip()

# question = 'Which paintings in the MET are by an Italian artist'
 
# answer = rag_query(question)
# print(answer)

# my comments!!!