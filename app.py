import os
import openai
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import ast
from dotenv import load_dotenv

load_dotenv()

def connect_to_db():
    """Initialize and return MongoDB connection."""
    try:
        connection_string = os.getenv("MONGO_CONNECTION_STRING")
        db_name = os.getenv("MONGO_DB_NAME")
        collection_name = os.getenv("MONGO_COLLECTION_NAME")

        client = MongoClient(connection_string)
        db = client[db_name]
        collection = db[collection_name]
        print("Connected to MongoDB successfully.")
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None


def gpt_classify(query):
    """Classify the query as either 'count' or 'reviews'."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a decision-making agent. Your task is to analyze the user's query and categorize it into one of two types: 'count' or 'reviews'. Based on the query, respond only with 'count' or 'reviews', without adding any additional information."},
                      {"role": "user", "content": query}],
            temperature=0.0
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error: {str(e)}, Traceback: {e.__traceback__}"


def chunk_text(text, chunk_size=300, chunk_overlap=30):
    """Split long text into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def generate_and_store_chunks(collection):
    """Process reviews and generate chunks if not already present."""
    reviews = collection.find({"chunks": {"$exists": False}},
                               {"_id": 1, "ratingText": 1, "sentimentAnalysis": 1, "reviewerTitle": 1,
                                "satisfactoryLevel": 1, "date": 1, "locationId": 1,
                                "thirdPartyReviewSourcesId": 1, "companyId": 1})

    for review in reviews:
        combined_text = f"{review.get('ratingText', '')} {review.get('sentimentAnalysis', '')} " \
                        f"{review.get('reviewerTitle', '')} {review.get('satisfactoryLevel', '')} " \
                        f"{review.get('date', '')} {review.get('locationId', '')} " \
                        f"{review.get('thirdPartyReviewSourcesId', '')} {review.get('companyId', '')}"

        if combined_text.strip():
            chunks = chunk_text(combined_text)
            collection.update_one({"_id": review["_id"]}, {"$set": {"chunks": chunks}})


def generate_embeddings_for_chunks(collection):
    """Generate embeddings for the chunks if not already present."""
    reviews = collection.find({"chunks": {"$exists": True}, "embeddings": {"$exists": False}},
                               {"_id": 1, "chunks": 1})

    for review in reviews:
        embeddings = [openai.Embedding.create(input=chunk, model="text-embedding-ada-002")["data"][0]["embedding"]
                      for chunk in review.get("chunks", [])]
        collection.update_one({"_id": review["_id"]}, {"$set": {"embeddings": embeddings}})


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_relevant_reviews(query, location_ids, third_party_ids, company_ids, collection, top_k=50):
    """Retrieve reviews relevant to the user's query."""
    query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    results = []

    for location_id in location_ids:
        for third_party_id in third_party_ids:
            for company_id in company_ids:
                reviews = collection.find({"locationId": location_id, "thirdPartyReviewSourcesId": third_party_id,
                                           "companyId": company_id}, {"_id": 1, "chunks": 1, "embeddings": 1})

                for review in reviews:
                    if "embeddings" in review:
                        for chunk, embedding in zip(review["chunks"], review["embeddings"]):
                            similarity = cosine_similarity(query_embedding, embedding)
                            results.append({"text": chunk, "similarity": similarity})

    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]

def build_dynamic_aggregation_pipeline(
    location_ids, third_party_ids, company_ids, filters=None
):
    match_stage = {
        "$match": {
            "locationId": {"$in": location_ids},
            "thirdPartyReviewSourcesId": {"$in": third_party_ids},
            "companyId": {"$in": company_ids},
        }
    }
    group_stage = {
        "$group": {
            "_id": None,
            "total_reviews": {"$sum": 1},
        }
    }
    if filters:
        for key, condition in filters.items():
            group_stage["$group"][f"{key}_reviews"] = {
                "$sum": {"$cond": [{"$expr": {condition["operator"]: [f"${condition['field']}", condition["value"]]}}, 1, 0]}
            }
    project_stage = {
        "$project": {
            "_id": 0,
            "total_reviews": 1,
        }
    }
    if filters:
        for key in filters.keys():
            project_stage["$project"][f"{key}_reviews"] = 1

    return [match_stage, group_stage, project_stage]

def generate_response(query, reviews):
    """Generate a response using GPT-4 based on the retrieved reviews."""
    if not reviews:
        return "No relevant reviews found."

    context = "\n\n".join([review['text'] for review in reviews])
    prompt = f"""
    You are an AI assistant. Use the following reviews to answer the query concisely and informatively.

    Query: {query}

    Reviews:
    {context}

    Answer the query:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]

def run_pipeline(user_query, location_ids, third_party_ids, company_ids, collection, top_k=50):
    """Determine query type and execute the appropriate pipeline."""
    query_type = gpt_classify(user_query)

    print("_______________________________________")
    print(query_type)
    print("_______________________________________")
    
    if query_type == 'count':
        filters = {}
        if "positive" in user_query.lower():
            filters["positive"] = {"field": "ratingValue", "operator": "$gte", "value": 4}
        if "negative" in user_query.lower():
            filters["negative"] = {"field": "ratingValue", "operator": "$lte", "value": 2}

        pipeline = build_dynamic_aggregation_pipeline(location_ids, third_party_ids, company_ids, filters)
        result = list(collection.aggregate(pipeline))[0] if collection.aggregate(pipeline) else {}

        response_parts = [f"The total number of reviews is {result.get('total_reviews', 0)}."]
        if "positive_reviews" in result:
            response_parts.append(f"Positive reviews: {result['positive_reviews']}.")
        if "negative_reviews" in result:
            response_parts.append(f"Negative reviews: {result['negative_reviews']}.")
        return " ".join(response_parts)
    else:
        relevant_reviews = retrieve_relevant_reviews(user_query, location_ids, third_party_ids, company_ids, collection, top_k)
        return generate_response(user_query, relevant_reviews)
def process_input_field(input_str):
    """Process the input to handle both single values and lists."""
    try:
        return ast.literal_eval(input_str) if isinstance(ast.literal_eval(input_str), list) else [input_str]
    except:
        return [input_str]


if __name__ == "__main__":
    openai.api_key = os.getenv("API_KEY")
    collection = connect_to_db()

    print("Starting chunk and embedding generation process...")
    generate_and_store_chunks(collection) 
    generate_embeddings_for_chunks(collection)
    print("Chunks and embeddings are created and stored successfully.")

    location_id_input = input("Enter Location ID(s): ")
    third_party_input = input("Enter Third Party Review Source ID(s): ")
    company_id_input = input("Enter Company ID(s): ")
    user_query = input("Enter Your Query: ")

    location_ids = process_input_field(location_id_input)
    third_party_ids = process_input_field(third_party_input)
    company_ids = process_input_field(company_id_input)


    user_query = user_query.strip().lower()

    print("Processing your query...")
    response = run_pipeline(user_query, location_ids, third_party_ids, company_ids, collection, top_k=50)
    print("\nAI Response:\n")
    print(response)
