import os
import openai
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def connect_to_db():
    """Establish a connection to the MongoDB database."""
    client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
    db = client[os.getenv("MONGO_DB_NAME")]
    return db[os.getenv("MONGO_COLLECTION_NAME")]

def classify_query(query):
    """Classify the query type using GPT-4."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Classify the query into 'count', 'positive', 'negative', or 'specific'. Respond with only the category."},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
    )
    return response.choices[0].message["content"]

def chunk_text(text, chunk_size=300, chunk_overlap=30):
    """Split text into overlapping chunks for embedding generation."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def generate_chunks(collection):
    """Generate chunks for reviews without existing chunks."""
    reviews = collection.find({"chunks": {"$exists": False}}, {"_id": 1, "ratingText": 1})
    for review in reviews:
        text = review.get("ratingText", "").strip()
        if text:
            chunks = chunk_text(text)
            collection.update_one({"_id": review["_id"]}, {"$set": {"chunks": chunks}})

def generate_embeddings(collection):
    """Generate embeddings for chunks without embeddings."""
    reviews = collection.find({"chunks": {"$exists": True}, "embeddings": {"$exists": False}}, {"_id": 1, "chunks": 1})
    for review in reviews:
        embeddings = [
            openai.Embedding.create(input=chunk, model="text-embedding-ada-002")["data"][0]["embedding"]
            for chunk in review.get("chunks", [])
        ]
        collection.update_one({"_id": review["_id"]}, {"$set": {"embeddings": embeddings}})

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_relevant_reviews(query, location_ids, third_party_ids, company_ids, collection, top_k=50):
    """Retrieve the most relevant reviews for a query using embeddings."""
    query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    results = []

    reviews = collection.find(
        {
            "locationId": {"$in": location_ids},
            "thirdPartyReviewSourcesId": {"$in": third_party_ids},
            "companyId": {"$in": company_ids},
        },
        {"chunks": 1, "embeddings": 1},
    )

    for review in reviews:
        for chunk, embedding in zip(review.get("chunks", []), review.get("embeddings", [])):
            similarity = cosine_similarity(query_embedding, embedding)
            results.append({"text": chunk, "similarity": similarity})

    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]

def build_aggregation_pipeline(location_ids, third_party_ids, company_ids, item_name=None, filters=None):
    """Construct a MongoDB aggregation pipeline based on query parameters."""
    pipeline = [
        {
            "$match": {
                "locationId": {"$in": location_ids},
                "thirdPartyReviewSourcesId": {"$in": third_party_ids},
                "companyId": {"$in": company_ids},
            }
        }
    ]

    if item_name:
        pipeline[0]["$match"].update({"ratingText": {"$regex": item_name, "$options": "i"}})

    group_stage = {"$group": {"_id": None, "total_reviews": {"$sum": 1}}}

    if filters:
        for key, condition in filters.items():
            group_stage["$group"].update({
                f"{key}_reviews": {
                    "$sum": {
                        "$cond": [
                            {condition["operator"]: [f"${condition['field']}", condition["value"]]},
                            1,
                            0,
                        ]
                    }
                }
            })

    pipeline.append(group_stage)
    pipeline.append({"$project": {"_id": 0, "total_reviews": 1, **{f"{key}_reviews": 1 for key in filters.keys()}}})
    return pipeline

def generate_response(query, reviews):
    """Generate a response to the user query based on relevant reviews."""
    if not reviews:
        return "No relevant reviews found."

    context = "\n\n".join([review["text"] for review in reviews])
    prompt = f"Query: {query}\n\nReviews:\n{context}\n\nAnswer the query:"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate responses based on the query."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message["content"]

def run_pipeline(user_query, location_ids, third_party_ids, company_ids, collection, top_k=50):
    """Run the complete query processing pipeline."""
    query_type = classify_query(user_query)
    item_name = user_query.split("about")[-1].strip() if "about" in user_query else None

    if query_type == "count":
        filters = {}
        if "positive" in user_query.lower():
            filters["positive"] = {"field": "ratingValue", "operator": "$gte", "value": 4}
        if "negative" in user_query.lower():
            filters["negative"] = {"field": "ratingValue", "operator": "$lte", "value": 2}

        pipeline = build_aggregation_pipeline(location_ids, third_party_ids, company_ids, item_name, filters)
        result = next(collection.aggregate(pipeline), {})

        response_parts = [f"The total number of reviews is {result.get('total_reviews', 0)}."]
        if "positive_reviews" in result:
            response_parts.append(f"Positive reviews: {result['positive_reviews']}.")
        if "negative_reviews" in result:
            response_parts.append(f"Negative reviews: {result['negative_reviews']}.")
        return " ".join(response_parts)

    relevant_reviews = retrieve_relevant_reviews(user_query, location_ids, third_party_ids, company_ids, collection, top_k)
    return generate_response(user_query, relevant_reviews)

def process_input(input_str):
    """Safely process user input for single or multiple values."""
    try:
        parsed = eval(input_str)
        return parsed if isinstance(parsed, list) else [parsed]
    except:
        return [input_str]

if __name__ == "__main__":
    openai.api_key = os.getenv("API_KEY")
    collection = connect_to_db()

    location_ids = process_input(input("Enter Location ID(s): "))
    third_party_ids = process_input(input("Enter Third Party Review Source ID(s): "))
    company_ids = process_input(input("Enter Company ID(s): "))
    user_query = input("Enter Your Query: ").strip()

    generate_chunks(collection)
    generate_embeddings(collection)

    response = run_pipeline(user_query, location_ids, third_party_ids, company_ids, collection)
    print("\nAI Response:\n")
    print(response)
