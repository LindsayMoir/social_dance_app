from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Ship Point, Victoria, BC V8W 1T3",
    "Ship Point (Inner Harbour), Wharf and Broughton, Victoria, BC",
    "Ship Point (Inner Harbour), Wharf and Broughton, Victoria, BC V8W 1T3"
]

embeddings = model.encode(texts)
sim_matrix = cosine_similarity(embeddings)

print(sim_matrix)
