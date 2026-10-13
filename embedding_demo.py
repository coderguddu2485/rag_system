
#               this  is the converted the chunk to vector using sentence transformer model


# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2") 

# sentence = "I love programming."

# embedding = model.encode(sentence)

# print(type(embedding))
# print(len(embedding))
# print(embedding[:10])


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love programming.",
    "Coding is my passion.",
    "I enjoy writing software.",
    "The sun is very hot.",
    "Today it is sunny."
]

embeddings = model.encode(sentences)

similarity = cosine_similarity(embeddings)

print(similarity)