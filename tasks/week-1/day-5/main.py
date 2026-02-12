import chromadb

client = chromadb.PersistentClient(path="D:\\Work\\gen-ai\\tasks\\week-1\\day-5\\storage")

collection = client.get_or_create_collection("my-docs")

collection.upsert(
    documents= ["this is doc3", "this is doc4"],
    metadatas=[{"source": "notion"}, {"source": "google-doc"}],
    ids=["doc3", "doc4"]
)

results = collection.query(
    query_texts=["this is a query doc2"]
)
print(results)