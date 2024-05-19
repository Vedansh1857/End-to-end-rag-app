# End-to-end-rag-app

### 1. Took a blog post on LLM, scraped it using beautifulsoup.
### 2. Took the content & split it into chunks to feed it to db.
### 3. Used Hugginface Embeddings to convert the chunks into vectors.
### 4. Used cassandra(AstrDB) to store these these embeddings.
### 5. Used Mixtral model & our custom prompt template.
### 6. Finally, created a retrieval chain and invoked it to get results.
