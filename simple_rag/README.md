# Mini Q&A using RAG

The `simple_rag/` directory contains a lightweight Q&A application using RAG. The application supports retrieving text content from a provided file. This implementation does *not* use LangChain.

## Getting started

To run the app, execute:

```bash
pip install -r simple_rag/requirements.txt
python3 simple_rag/q_and_a_app.py --path some_file.txt
```

You will be prompted to input questions, and the app will generate answers.

## Components

### 1. Embeddings

We need to be able to convert text into embeddings (semantic vectors). This is implemented in `embeddings.py`. The main functionalities are:

- `get_embedding(text: str, model: str)`: gets embedding for the given text
- `cosine_similarity(vector1: List[float], vector2: List[float])`: calculates cosine similarity between two vectors

To test our implementation, we can run

```
python3 simple_rag/test_embeddings.py
```

Output:

```
text1: Kelly got a kitty for Christmas
=> embedding length: 1536
text2: Jonas wanted a pet as a gift
=> embedding length: 1536
text3: I had a dream last night
=> embedding length: 1536

Cosine similarity (1 and 2): 0.4107805791230994
Cosine similarity (1 and 3): 0.15371315012434492
Cosine similarity (2 and 3): 0.15068478771970678
```

The similarity score between text1 and text2 is a lot higher than the other two scores, which is exactly what we would expect.

### 2. Chunker

We need to chunk the text into smaller segments. This is implemented in `chunker.py`.

The chunker takes below arguments:

- `path`: relative path of the input file
- `chunk_size`: size of each chunk; this is the number of *tokens*, not of *characters*
- `chunk_overlap`: how many tokens can overlap between chunks; *this avoids losing context at chunk boundaries

The chunker's key functionalities are:

- `read_content()`: reads the content of the given file
- `get_chunks(content)`: splits the content into chunks; returns a list of chunks

To test our implementation, we can run

```
python3 simple_rag/test_chunker.py
```

This will split `test.txt`, which contains the first 10 chapters of Pride and Prejudice, into smaller chunks.

Output:

```
Content length: 86456
Number of chunks: 244
=== Chunk 1 ===
Chapter 1

It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.

“My dear Mr. Bennet,” said his lady to him one day, “have you heard that

=== Chunk 2 ===
“My dear Mr. Bennet,” said his lady to him one day, “have you heard that Netherfield Park is let at last?”

Mr. Bennet replied that he had not.

“But it is,” returned she; “for Mrs. Long has just been here, and she told me all about it.”

Mr. Bennet made no answer.

“Do you not want to know who has taken it?” cried his wife impatiently.

“YOU want to tell me, and I have

=== Chunk 3 ===
 has taken it?” cried his wife impatiently.

“YOU want to tell me, and I have no objection to hearing it.”

This was invitation enough.

“Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he

```

Note the overlap between the chunks.

### 3. Vector store

We then need to convert the chunks into embeddings and store them in a vector store. This is implemented in `vector_store.py`. The key functionalities are:

- `build_store()`: convert the list of text chunks into a list of embeddings
- `persist_store()`: persist the vector store to a directory as json files; we store both the original text chunks and the embedding vectors
- `load_store()`: load the vector store from the storage directory
- `query_store(query: str, top_k: int)`: query the vector store for the most similar chunks to the given query

To test our vector store implementation, run

```
python3 simple_rag/test_vector_store.py
```

This may take a minute to run, but the output is as follows:

```
=== Result 1 ===
y returned Mr. Bennet’s visit, and sat about ten minutes with him in his library. He had entertained hopes of being admitted to a sight of the young ladies, of whose beauty he had heard much; but he saw only the father. The ladies were somewhat more fortunate, for they had the advantage of ascertaining from an upper window that he wore a blue coat, and rode a black horse.

An invitation to dinner was soon afterwards dispatched; and already had Mrs. Bennet planned

=== Result 2 ===
 with. Mrs. Bennet, accompanied by her two youngest girls, reached Netherfield soon after the family breakfast.

Had she found Jane in any apparent danger, Mrs. Bennet would have been very miserable; but being satisfied on seeing her that her illness was not alarming, she had no wish of her recovering immediately, as her restoration to health would probably remove her from Netherfield. She would not listen, therefore, to her daughter’s proposal of being carried home; neither did the ap

=== Result 3 ===
; four or five thousand a year. What a fine thing for our girls!”

“How so? How can it affect them?”

“My dear Mr. Bennet,” replied his wife, “how can you be so tiresome! You must know that I am thinking of his marrying one of them.”

“Is that his design in settling here?”

“Design! Nonsense, how can you talk so! But it is very likely that he MAY fall in love with one of them, and therefore you must

```

## Putting everything together

The main logic of the Q&A app is in `q_and_a_app.py`. There is also a test file `test.txt`. It contains the first 10 chapters of Pride and Prejudice.

You can run the app using:

```
python3 simple_rag/q_and_a_app.py --path simple_rag/test.txt
```

The output, including 2 sample questions, is below. The app was able to answer questions about the text in its own words!

```
(1) Loading content...
 - loaded 25 chunks total
(2) Building vector store...
 - vector store loaded from storage
(3) Initializing LLM...
Question: what does Mrs. Bennett want Mr. Bennett to do?
Answer: Mrs. Bennet wants Mr. Bennet to visit Mr. Bingley as soon as he comes into the neighborhood, as she is eager for him to meet one of their daughters and hopes that he may fall in love with one of them. She believes that Mr. Bingley, being a single man of large fortune, would be a fine match for one of her daughters.

Question: How did Mr. Bennett respond to Mrs. Bennett's request?
Answer: Mr. Bennet responded to Mrs. Bennet's request to visit Mr. Bingley by initially expressing indifference and saying he saw no occasion to visit. He suggested that Mrs. Bennet and the girls could go or even send them by themselves, indicating that he was not particularly eager to make the visit himself. Ultimately, despite his reluctance, he did visit Mr. Bingley, but only after assuring his wife that he would not go, which he later revealed to the family as a surprise.
```