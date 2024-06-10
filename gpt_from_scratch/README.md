## Implementing GPT from scratch

following `https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7`

### notes
self attention: 
each token (llm lingo, usually this means a word) emits two vectors: a key and a query vector. The former basically asks what am i looking for and the latter what do i contain? Concatenation those vectors yields a query and key matrix. If query and key vector align (ie key has what query is looking for results in a longer vector)