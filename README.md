# Locally powered RAG chatbot 

The chatbot is powered by [Mistral-7B](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF). It is meant to be ran locally though the it is not hard to refactor to make it work with and LLM API such as Chatgpt or Claude.
Here is the stack of technologies that power this program:
* Streamlit for the UI.
* LlamaCPP for calling the LLM.
*  [Instructor XL](https://huggingface.co/hkunlp/instructor-xl) as the embedding model.
* HuggingFace embeddings for  calling the embedding
* ChromaDB for storing the Vector Database
* Langchain as the LLM framework

By default the Vector Database is computed in the CPU because my local machine doesn't have enough VRAM to run both the embeddings and the LLM at the same time in the GPU, for a production ready system be mindful of using a GPU that at least has 12gb of VRAM, or use cloud.

Also be mindful if you want to run this on your own that you need to install [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers).That library tends to brake with certain models in the in this case I use [this solution](https://github.com/Muennighoff/sgpt/issues/14#issuecomment-1405205453) in order to ran Instructor-XL on HuggingFace Embeddings
## How to run this App:
* Clone this repository
```
https://github.com/EmanuelRiquelme/Local-chatbot-UI
```
* Setup conda
```
conda create -n rag-ui python=3.11.5
conda activate rag-ui
pip3 install -r -q requirements.txt
pip3 install -U sentence-transformers
```
* Download the model
```
pip3 install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```
* Finally run the program
```
streamlit run app.py
```
### Hardware used: 
* CPU: i5 11400f
*  GPU: NVIDIA RTX 3070 TI
*  RAM: 32GB 
* OS: Ubuntu 22.04.4 LTS
*  Python version: Python 3.10.13
