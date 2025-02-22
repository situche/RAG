import faiss
import nltk
import warnings
import numpy as np
from openai import OpenAI
from nltk.tokenize import sent_tokenize
from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_pages
from sentence_transformers import CrossEncoder, SentenceTransformer

warnings.simplefilter("ignore")
 
nltk.download('punkt') 
nltk.download('stopwords') 

client = OpenAI(api_key="OPENAI_API_KEY", base_url="https://api.deepseek.com")

class FaissVectorDBConnector:
    def __init__(self, embedding_fn, dimension):
        self.embedding_fn = embedding_fn
        self.index = faiss.IndexFlatL2(dimension)

        self.documents = []
        self.metadata = []

    def add_documents(self, documents, metadata={}):
        embeddings = np.array(self.embedding_fn([documents])).astype('float32')
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.metadata.extend([metadata] * len(documents))

    def search(self, query, top_n):
        query_embedding = np.array(self.embedding_fn([query])).astype('float32')
        distances, indices = self.index.search(query_embedding, top_n)

        results = {
            'documents': [self.documents[i] for i in indices[0]],
            'distances': distances[0],
            'metadata': [self.metadata[i] for i in indices[0]]
        }
        return results

class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results
    
    def chat(self, user_query, model, prompt_template):
        search_results = self.vector_db.search(user_query, self.n_results)

        scores = model.predict([(user_query, doc) for doc in search_results['documents'][0]])
        sorted_list = sorted(
            zip(scores, search_results['documents'][0]), 
            key=lambda x: x[0],
            reverse=True
        )
        scores, best_document = sorted_list[0]

        prompt = build_prompt(
            prompt_template, 
            info=best_document, 
            query=user_query
        )

        response = self.llm_api(prompt)
        return response

def process_pdf_and_split(filename, page_number=None, min_line_length=1, chunk_size=300, overlap_size=100):
    paragraphs = []
    buffer = ''

    for i, page_layout in enumerate(extract_pages(filename)):
        if page_number is not None and i not in page_number:
            continue

        full_text = ''
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
                
        lines = full_text.split('\n')
        for text in lines:
            if len(text) >= min_line_length:
                buffer += (''+text) if not text.endswith('-') else text.strip('-')
            elif buffer:
                paragraphs.append(buffer)
                buffer = ''
        if buffer:
            paragraphs.append(buffer)
    
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i = 0

    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev = i - 1
        while prev >= 0 and len(sentences[prev]) + len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        
        chunk = overlap + chunk
        next = i + 1

        while next < len(sentences) and len(sentences[next]) + len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        
        chunks.append(chunk)
        i = next
        
    return chunks

def get_embeddings(texts, model='text-embedding-ada-002'):
    data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]


def build_prompt(prompt_template, **kwargs):
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        
    prompt = prompt.replace(f'__{k.upper()}__', val)
    return prompt

prompt_template = '''
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全一句下述已知信息，不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复“我无法回答您的问题”。

已知信息：
__INFO__

用户问题：
__QUERY__

请用中文回答用户问题。
'''

paragraphs = process_pdf_and_split('llava.pdf', page_number=[1, 2], min_line_length=10)
model = CrossEncoder('cross-encoder/ma-marco-MiniLM-L-6-v2', max_length=512)

vector_db = FaissVectorDBConnector(get_embeddings, 1536)
vector_db.add_documents(paragraphs)

user_query = 'miniCPM模型有多少个参数？'

bot = RAG_Bot(vector_db, llm_api=get_embeddings)
response = bot.chat(user_query, model, prompt_template)
print(response)
