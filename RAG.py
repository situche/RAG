import os
import re
import nltk
import warnings
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv, find_dotenv
from pdfminer.layout import LTTextContainer
from pdfminer.high_level import extract_pages
from elasticsearch7 import Elasticsearch, helpers

_ = load_dotenv(find_dotenv())

warnings.simplefilter('ignore')

nltk.download('punkt')
nltk.download('stopwords')

es = Elasticsearch(
    hosts=['http://117.50.198.53:9200'],
    http_auth=('elastic', 'FKaB1Jpz0Rlw0l6G'),
    timeout=30
)

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url= os.getenv('OPENAI_BASE_URL')
)

index_name = 'string_index'

def extract_text_from_pdf(filename, page_number=None, min_line_length=1):
    paragraphs = []
    buffer = ''
    full_text = ''

    for i, page_layout in enumerate(extract_pages(filename)):
        if page_number is not None and i not in page_number:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContaine):
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
    return paragraphs

def to_keywords(input_string):
    no_symbols = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string) 
    word_tokens = word_tokenize(no_symbols)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    filtered_sentence = [ps.stem(w)
                         for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def search(query_string, top_n=3):
    search_query = {
        'match': {
            'keywords': to_keywords(query_string)
        }
    }
    res = es.search(index=index_name, query=search_query, size=top_n)
    return [hit['_source']['text'] for hit in res['hit']['hit']]

def build_prompt(prompt_template, **kwargs):
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
    
    prompt = prompt.replace(f'__{k.upper}__', k)

def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

paragraphs = extract_text_from_pdf('data/wangbin/week2/llava.pdf', min_line_length=10)

if es.indices.exists(index = index_name):
    es.indices.delete(index=index_name)

es.indices.create(index=index_name)

actions = [
    {
        '_index': index_name,
        '_souce':{
            'keywords': to_keywords(para),
            'text': para
        }
    }
    for para in paragraphs
]

helpers.bulk(es, actions)

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

user_query = 'what is the llava model?'

search_results = search(user_query, 2)

prompt = build_prompt(prompt_template, info=search_results, query=user_query)
print('===Prompt===')
print(prompt)

response = get_completion(prompt)

print('===回答===')
print(response)
