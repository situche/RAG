# RAG(Retrieval-Augmented Generation)：检索增强生成

该项目实现了一个基于 FAISS 向量数据库的检索增强生成（RAG）问答系统。系统可以从 PDF 文档中提取信息，使用 OpenAI API 和 CrossEncoder 模型进行文档匹配和生成问答回答。

### 主要功能

1. 使用 FAISS 向量数据库存储和检索文档。
2. 从 PDF 中提取文本，并将其分割为可处理的小段落。
3. 使用 CrossEncoder 对文档与用户查询进行匹配评分。
4. 构建动态提示（Prompt）并使用 OpenAI API 获取响应。

## 安装指南

### 克隆仓库

首先，克隆本仓库到本地：

'''git clone https://github.com/你的用户名/项目名.git
cd 项目名'''

### 安装依赖

建议使用 virtualenv 或 conda 环境来安装依赖，确保依赖不会影响其他项目。

使用 pip 安装所需的 Python 库：

'''
pip install -r requirements.txt
'''

requirements.txt 文件内容：

'''
faiss-cpu
nltk
numpy
openai
sentence-transformers
pdfminer.six
'''

### API设置
为了使用 OpenAI API，你需要在环境变量中设置你的 API 密钥：

'''
export OPENAI_API_KEY="你的API密钥"
'''

你也可以将其写入 .env 文件，并使用 python-dotenv 来加载。

## 依赖项

faiss: 用于向量数据库索引和检索。
nltk: 用于自然语言处理，文本切分。
openai: 用于访问 OpenAI API。
sentence-transformers: 用于获取句子级别的嵌入表示。
pdfminer.six: 用于从 PDF 中提取文本。

## 使用方法

1. **准备文档**：将你的 PDF 文档上传到项目文件夹。
2. **运行程序**：可以通过以下命令运行程序，开始提问。

```bash
python run_bot.py
```

3. **交互式问答**：程序会根据给定的 PDF 文档生成问答机器人，你可以与其进行交互并提问。

### 示例

假设你有一个 PDF 文件 `llava.pdf`，其中包含有关 `miniCPM` 模型的文档。运行程序后，你可以向问答系统提问：

```bash
请输入您的问题：miniCPM模型有多少个参数？
```

系统会从 PDF 文档中提取相关信息并生成回答。

## 功能说明

### `FaissVectorDBConnector`

这是一个使用 FAISS 向量数据库的类，用于处理文档的向量化表示和相似性搜索。该类提供了以下功能：
- **添加文档**：将文档嵌入到 FAISS 向量数据库中。
- **搜索**：根据查询进行相似性搜索，返回最相关的文档。

### `RAG_Bot`

这是一个基于 RAG 模型的问答机器人，结合了检索（使用向量数据库）和生成（使用 OpenAI 模型）的功能。它提供了以下功能：
- **查询处理**：接收用户的查询并通过向量数据库检索相关文档。
- **回答生成**：将检索到的文档与用户查询结合，生成回答。

### `process_pdf_and_split`

这是一个用于处理 PDF 文件并将其分割为小段落的函数。它还将长段落拆分为更小的句子块，以便于后续处理。

### `get_embeddings`

这是一个用于获取文本嵌入的函数，使用 OpenAI 的 `text-embedding-ada-002` 模型来生成文本的向量表示。

### `build_prompt`

用于根据提供的模板生成用户输入的提示。

## API 文档

### `FaissVectorDBConnector.add_documents(documents, metadata={})`

将文档添加到向量数据库中。

#### 参数：
- `documents` (list): 要添加的文档。
- `metadata` (dict): 每个文档的元数据（可选）。

### `FaissVectorDBConnector.search(query, top_n)`

根据查询进行搜索并返回相关文档。

#### 参数：
- `query` (str): 用户查询。
- `top_n` (int): 返回的最相关文档数量。

#### 返回：
- 一个字典，包含文档、距离和元数据。

### `RAG_Bot.chat(user_query, model, prompt_template)`

根据用户查询生成回答。

#### 参数：
- `user_query` (str): 用户的查询问题。
- `model` (CrossEncoder): 使用的模型。
- `prompt_template` (str): 用于生成提示的模板。

#### 返回：
- 生成的回答。

## 示例

```python
# 示例代码
vector_db = FaissVectorDBConnector(get_embeddings, 1536)
vector_db.add_documents(paragraphs)

user_query = 'miniCPM模型有多少个参数？'
bot = RAG_Bot(vector_db, llm_api=get_embeddings)
response = bot.chat(user_query, model, prompt_template)
print(response)
```

## 许可证

本项目使用 MIT 许可证，详情请参见 [LICENSE](./LICENSE) 文件。
