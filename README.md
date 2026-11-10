LangChain e FastAPI - Parte 2: Questão 3 (Tradução EN-DE com Hugging Face e LangChain)

Aplicação avançada integrando o framework LangChain (via classe de LLM customizada para compatibilidade com o ecossistema local) com o modelo Helsinki-NLP/opus-mt-en-de da Hugging Face para tradução de inglês para alemão.
Tecnologias Utilizadas

    Python / FastAPI / Pydantic

    LangChain Core e LangChain Custom LLM Wrapper

    Hugging Face Transformers (MarianMTModel, MarianTokenizer)

    PyTorch / Sacremoses

Como Executar

    Ative o ambiente virtual executando no PowerShell:
    .\venv\Scripts\Activate

    Instale as dependências do projeto executando:
    pip install fastapi uvicorn langchain langchain-core transformers torch sentencepiece sacremoses pydantic

    Execute o aplicativo no Uvicorn executando:
    uvicorn tradutor_hf_langchain_api:app --reload

    Acesse a interface Swagger UI para testes da rota /traduzir-alemaum em:
