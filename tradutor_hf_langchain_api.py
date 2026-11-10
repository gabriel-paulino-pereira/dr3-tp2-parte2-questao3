"""
Módulo de API FastAPI para tradução automática de Inglês para Alemão 
utilizando LangChain integrado ao modelo Hugging Face por meio de um 
wrapper customizado compatível com Transformers v5.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import MarianMTModel, MarianTokenizer

# Inicializo a aplicação FastAPI definindo metadados descritivos
app = FastAPI(
    title="API de Tradução EN-DE com LangChain e Hugging Face",
    description="Serviço de tradução de inglês para alemão utilizando wrapper customizado LangChain.",
    version="1.0.5"
)

# Defino o identificador do modelo da Hugging Face
MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"

# Crio uma classe customizada de LLM para o LangChain baseada nos modelos MarianMT
class MarianMTLLM(LLM):
    model_name: str = MODEL_NAME
    model: Any = Field(default=None, exclude=True)
    tokenizer: Any = Field(default=None, exclude=True)

    def __init__(self, model_name: str = MODEL_NAME, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    @property
    def _llm_type(self) -> str:
        return "marian_mt_custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        # Tokenizo o texto de entrada gerando os tensores compatíveis
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Executo a inferência do modelo para gerar os tokens traduzidos
        translated_tokens = self.model.generate(**inputs)
        
        # Decodifico os tokens gerados em string legível
        resultado = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return resultado

# Instancio o wrapper customizado do LLM
try:
    llm_hf = MarianMTLLM(model_name=MODEL_NAME)
except Exception as erro:
    raise RuntimeError(f"Erro ao inicializar o modelo MarianMT customizado: {str(erro)}")

# Construo o template de prompt adequado para a tarefa de tradução
prompt_template = PromptTemplate.from_template(
    "Translate the following English text to German: {texto_ingles}"
)

# Encadeio os componentes utilizando o operador pipe (|) do LangChain (LCEL)
cadeia_traducao_de = prompt_template | llm_hf | StrOutputParser()

# Defino o esquema de dados de entrada via Pydantic para validação do JSON
class RequisicaoTraducaoDE(BaseModel):
    texto: str

@app.post("/traduzir-alemaum", summary="Traduz texto do inglês para o alemão via LangChain e Hugging Face")
def traduzir_para_alemaum(dados: RequisicaoTraducaoDE):
    """
    Recebo um texto em inglês via JSON, executo a cadeia do LangChain 
    com o modelo customizado e retorno o texto traduzido em alemão estruturado em JSON.
    """
    try:
        # Valido se o texto de entrada não está vazio
        if not dados.texto.strip():
            raise HTTPException(status_code=400, detail="O texto de entrada não pode estar vazio.")
        
        # Executo a cadeia do LangChain passando o input validado
        resultado_traducao = cadeia_traducao_de.invoke({"texto_ingles": dados.texto})
        
        # Retorno o resultado estruturado para o cliente HTTP
        return {
            "texto_original_en": dados.texto,
            "texto_traduzido_de": resultado_traducao
        }
        
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento da tradução: {str(erro)}")

