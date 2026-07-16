from typing import Literal, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from ..config import Settings

TRIAGE_PROMPT = (
    "Você é um classificador de perguntas para um assistente de IA do Instituto Tadao Takahashi (ITT). "
    "Sua função é decidir se uma pergunta pode ser respondida buscando informações no estatuto do ITT ou se ela é incompleta. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO",\n'
    '  "campos_faltantes": ["..."]\n'
    "}}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas.\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações.\n'
    "Analise a mensagem e decida a ação mais apropriada.\n\n"
    "IMPORTANTE: Responda APENAS com o JSON, sem texto adicional."
)


class TriageOutput(BaseModel):
    """Resultado da triagem de uma pergunta."""
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"] = Field(
        description="Decisão sobre como processar a pergunta"
    )
    campos_faltantes: Optional[List[str]] = Field(
        default=None,
        description="Lista de campos ou informações que estão faltando na pergunta"
    )


def get_triage_chain(settings: Settings):
    """
    Cria chain de triagem usando JsonOutputParser em vez de with_structured_output.
    Isso evita problemas de conversão de schema do Pydantic v1 para Gemini.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=3,
        timeout=30,
    )

    parser = JsonOutputParser(pydantic_object=TriageOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", TRIAGE_PROMPT),
        ("human", "{input}")
    ])

    chain = prompt | llm | parser

    def wrapped_chain(messages):
        user_message = None
        for msg in messages:
            if hasattr(msg, 'content') and msg.type == 'human':
                user_message = msg.content
                break

        if not user_message:
            user_message = str(messages[-1]) if messages else ""

        result = chain.invoke({"input": user_message})

        if result.get("campos_faltantes") is None:
            result["campos_faltantes"] = []

        return TriageOutput(**result)

    class TriageChain:
        def invoke(self, messages):
            return wrapped_chain(messages)

    return TriageChain()

DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content", "source"],
    template="[Fonte: {source}]\n{page_content}"
)

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente especialista sobre o Instituto Tadao Takahashi (ITT). "
     "Sua missão é responder às perguntas do usuário de forma clara, completa e útil. "
     "Use o estatuto do ITT, fornecido no 'Contexto', como sua principal fonte de verdade, seguindo estas regras rigorosamente:\n\n"
     "--- PRINCÍPIO DE RACIOCÍNIO ---\n\n"
     "1. **Analise o Tipo de Pergunta:**\n"
     "   - **a) Perguntas Específicas:** Se a pergunta for sobre regras, políticas, finanças, procedimentos ou detalhes operacionais (geralmente com 'como', 'qual', 'pode', 'deve'), você deve se basear **ESTRITAMENTE** no contexto fornecido.\n"
     "   - **b) Perguntas Gerais:** Se a pergunta for introdutória ('o que é o ITT?', 'qual o objetivo do instituto?'), o contexto pode ser insuficiente. Neste caso, **VOCÊ PODE** usar seu conhecimento geral para dar uma resposta informativa, mas sempre priorize e integre qualquer informação relevante que encontrar no contexto.\n\n"
     "--- REGRAS DE EXECUÇÃO DA RESPOSTA ---\n\n"
     "2. **Sintetize a Informação:** Combine informações de diferentes partes do contexto para construir uma resposta coesa e completa. Não se limite a extrair trechos isolados.\n\n"
     "3. **Seja Direto e Formate Bem:** Responda diretamente à pergunta do usuário. Use listas (bullet points) para organizar informações complexas e facilitar a leitura.\n\n"
     "4. **Quando não souber:** Se a resposta para uma pergunta específica (tipo 1a) não puder ser encontrada no contexto, responda de forma educada: 'Com base no estatuto do ITT que tenho acesso, não encontrei uma resposta para sua pergunta.'\n\n"
     "5. **Cite as fontes:** Ao final de cada parágrafo onde você usou informação de uma fonte, escreva em itálico a origem, usando apenas o nome do arquivo sem o caminho completo. "
     "Formato obrigatório: *fonte: nome_do_arquivo*. Exemplo: *fonte: estatuto_itt.pdf*. "
     "Se o parágrafo usar informações de múltiplas fontes, liste todas separadas por vírgula. "
     "Não cite fontes que você não utilizou naquele parágrafo."),
    ("human", "Pergunta do Usuário: {input}\n\nContexto do Estatuto do ITT:\n{context}")
])


def get_rag_chain(settings: Settings):
    """
    Creates RAG chain for question answering.
    How it works:
    1. Graph passes {"input": question, "context": docs} to chain
    2. ChatPromptTemplate formats the prompt with input/context
    3. LLM generates text response (NOT JSON, just plain text)
    4. create_stuff_documents_chain combines docs and returns string
    5. Graph receives plain string response

    Each document is formatted via DOCUMENT_PROMPT (includes source filename)
    so the LLM can cite *fonte: filename* at the end of each paragraph.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        google_api_key=settings.GOOGLE_API_KEY,
        max_retries=3,
        timeout=30,
    )
    return create_stuff_documents_chain(llm, RAG_PROMPT_TEMPLATE, document_prompt=DOCUMENT_PROMPT)


# from typing import Literal, List
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain

# from ..config import Settings

# TRIAGE_PROMPT = (
#     "Você é um classificador de perguntas para um assistente de IA do Instituto Tadao Takahashi (ITT). "
#     "Sua função é decidir se uma pergunta pode ser respondida buscando informações no estatuto do ITT ou se ela é incompleta. "
#     "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
#     "{\n"
#     '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO",\n'
#     '  "campos_faltantes": ["..."]\n'
#     "}\n"
#     "Regras:\n"
#     '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas.\n'
#     '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações.\n'
#     "Analise a mensagem e decida a ação mais apropriada."
# )

# class TriageOutput(BaseModel):
#     decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"] = Field(
#         description="Decisão sobre como processar a pergunta"
#     )
#     campos_faltantes: List[str] = Field(
#         default_factory=list,
#         description="Lista de campos faltantes"
#     )

#     class Config:
#         schema_extra = {
#             "properties": {
#                 "campos_faltantes": {
#                     "type": "array",
#                     "items": {"type": "string"}
#                 }
#             }
#         }


# def get_triage_chain(settings: Settings):

#     llm = ChatGoogleGenerativeAI(
#         model=settings.LLM_MODEL,
#         temperature=settings.LLM_TEMPERATURE,
#         google_api_key=settings.GOOGLE_API_KEY
#     )
#     return llm.with_structured_output(TriageOutput)

# RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
#     ("system",
#      "Você é um assistente especialista sobre o Instituto Tadao Takahashi (ITT). "
#      "Sua missão é responder às perguntas do usuário de forma clara, completa e útil. "
#      "Use o estatuto do ITT, fornecido no 'Contexto', como sua principal fonte de verdade, seguindo estas regras rigorosamente:\n\n"
#      "--- PRINCÍPIO DE RACIOCÍNIO ---\n\n"
#      "1. **Analise o Tipo de Pergunta:**\n"
#      "   - **a) Perguntas Específicas:** Se a pergunta for sobre regras, políticas, finanças, procedimentos ou detalhes operacionais (geralmente com 'como', 'qual', 'pode', 'deve'), você deve se basear **ESTRITAMENTE** no contexto fornecido.\n"
#      "   - **b) Perguntas Gerais:** Se a pergunta for introdutória ('o que é o ITT?', 'qual o objetivo do instituto?'), o contexto pode ser insuficiente. Neste caso, **VOCÊ PODE** usar seu conhecimento geral para dar uma resposta informativa, mas sempre priorize e integre qualquer informação relevante que encontrar no contexto.\n\n"
#      "--- REGRAS DE EXECUÇÃO DA RESPOSTA ---\n\n"
#      "2. **Sintetize a Informação:** Combine informações de diferentes partes do contexto para construir uma resposta coesa e completa. Não se limite a extrair trechos isolados.\n\n"
#      "3. **Seja Direto e Formate Bem:** Responda diretamente à pergunta do usuário. Use listas (bullet points) para organizar informações complexas e facilitar a leitura.\n\n"
#      "4. **Quando não souber:** Se a resposta para uma pergunta específica (tipo 1a) não puder ser encontrada no contexto, responda de forma educada: 'Com base no estatuto do ITT que tenho acesso, não encontrei uma resposta para sua pergunta.'"),
#     ("human", "Pergunta do Usuário: {input}\n\nContexto do Estatuto do ITT:\n{context}")
# ])

# def get_rag_chain(settings: Settings):
#     """
#     Creates RAG chain for question answering.
    
#     How it works:
#     1. Graph passes {"input": question, "context": docs} to chain
#     2. ChatPromptTemplate formats the prompt with input/context
#     3. LLM generates text response (NOT JSON, just plain text)
#     4. create_stuff_documents_chain combines docs and returns string
#     5. Graph receives plain string response
#     """
#     llm = ChatGoogleGenerativeAI(
#         model=settings.LLM_MODEL,
#         temperature=settings.LLM_TEMPERATURE,
#         google_api_key=settings.GOOGLE_API_KEY
#     )
#     return create_stuff_documents_chain(llm, RAG_PROMPT_TEMPLATE)
