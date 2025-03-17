import os
import time

from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from scholarly import scholarly
from transformers import pipeline

load_dotenv()
# Configurar a API da Hugging Face
HUGGINGFACEHUB_API_TOKEN = os.getenv(
    "HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError(
        "❌ ERRO: A variável de ambiente HUGGINGFACEHUB_API_TOKEN não está definida ou está vazia."
    )


def translate_to_english(text):
    """Traduz um termo do português para o inglês automaticamente."""
    return GoogleTranslator(source="auto", target="en").translate(text)


def search_researchers(topic, max_results=10):
    """
    Busca pesquisadores no Google Scholar com base em um tema específico.

    :param topic: Tema de interesse
    :param max_results: Número máximo de pesquisadores a retornar
    :return: Lista de dicionários com nome, afiliação e link de citações
    """
    # Traduzir o termo para inglês antes da busca
    translated_topic = translate_to_english(topic)

    search_query = scholarly.search_author(translated_topic)
    researchers = []

    for i, author in enumerate(search_query):
        if i >= max_results:
            break

        try:
            author_info = scholarly.fill(author, sections=["basic"])
            scholar_id = author_info.get("scholar_id", "N/A")
            citations_link = (
                f"https://scholar.google.com/citations?user={scholar_id}"
                if scholar_id != "N/A"
                else "N/A"
            )

            researcher_data = {
                "Nome": author_info.get("name", "N/A"),
                "Afiliação": author_info.get("affiliation", "N/A"),
                "Link de Citações": citations_link,
            }
            researchers.append(researcher_data)

            # Delay para evitar bloqueio do Google Scholar
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ Erro ao processar autor: {e}")

    return researchers


def generate_response_with_huggingface(researchers):
    """
    Utiliza a API da Hugging Face para formatar uma resposta com os pesquisadores encontrados.

    :param researchers: Lista de dicionários com informações dos pesquisadores
    :return: Texto gerado pelo modelo
    """
    model = "gpt2-medium"  # Modelo menor e mais leve que o OPT-1.3B
    generator = pipeline("text-generation", model=model)

    context = "\n".join(
        [
            f"- Nome: {r['Nome']}\n  Afiliação: {r['Afiliação']}\n  Link de Citações: {r['Link de Citações']}\n"
            for r in researchers
        ]
    )

    prompt = f"Forneça as informações dos seguintes pesquisadores de forma organizada:\n{context}"

    response = generator(prompt, max_new_tokens=100,
                         do_sample=True, truncation=True, )

    return response[0]["generated_text"]


if __name__ == "__main__":
    topic = input("🔎 Digite o tema da pesquisa: ")
    researchers = search_researchers(topic)

    if researchers:
        print("📚 Pesquisadores encontrados:")
        response = generate_response_with_huggingface(researchers)
        print(response)
    else:
        print("⚠️ Nenhum pesquisador encontrado.")
