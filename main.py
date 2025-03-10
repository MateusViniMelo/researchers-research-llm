import os

from scholarly import scholarly
from transformers import pipeline

# Configurar a API da Hugging Face
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError(
        "❌ ERRO: A variável de ambiente HUGGINGFACEHUB_API_TOKEN não está definida ou está vazia.")


def search_researchers(topic):
    search_query = scholarly.search_author(topic)
    researchers = []

    for author in search_query:
        try:
            author_info = scholarly.fill(author, sections=["basic"])
            scholar_id = author_info.get("scholar_id", "N/A")
            citations_link = f"https://scholar.google.com/citations?user={scholar_id}" if scholar_id != "N/A" else "N/A"

            researcher_data = {
                "Nome": author_info.get("name", "N/A"),
                "Afiliação": author_info.get("affiliation", "N/A"),
                "Link de Citações": citations_link
            }
            researchers.append(researcher_data)
            if len(researchers) >= 10:
                break
        except Exception as e:
            print(f"⚠️ Erro ao processar autor: {e}")

    return researchers


def generate_response_with_huggingface(researchers):
    """Utiliza a API da Hugging Face para formatar a resposta."""
    model = "facebook/opt-1.3b"  # Modelo exemplo, pode ser ajustado
    generator = pipeline("text-generation", model=model,
                         token=HUGGINGFACEHUB_API_TOKEN)

    context = "\n".join(
        [f"- Nome: {r['Nome']}\n  Afiliação: {r['Afiliação']}\n  Link de Citações: {r['Link de Citações']}\n" for r in researchers]
    )
    prompt = f"Forneça as informações dos seguintes pesquisadores em formato de lista:\n{context}"

    response = generator(prompt, max_length=500,
                         do_sample=True, truncation=True)
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
