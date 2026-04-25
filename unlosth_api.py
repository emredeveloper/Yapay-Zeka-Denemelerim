import json
import re
import traceback
from typing import List, Dict, Any, Optional

import requests
from ddgs import DDGS
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

UNSLOTH_BASE_URL = "http://127.0.0.1:PORT/v1"
UNSLOTH_API_KEY = "sk-unsloth-....."
MODEL = "gemma-4-E2B-it-BF16.gguf"

MAX_SEARCH_RESULTS_PER_QUERY = 6
MAX_TOTAL_RESULTS_TO_MODEL = 12
HTTP_TIMEOUT = 15

client = OpenAI(
    base_url=UNSLOTH_BASE_URL,
    api_key=UNSLOTH_API_KEY,
)


# ============================================================
# MODEL CALL
# ============================================================

def call_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
):
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools is not None:
        kwargs["tools"] = tools

    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**kwargs)


# ============================================================
# OFFICIAL VERSION TOOLS
# ============================================================

def python_latest_version() -> str:
    """
    Python'ın en güncel kararlı sürümünü resmi python.org sayfasından çeker.
    """
    urls = [
        "https://www.python.org/",
        "https://www.python.org/downloads/",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            text = r.text

            patterns = [
                r"Latest:\s*Python\s*([0-9]+\.[0-9]+\.[0-9]+)",
                r"Download Python\s*([0-9]+\.[0-9]+\.[0-9]+)",
                r"Latest Python 3 Release\s*-\s*Python\s*([0-9]+\.[0-9]+\.[0-9]+)",
            ]

            for pattern in patterns:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m:
                    version = m.group(1)
                    return (
                        f"Python official latest stable version: {version}\n"
                        f"Source: {url}"
                    )

        except Exception as e:
            last_error = str(e)

    return f"Python resmi sürümü alınamadı. Hata: {last_error}"


def pypi_latest_version(package_name: str) -> str:
    """
    PyPI JSON API'den bir Python paketinin en güncel sürümünü çeker.
    Örnek package_name: transformers, torch, fastapi, unsloth
    """
    package_name = package_name.strip().lower()
    package_name = package_name.replace(" ", "-")

    url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        if r.status_code == 404:
            return (
                f"PyPI üzerinde '{package_name}' paketi bulunamadı.\n"
                f"Source: {url}"
            )

        r.raise_for_status()
        data = r.json()

        info = data.get("info", {})
        version = info.get("version")
        name = info.get("name", package_name)
        summary = info.get("summary", "")

        if not version:
            return f"PyPI cevabında sürüm bulunamadı. Source: {url}"

        return (
            f"PyPI package: {name}\n"
            f"Latest version: {version}\n"
            f"Summary: {summary}\n"
            f"Source: {url}"
        )

    except Exception as e:
        return f"PyPI sorgusu başarısız oldu. Hata: {e}\nSource: {url}"


# ============================================================
# WEB SEARCH
# ============================================================

def build_dynamic_queries(query: str) -> List[str]:
    q = query.strip()

    queries = [
        q,
        f"{q} official",
        f"{q} latest official",
        f"{q} release notes",
        f"{q} documentation",
    ]

    cleaned = []
    seen = set()

    for item in queries:
        item = " ".join(item.split())
        key = item.lower()

        if item and key not in seen:
            seen.add(key)
            cleaned.append(item)

    return cleaned


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS_PER_QUERY) -> str:
    """
    Genel web araması. Sürüm/paket soruları için önce PyPI/Python özel tool'ları kullanılmalı.
    """
    queries = build_dynamic_queries(query)
    all_results = []
    seen_urls = set()

    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    results = ddgs.text(
                        q,
                        region="wt-wt",
                        safesearch="moderate",
                        max_results=max_results,
                    )

                    for r in results:
                        title = r.get("title", "").strip()
                        snippet = r.get("body", "").strip()
                        url = r.get("href", "").strip()

                        if not url or url in seen_urls:
                            continue

                        seen_urls.add(url)

                        all_results.append(
                            {
                                "query": q,
                                "title": title,
                                "snippet": snippet,
                                "url": url,
                            }
                        )

                        if len(all_results) >= MAX_TOTAL_RESULTS_TO_MODEL:
                            break

                    if len(all_results) >= MAX_TOTAL_RESULTS_TO_MODEL:
                        break

                except Exception as e:
                    all_results.append(
                        {
                            "query": q,
                            "title": "Search query failed",
                            "snippet": str(e),
                            "url": "",
                        }
                    )

    except Exception as e:
        return f"Web araması yapılamadı. Hata: {e}"

    if not all_results:
        return "Web aramasında sonuç bulunamadı."

    blocks = []

    for i, r in enumerate(all_results, start=1):
        blocks.append(
            f"[{i}]\n"
            f"Arama sorgusu: {r['query']}\n"
            f"Başlık: {r['title']}\n"
            f"Özet: {r['snippet']}\n"
            f"URL: {r['url']}"
        )

    return "\n\n".join(blocks)


# ============================================================
# TOOLS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_latest_version",
            "description": (
                "Python programlama dilinin en güncel kararlı sürümünü resmi python.org "
                "kaynağından öğrenir. Kullanıcı Python'un güncel/en son/latest sürümünü sorarsa bunu kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pypi_latest_version",
            "description": (
                "PyPI üzerindeki bir Python paketinin en güncel sürümünü resmi PyPI JSON API ile öğrenir. "
                "Kullanıcı transformers, torch, fastapi, flask, django, numpy, pandas, unsloth gibi "
                "bir Python kütüphanesinin en güncel/latest sürümünü sorarsa bunu kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "package_name": {
                        "type": "string",
                        "description": "PyPI paket adı. Örnek: transformers, torch, fastapi, pandas, numpy, unsloth",
                    }
                },
                "required": ["package_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Genel web araması yapar. Haber, ürün bilgisi, teknik özellik, GitHub release notları, "
                "dokümantasyon veya web'de doğrulanması gereken konular için kullanılır. "
                "Paket sürümü sorularında önce pypi_latest_version kullanılmalıdır."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Web arama sorgusu. Örnek: "
                            "'NVIDIA RTX 5070 Ti official specifications VRAM', "
                            "'Unsloth v0.1.37 beta release notes GitHub'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ============================================================
# SIMPLE ROUTER FALLBACK
# ============================================================

def fallback_tool_choice(user_text: str) -> Optional[Dict[str, Any]]:
    """
    Model tool çağırmazsa ya da yanlış tool seçerse basit güvenlik ağı.
    Bu ana karar mekanizması değil; sadece çok bariz sürüm sorularını yakalar.
    """
    text = user_text.lower()

    version_words = [
        "en güncel",
        "güncel",
        "latest",
        "son sürüm",
        "sürümü",
        "version",
    ]

    if not any(w in text for w in version_words):
        return None

    if "python" in text and not any(pkg in text for pkg in ["transformers", "torch", "pandas", "numpy", "fastapi"]):
        return {
            "name": "python_latest_version",
            "args": {},
        }

    known_packages = [
        "transformers",
        "torch",
        "pytorch",
        "tensorflow",
        "numpy",
        "pandas",
        "fastapi",
        "flask",
        "django",
        "unsloth",
        "accelerate",
        "datasets",
        "tokenizers",
        "scikit-learn",
        "sklearn",
        "openai",
        "langchain",
        "gradio",
        "streamlit",
    ]

    for pkg in known_packages:
        if pkg in text:
            package_name = pkg
            if package_name == "pytorch":
                package_name = "torch"
            if package_name == "sklearn":
                package_name = "scikit-learn"

            return {
                "name": "pypi_latest_version",
                "args": {"package_name": package_name},
            }

    return None


def run_tool(name: str, args: Dict[str, Any]) -> str:
    if name == "python_latest_version":
        return python_latest_version()

    if name == "pypi_latest_version":
        return pypi_latest_version(args.get("package_name", ""))

    if name == "web_search":
        return web_search(args.get("query", ""))

    return f"Bilinmeyen tool: {name}"


# ============================================================
# ASK FUNCTION
# ============================================================

def ask(user_text: str) -> str:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Sen Türkçe konuşan yardımcı bir asistansın.\n\n"
                "Araç kullanma kuralları:\n"
                "- Python dilinin en güncel sürümü sorulursa python_latest_version kullan.\n"
                "- Python kütüphanesi/paketi sürümü sorulursa pypi_latest_version kullan.\n"
                "- Güncel haber, teknik özellik, ürün bilgisi, GitHub release veya genel araştırma için web_search kullan.\n"
                "- Sürüm sorularında DDG/web snippet'lerine güvenme; resmi registry/API tool'unu tercih et.\n\n"
                "Cevap kuralları:\n"
                "- Kaynaklarda olmayan bilgiyi uydurma.\n"
                "- Kaynaklar çelişirse bunu açıkça söyle.\n"
                "- Cevabı Türkçe, kısa ve net ver.\n"
                "- Tool sonucu kullandıysan kaynak URL'sini yaz.\n"
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    first_response = call_chat(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,
        max_tokens=700,
    )

    first_msg = first_response.choices[0].message
    messages.append(first_msg)

    used_tool = False

    if first_msg.tool_calls:
        for tool_call in first_msg.tool_calls:
            tool_name = tool_call.function.name

            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            print(f"\n[{tool_name}] {args}")

            tool_result = run_tool(tool_name, args)
            used_tool = True

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    else:
        fallback = fallback_tool_choice(user_text)

        if fallback:
            tool_name = fallback["name"]
            args = fallback["args"]

            print(f"\n[fallback:{tool_name}] {args}")

            tool_result = run_tool(tool_name, args)
            used_tool = True

            # Assistant tool_call yokken role=tool eklemek bazı serverlarda sorun çıkarabilir.
            # Bu yüzden tool sonucunu user mesajı olarak ekliyoruz.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Aşağıdaki resmi araç sonucuna göre cevap ver:\n\n"
                        f"Tool: {tool_name}\n"
                        f"Sonuç:\n{tool_result}"
                    ),
                }
            )

        else:
            content = first_msg.content or ""
            content = content.strip()
            if content:
                return content

            return "Model boş cevap döndürdü."

    if not used_tool:
        return first_msg.content or "Model cevap üretemedi."

    final_response = call_chat(
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
    )

    final_msg = final_response.choices[0].message
    final_content = final_msg.content or ""
    final_content = final_content.strip()

    if final_content:
        return final_content

    return "Model tool sonucundan sonra boş cevap döndürdü."


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check() -> bool:
    try:
        response = client.models.list()
        print("[OK] Local API çalışıyor.")
        print("[Models]")
        for m in response.data:
            print("-", m.id)
        return True

    except Exception as e:
        print("[ERROR] Local API bağlantısı kurulamadı.")
        print("Base URL:", UNSLOTH_BASE_URL)
        print("Hata:", e)
        print()
        print("Port değişmiş olabilir. Kontrol için:")
        print("netstat -ano | findstr llama")
        return False


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("Local Gemma + Tools + Web Search")
    print("Çıkmak için: q / quit / exit / çık / çıkış")
    print("-" * 60)

    if not health_check():
        return

    while True:
        user_text = input("\nSen: ").strip()

        if user_text.lower() in ["q", "quit", "exit", "çık", "çıkış"]:
            print("Çıkılıyor.")
            break

        if not user_text:
            continue

        try:
            answer = ask(user_text)

            print("\nModel:")
            print(answer)

        except KeyboardInterrupt:
            print("\nİşlem iptal edildi.")

        except Exception:
            print("\n[HATA]")
            traceback.print_exc()


if __name__ == "__main__":
    main()