import os

urls = [
    "https://blog.dsacademy.com.br/firecrawl-e-web-scraping-inteligente-com-ia/",
    "https://blog.dsacademy.com.br/python-para-dados-o-guia-definitivo/",
    "https://blog.dsacademy.com.br/vagas-em-tech-2024-o-que-esperar/",
    "https://example.com/",
    "https://httpbin.org/html"
]

target_file = r"c:\Users\Ti\Desktop\process-cloner\missoes\stress_test_10k.txt"
os.makedirs(os.path.dirname(target_file), exist_ok=True)

with open(target_file, "w", encoding="utf-8") as f:
    for i in range(2000): # 2000 x 5 = 10,000 URLs
        for url in urls:
            f.write(f"{url}?stress_test_id={i}\n")

print(f"Gerado 10000 URLs em {target_file}")
