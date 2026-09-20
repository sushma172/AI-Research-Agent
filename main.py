# from src.pipelines.pipeline import run_research_pipeline


# topic = "The impact of AI on the job market in 2026"
# run_research_pipeline(topic)


from src.tools.tools import web_search,scrape_url
query = "future of LLM in tech industry"

print("\n========== DIRECT TAVILY TEST ==========\n")

output=web_search.invoke({"query": query})
print(output)

print("*"*50)
res=scrape_url.invoke(" https://www.geeksforgeeks.org/machine-learning/future-of-large-language-models")
print(res)