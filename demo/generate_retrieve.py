import os
import torch
from khaosz import Khaosz, SemanticTextSplitter, Retriever


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    model_dir = os.path.join(PROJECT_ROOT, "params")
    context_path = os.path.join(PROJECT_ROOT, "README.md")
    
    model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
    spliter = SemanticTextSplitter(model.encode)
    retriever = Retriever()
    text = open(context_path, "r", encoding="utf-8").read()
    
    res = spliter.split(text, threshold=0.8, window_size=1)
    # print(("\n" + "+"*100 + "\n").join(res))

    res_embs = model.encode(res)
    for sentence, emb in zip(res, res_embs):
        retriever.add_vector(sentence, emb)

    retrive_top_k = 5
    query = "作者设计了一个怎样的模型"
    emb_query = model.encode(query)
    retrieved = retriever.retrieve(emb_query, retrive_top_k)
    
    retrive_response = model.retrieve_generate(
        retrieved=retrieved,
        query=query,
        temperature=0.8,
        top_p=0.95,
        top_k=50
    )

    print("retrieve content:")
    print("\n".join([f"{idx + 1}. " + text for idx, (text, _) in enumerate(retrieved)]))

    print("\n\nretrive generate:")
    print(retrive_response)