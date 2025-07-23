import os
import torch
from khaosz import Khaosz, TextSplitter, Retriever


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "params")
    
    model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)
    spliter = TextSplitter(model.encode)
    retriever = Retriever()
    text = open("_base.txt").read()
    
    res = spliter.chunk(text, threshold=0.8, window_size=2)
    print(("\n" + "+"*100 + "\n").join(res))

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
        temperature=0.6,
        top_k=30,
        top_p=0.95,
    )

    print("retrive content:")
    print("\n".join([f"{idx + 1}. " + text for idx, (text, _) in enumerate(retrieved)]))

    print("\n\nretrive generate:")
    print(retrive_response)