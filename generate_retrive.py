import os
import torch
from module import Khaosz

script_dir = os.path.dirname(__file__)
model_dir = os.path.join(script_dir, "params")
model = Khaosz(model_dir).to(device='cuda', dtype=torch.bfloat16)

if __name__ == "__main__":
    text = open("test_base.txt").read()
    res = model.chunk(text, threshold=0.8, window_size=2)
    # print(("\n" + "+"*100 + "\n").join(res))

    res_embs = [model.sentence_embedding(text) for text in res]
    for sentence, emb in zip(res, res_embs):
        model.retriever.add_vector(sentence, emb)

    query = "作者设计了一个怎样的模型"
    retrive_content = model.retrieve_generate(
        query=query,
        retrive_top_k=5,
        temperature=0.6,
        top_k=30,
        top_p=0.95,
    )

    print("retrive content:")
    vec = model.sentence_embedding(query)
    asw = model.retriever.retrieve(query=vec, top_k=5)
    print("\n".join([f"{idx + 1}. " + text for idx, (text, _) in enumerate(asw)]))

    print("\n\nretrive generate:")
    print(retrive_content)