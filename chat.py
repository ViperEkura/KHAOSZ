from module import *

def chat():
    model = Khaosz("params")
    response_size = 0
    histroy = []

    while True:
        query = input(">> ")
        if query == "!exit":
            break
        
        for response, histroy in model.stream_generate(
            query=query, 
            history=histroy,
            temperature=0.9,    
        ):
            print(response[response_size:], end="")
            response_size = len(response)
            
        print("")

def test():
    cfg = Config("params/config.json")
    model = Transfomer(cfg)
    print(model)
    print(f"parameter size: {model.parameter_size():,}")

if __name__ == "__main__":
    test()