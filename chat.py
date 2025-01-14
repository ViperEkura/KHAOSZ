from module.generate import *

if __name__ == "__main__":
    model = Khaosz("params")
    response_size = 0
    histroy = []

    while True:
        query = input(">> ")
        if query == "!exit":
            break
        
        for querry, response, histroy in model.stream_generate(
            query=query, 
            history=histroy,
            temperature=0.9,    
        ):
            print(response[response_size:], end="")
            response_size = len(response)
            
        print("")
