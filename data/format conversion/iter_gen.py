import jsonlines

import os    
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
           if os.path.splitext(file)[1] == '.jsonl' and "iter" not in os.path.splitext(file)[0] and "tapex" not in os.path.splitext(file)[0]:  
                L.append(os.path.join(root, file))  
    return L  

def iter(path):
    with jsonlines.open(f"{path.replace('.jsonl','_iter.jsonl')}", mode='w') as writer:
        with jsonlines.open(f"{path}", "r") as f:
            for item in f:
                output=item['output'][0].split(' | ')
                for i in range(len(output)):
                    if "SCENE DESCRIPTION" not in item['input']:
                        id=item['id']+f'%{i}'
                        input=[]
                        input.append(item['input'])
                        for _ in range(i):
                            input.append(output[_])
                        input=" | ".join(input)
                        z=[output[i]]
                        dic={'id':id,"input":input,"output":z}
                        writer.write(dic)
                    else :
                        id=item['id']+f'%{i}'
                        input_split=item['input'].split("SCENE DESCRIPTION")
                        scene_info=input_split[1]
                        input=[]
                        input.append(input_split[0])
                        for _ in range(i):
                            input.append(output[_])
                        input=" | ".join(input)
                        z=[output[i]]
                        dic={'id':id,"input":input+"SCENE DESCRIPTION"+scene_info,"output":z}
                        writer.write(dic)
    
L=file_name('../')
for i in L:
    print(i)
    iter(i)