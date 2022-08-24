import jsonlines
import json
import os    
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
           if os.path.splitext(file)[1] == ".jsonl" and "iter" not in os.path.splitext(file)[0] and "tapex" not in os.path.splitext(file)[0] and 'table' in os.path.splitext(file)[0]:  
                L.append(os.path.join(root, file))  
    return L  


attributes=["id","object_type","position_x","position_y","position_z","rotation_x","rotation_y","rotation_z","parent_receptacle"]

def convert(item):
    input_table={attribute:[] for attribute in attributes}
    id=item["id"]
    input_split=item["input"].split("SCENE DESCRIPTION")
    try:
        scene_info=input_split[1]
    except:
        try:
            scene_info=item["input"].split(":")[1]
        except:return False
    k=scene_info.split("[SEP]")
    t=True
    for object in k:
        if t:
            t=False
            continue 
        atts=object.split(',')
        for att,attribute in zip(atts,attributes):
            input_table[attribute].append(att.strip())
    dic={"id":id,"input":input_split[0],"output":item["output"][0],"input_table":input_table}
    return dic


def convert_file(path):
    with jsonlines.open(f"{path}", "r") as f:
        with jsonlines.open(f'{path.replace(".jsonl","_tapex.jsonl")}', mode="w") as writer:
            for item in f:
                if convert(item)==False:continue
                writer.write(convert(item))
    


L=file_name("../")
print(L)
for i in L:
    print(i)
    convert_file(i)