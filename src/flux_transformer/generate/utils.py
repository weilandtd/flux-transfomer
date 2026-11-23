import ast 

def read_subnet(filename):
    with open(f"{filename}") as f :
        data = f.read()
        data = data.strip()
        subnetworks = ast.literal_eval(data)
    return subnetworks

