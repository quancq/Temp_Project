

def load_str(path):
    data = ""
    with open(path, 'r') as f:
        data = f.read().strip()
    return data
