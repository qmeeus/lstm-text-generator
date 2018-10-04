

def load(config):
    with open(config.save_path('data'), 'r', encoding=config.encoding) as f:
        data = f.read().lower()
    return data
