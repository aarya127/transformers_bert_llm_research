import os

# Simple loader that reads api_keys.txt and sets environment variables for keys present.
# Lines should be KEY=VALUE. Lines starting with # are ignored.

def load_keys(path=None):
    path = path or os.path.join(os.path.dirname(__file__), 'api_keys.txt')
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                os.environ.setdefault(k, v)
