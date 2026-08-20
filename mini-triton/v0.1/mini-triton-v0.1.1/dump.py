from pathlib import Path

def dump_file(name, text):
    Path(name).write_text(text)
