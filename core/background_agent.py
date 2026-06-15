import os
import time
from memory_graph_module import MemoryGraph

class BackgroundAgent:
    def __init__(self):
        self.memory_graph = MemoryGraph()

    def write_file(self, filename, content):
        with open(filename, 'w') as f:
            f.write(content)

    def run(self):
        while True:
            # Check for new files to write
            if os.path.exists('new_file.txt'):
                with open('new_file.txt', 'r') as f:
                    filename, content = f.read().split('\n')
                self.write_file(filename, content)
                os.remove('new_file.txt')
            time.sleep(1)
