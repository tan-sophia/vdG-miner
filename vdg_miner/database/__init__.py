import os
import sys

module = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, module)