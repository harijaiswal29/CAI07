import sys
import os

# This will prevent the problematic import behavior
sys.modules['torch.classes'] = type('', (), {})
sys.modules['torch.classes.__path__'] = type('', (), {})
