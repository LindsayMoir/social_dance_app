from pathlib import Path
from datetime import datetime
import sys, types
# Stub heavy deps so we can import images module-level functions without DB/browser
sys.modules['db'] = types.ModuleType('db')
sys.modules['llm'] = types.ModuleType('llm')
setattr(sys.modules['db'], 'DatabaseHandler', type('DatabaseHandler', (), {}) )
setattr(sys.modules['llm'], 'LLMHandler', type('LLMHandler', (), {}) )
sys.path.insert(0, 'src')
from images import detect_date_from_image


def test_detect_date_from_debug_poster():
    img_path = Path('debug/debug.png')
    assert img_path.exists(), 'debug/debug.png not found'
    d, dow = detect_date_from_image(img_path)
    assert d is not None and dow is not None
    parsed = datetime.strptime(d, "%Y-%m-%d")
    assert parsed.strftime("%A").lower() == dow.lower()
