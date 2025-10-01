import re
from typing import Optional





def extract_pattern(content: str, pattern: str) -> Optional[str]:
    try:
        _pattern = rf"<{pattern}>(.*?)</{pattern}>"
        match = re.search(_pattern, content, re.DOTALL)
        if match:
            text = match.group(1)
            return text.strip()
        else:
            return None
    except Exception as e:
        print(f"Error extracting pattern {pattern}: {e}")
        return None