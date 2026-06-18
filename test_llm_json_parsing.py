import unittest
import json
from backend.core.llm_client import OrbitalLLMClient

class TestLLMJsonParsing(unittest.TestCase):
    def test_markdown_fence_and_trailing_prose(self):
        # Sample response with markdown fences, trailing prose, and '±' character
        malformed_response = """
Here is the JSON you requested:
```json
{
  "actor": "Bank",
  "action": "Ensure deviation is within ±10%",
  "severity": "medium"
}
```
Please let me know if you need any more extracted obligations.
"""
        result = OrbitalLLMClient._parse_json(malformed_response)
        self.assertIsNotNone(result)
        self.assertEqual(result["actor"], "Bank")
        self.assertEqual(result["action"], "Ensure deviation is within ±10%")
        self.assertEqual(result["severity"], "medium")

    def test_no_fences_but_prose(self):
        malformed_response = """
[
  {
    "id": "1",
    "trigger": "if condition ± occurs"
  }
]
Hope this helps!
"""
        result = OrbitalLLMClient._parse_json(malformed_response)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["id"], "1")
        self.assertEqual(result[0]["trigger"], "if condition ± occurs")

    def test_trailing_comma(self):
        malformed_response = """
{
  "actor": "Bank",
  "action": "Do something",
}
"""
        result = OrbitalLLMClient._parse_json(malformed_response)
        self.assertIsNotNone(result)
        self.assertEqual(result["actor"], "Bank")

if __name__ == "__main__":
    unittest.main()
