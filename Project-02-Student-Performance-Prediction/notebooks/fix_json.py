"""Fix JSON syntax error in eda.ipynb"""
import json

path = "d:/New Project/CodeVedex Projects/CODEVEDX/Project-02-Student-Performance-Prediction/notebooks/eda.ipynb"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# The issue is extra escaping: \\\"\\\"\\\") should be \"\"\")
# In the raw JSON, we have: \\"\\"\\")
# Which after JSON parsing would be: \"\"\")
# But the value is already a string in JSON, so we need: \"\"\")
# Let's fix: replace the broken pattern
# Looking at the content, we see: \\\"\\\"\\\")\\n
# This should be: \"\"\")\\n
content = content.replace('\\\\\\"\\\\\\"\\\\\\")\\n', '\\"\\"\\")\\n')

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

# Validate
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"JSON is valid! {len(data['cells'])} cells found.")

