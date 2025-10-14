# find_t_type_references.py
import os
import re

def find_t_type_references():
    templates_dir = 'app/templates'
    agents_dir = 'agents'
    routes_dir = 'app/routes'
    
    print("🔍 Searching for remaining t_type references...")
    
    # Search templates
    for root, dirs, files in os.walk(templates_dir):
        for file in files:
            if file.endswith('.html'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 't_type' in content:
                        print(f"📄 Template: {filepath}")
                        # Show context
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 't_type' in line:
                                print(f"   Line {i+1}: {line.strip()}")
    
    # Search Python files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                # Skip virtual environment
                if '.venv' in filepath or '__pycache__' in filepath:
                    continue
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 't_type' in content:
                        print(f"🐍 Python: {filepath}")
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 't_type' in line and not line.strip().startswith('#'):
                                print(f"   Line {i+1}: {line.strip()}")

if __name__ == '__main__':
    find_t_type_references()