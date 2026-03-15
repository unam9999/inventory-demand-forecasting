import os

def generate_final_output():
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(project_root, 'FINAL_OUTPUT.md')
    report_file = os.path.join(project_root, 'reports', 'project_report.md')
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 1. Write Report
        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")
        
        outfile.write("# FILE CONTENTS\n\n")
        
        # 2. Write all other files
        for root, dirs, files in os.walk(project_root):
            # Skip hidden dirs and the output file itself
            if '.git' in root or '__pycache__' in root or 'venv' in root or '.pytest_cache' in root:
                continue
                
            for file in files:
                if file == 'FINAL_OUTPUT.md' or file == 'generate_final_output.py' or file.endswith('.pyc') or file.endswith('.pkl') or file.endswith('.png'):
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_root)
                
                outfile.write(f"### File: {rel_path}\n")
                outfile.write("```\n")
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"Error reading file: {e}")
                outfile.write("\n```\n\n")
                
    print(f"Generated {output_file}")

if __name__ == "__main__":
    generate_final_output()
