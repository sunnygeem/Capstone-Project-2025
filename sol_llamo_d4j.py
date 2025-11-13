import os
import subprocess
import pandas as pd
import argparse
import sys
import torch
from transformer import VoltronTransformerPretrained, TokenizeMask
import javalang

DEFECTS4J_CMD = "path/to/defects4j/cloned/directory"
DEFECTS4J_PROJECTS = {
    'Chart': 26,
    'Lang': 65,
    'Math': 106,
    'Time': 27,
    'Mockito': 38,
    'Closure': 176,
    'Cli': 40,
    'Codec': 18, 
    'Collections': 28,
    'Compress': 47,
    'Csv': 16,
    'Gson': 18,
    'JacksonCore': 26,
    'JacksonDatabind': 112,
    'JacksonXml': 6,
    'Jsoup': 93,
    'JxPath': 22,
}

# < static info extracting logic >

def get_complexity(node):
    complexity = 1
    if hasattr(node, 'body'):
        for path, child in node:
            if isinstance(child, (javalang.tree.IfStatement, javalang.tree.ForStatement,
                                   javalang.tree.WhileStatement, javalang.tree.DoStatement,
                                   javalang.tree.SwitchStatementCase, javalang.tree.CatchClause)):
                complexity += 1
    return complexity

def analyze_file_ast(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()

    tree = javalang.parse.parse(code)
    
    class_name = next((cls.name for cls in tree.types if isinstance(cls, javalang.tree.ClassDeclaration)), None)

    methods_info = {}
    line_to_method_map = {}

    for path, node in tree.filter(javalang.tree.MethodDeclaration):
        method_name = f"{class_name}.{node.name}"
        start_line = node.position.line
        
        end_line = start_line
        if node.body:
            end_line = max(stmt.position.line for stmt in node.body if stmt.position) if node.body else start_line

        complexity = get_complexity(node)

        called_methods = set()
        for _, call_node in node.filter(javalang.tree.MethodInvocation):
            if call_node.qualifier is None or call_node.qualifier == 'this':
                 called_methods.add(f"{class_name}.{call_node.member}")

        methods_info[method_name] = {
            "start_line": start_line,
            "end_line": end_line,
            "complexity": complexity,
            "calls": list(called_methods)
        }

        for line_num in range(start_line, end_line + 2):
             line_to_method_map[line_num] = method_name
    
    return methods_info, line_to_method_map


def buglines_prediction(demo_type, pretrain_type, code_file_path, output_filepath):
    num_layer = 2
    target_dim = 512
    if demo_type == 'defects4j' and pretrain_type == "16B":
        target_dim = 1024
    if pretrain_type == '16B':
        dim_model = 6144
    elif pretrain_type == '6B':
        dim_model = 4096
    elif pretrain_type == '350M':
        dim_model = 1024
    
    if target_dim == 1024:
        num_head = 16
    elif target_dim == 512:
        num_head = 8
    elif target_dim == 256:
        num_head = 4

    model = VoltronTransformerPretrained(
        num_layer=num_layer, dim_model=dim_model, num_head=num_head, target_dim=target_dim
    )
    model.load_state_dict(torch.load(
        f'model_checkpoints/{demo_type}_{pretrain_type}'), strict=False)
    model.eval()
    
    tokenize_mask = TokenizeMask(pretrain_type)

    try:
        methods_info, line_to_method_map = analyze_file_ast(code_file_path)
    except Exception as e:
        print(f"[Warning] Failed to parse AST for {code_file_path}: {e}")
        methods_info, line_to_method_map = {}, {}
    
    with open(code_file_path, encoding='utf-8', errors='ignore') as f:
        original_lines = f.readlines()
    
    filtered_code_map = {} # {filtered line idx : original line idx}
    filtered_code_lines = []
    for i, code_line in enumerate(original_lines):
        stripped_line = code_line.strip()
        if stripped_line and not stripped_line.startswith('/') and not stripped_line.startswith('*') and not stripped_line.startswith('#') and stripped_line not in ['{', '}']:
            filtered_code_map[len(filtered_code_lines)] = i + 1
            filtered_code_lines.append(code_line)

    # < sliding window logic >
    window_size = 128
    stride = 64
    all_predictions = []

    for i in range(0, len(filtered_code_lines), stride):
        chunk_lines = filtered_code_lines[i : i + window_size]
        if not chunk_lines:
            continue
        
        code_chunk_str = "".join(chunk_lines)
        
        first_line_in_chunk_original_num = filtered_code_map.get(i, 1)
        current_method_name = line_to_method_map.get(first_line_in_chunk_original_num)
        
        structural_context = ""
        if current_method_name and current_method_name in methods_info:
            info = methods_info[current_method_name]
            structural_context = (
                f"// --- STATIC STRUCTURAL CONTEXT ---\n"
                f"// Method: {current_method_name}\n"
                f"// Complexity: {info['complexity']}\n"
                f"    (Note that High complexity indicates higher bug probability.)\n"
                f"// This method calls other methods in this file: {info['calls'] if info['calls'] else 'None'}\n"
            )
        
        input_with_context_info = structural_context + "// --- CODE START ---\n" + code_chunk_str

        input_tensor, mask, input_size, decoded_input = tokenize_mask.generate_token_mask(input_with_context_info)
        input_tensor = input_tensor[None, :]
        mask = mask[None, :]
        
        with torch.no_grad():
            predictions = model(input_tensor, mask)
        
        probabilities = torch.flatten(torch.sigmoid(predictions))
        real_indices = torch.flatten(mask == 1)
        probabilities = probabilities[real_indices].tolist()
        probabilities = probabilities[:len(chunk_lines)]

        for chunk_line_idx, score in enumerate(probabilities):
            if score > 0:
                filtered_line_idx = i + chunk_line_idx
                original_line_num = filtered_code_map.get(filtered_line_idx)
                if original_line_num:
                    all_predictions.append({
                        "line": original_line_num,
                        "score": round(score * 100, 2)
                    })

    # duplicate removal
    final_scores = {}
    for pred in all_predictions:
        line = pred['line']
        score = pred['score']
        if line not in final_scores:
            final_scores[line] = score
        elif line in final_scores:
            final_scores[line] += score # cumulative sum

    result_dict = [{"line": l, "score": s} for l, s in final_scores.items()]
    result_dict = sorted(result_dict, key=lambda d: d['score'], reverse=True)

    final_results_to_save = []
    for res in result_dict:
        original_line_content = original_lines[res['line'] - 1].strip()
        if not original_line_content.startswith('import '):
            final_results_to_save.append(res)

    with open(output_filepath, 'w', encoding='utf-8') as f:
        for res in final_results_to_save:
            original_line_content = original_lines[res['line'] - 1].strip()
            output_line = f"line-{res['line']} sus-{res['score']}%: {original_line_content}\n"
            f.write(output_line)

    return final_results_to_save

def get_buggy_files(work_dir):
    cmd_modified = f"{DEFECTS4J_CMD} export -p classes.modified -w {work_dir}"
    proc_modified = subprocess.run(cmd_modified, shell=True, check=True, capture_output=True, text=True)
    modified_classes = proc_modified.stdout.strip().splitlines()
    if not modified_classes:
        return []
    cmd_src_dir = f"{DEFECTS4J_CMD} export -p dir.src.classes -w {work_dir}"
    proc_src_dir = subprocess.run(cmd_src_dir, shell=True, check=True, capture_output=True, text=True)
    source_dir = proc_src_dir.stdout.strip()
    buggy_files = []
    for class_name in modified_classes:
        relative_path = class_name.replace(".", "/") + ".java"
        full_path = os.path.join(work_dir, source_dir, relative_path)
        buggy_files.append(full_path)
    return buggy_files

def main(model_size):
    dataset_type = 'defects4j'
    results = []
    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)

    for project, num_bugs in DEFECTS4J_PROJECTS.items():
        for bug_id in range(1, num_bugs + 1):
            work_dir = f"/tmp/{project}-{bug_id}b"
            try:
                checkout_cmd = f"{DEFECTS4J_CMD} checkout -p {project} -v {bug_id}b -w {work_dir}"
                subprocess.run(checkout_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                buggy_files = get_buggy_files(work_dir)
                if not buggy_files:
                    print(f"[Warning] No modified files found for {project}-{bug_id}. Skipping.")
                    continue
                for file_path in buggy_files:
                    if not os.path.exists(file_path):
                        print(f"[Warning] File path does not exist: {file_path}. Skipping.")
                        continue

                    detail_output_path = os.path.join(output_dir, f"{project}-{bug_id}_{os.path.basename(file_path)}.txt")
                    
                    detailed_predictions = buglines_prediction(dataset_type, model_size, file_path, detail_output_path)
                    
                    results.append({
                        'project': project,
                        'bug_id': bug_id,
                        'file_path': file_path,
                        'predicted_lines': detailed_predictions,
                    })
            except subprocess.CalledProcessError as e:
                print(f"[Error] Subprocess failed for {project}-{bug_id} with exit code {e.returncode}.")
                print(f"--> Failed Command: {e.cmd}")
                print(f"--> Detailed Error: {e.stderr.decode('utf-8', 'ignore')}")
            except Exception as e:
                print(f"[Error] An unexpected error occurred while processing {project}-{bug_id}: {e}")
            finally:
                subprocess.run(f"rm -rf {work_dir}", shell=True, check=False)
    output_filename = f"defects4j_results_{model_size}.csv"
    pd.DataFrame(results).to_csv(output_filename, index=False)

if __name__ == '__main__':
    if "JAVA_HOME" not in os.environ:
        print("[Fatal] JAVA_HOME environment variable is not set. Please set it to your JDK path.")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Run LLMAO fault localization on the Defects4J benchmark.")
    parser.add_argument('pretrain_type', type=str, help="The pretrained model size (e.g., '350M').")
    args = parser.parse_args()
    main(args.pretrain_type)