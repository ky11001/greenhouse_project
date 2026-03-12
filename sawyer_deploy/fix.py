import os

file_path = "detr/main.py"

# Read the file
with open(file_path, "r") as f:
    content = f.read()

# 1. Remove "required=True" which causes the crash
if "required=True" in content:
    print("Found 'required=True'. Removing it...")
    content = content.replace("required=True", "required=False")
else:
    print("Good: 'required=True' not found.")

# 2. Fix the parser line to ignore sys.argv
# We change "args = parser.parse_args()" to "args, _ = parser.parse_known_args(args=[])"
old_line = "args = parser.parse_args()"
new_line = "args, _ = parser.parse_known_args(args=[])"

if old_line in content:
    print("Found 'parser.parse_args()'. Patching it...")
    content = content.replace(old_line, new_line)
else:
    print("Note: Exact 'parser.parse_args()' line not found (might already be fixed).")

# Save the fixed file
with open(file_path, "w") as f:
    f.write(content)

print("\nSUCCESS: detr/main.py has been patched!")