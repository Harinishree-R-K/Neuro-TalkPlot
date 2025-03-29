# Cleans null bytes from 'gcodetopy.py' before execution
file_path = "gcodetopy.py"

with open(file_path, "rb") as f:
    content = f.read()

# Remove null bytes
clean_content = content.replace(b"\x00", b"")

# Overwrite the original file
with open(file_path, "wb") as f:
    f.write(clean_content)

print("Null bytes removed successfully. Running gcodetopy.py...\n")

# Run the cleaned script
import subprocess
subprocess.run(["python", file_path])
