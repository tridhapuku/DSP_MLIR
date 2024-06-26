import os
import subprocess
import time

# The script does the following
# Input : filename.py , enableAffineOpt , enableCanonOpt  
# Output : TimeOfExecution for different IP sizes : 
# Pseudo-code:
    # Iterate for all the input-size & update the input value in file
    # Update logic -- change the 2nd parameter of line: var c = getRangeOfVector(init , Count, StepSize)
    # Run the respective commands on the file

# Path to the input file
input_file_path = "back2backDelay.py"   #"audioCompression.py"
# enableAffineOpt = False  # Set to True to enable affine optimization
# enableCanonOpt = False   # Set to True to enable canonical optimization


# ************ Don't change unless u required 
# Define the values dictionary
values = {
    # "10": 10,
    # "100": 100,
    # "1K": 1000,
    # "10K": 10000,
    # "20K": 20000,
    # "30K": 30000,
    # "40K": 40000,
    # "50K": 50000,
    "100K": 100000,
    # "1M": 1000000
}
NoOfIterations = 1

commands_base = [
    # "./dsp1 lowPassFull.py -emit=mlir-affine",
    f"./dsp1 {input_file_path} -emit=mlir-affine",
    # "clang-17 file.ll -o fileexe -lm",
    # "./fileexe"
]

# Define the cases
# cases = [
#     {"affineOpt": False, "canonOpt": False, "suffix": "file2.ll"},
#     {"affineOpt": True, "canonOpt": False, "suffix": "file3.ll"},
#     {"affineOpt": True, "canonOpt": True, "suffix": "file4.ll"},
# ]

cases = [
    {"affineOpt": False, "canonOpt": False, "suffix": "fileNoOpt.mlir"},
    {"affineOpt": True, "canonOpt": False, "suffix": "fileAffineOpt.mlir"},
    {"affineOpt": True, "canonOpt": True, "suffix": "fileAffineCanonOpt.mlir"},
]



# Modify the base command based on flags
# if enableAffineOpt and enableCanonOpt:
#     commands_base[0] += " -affineOpt -canonOpt 2> file4.ll"
#     commands_base[1] = "clang-17 file4.ll -o fileexe -lm"
# elif enableAffineOpt:
#     commands_base[0] += " -affineOpt 2> file3.ll"
#     commands_base[1] = "clang-17 file3.ll -o fileexe -lm"
# else:
#     commands_base[0] += " 2> file2.ll"
#     commands_base[1] = "clang-17 file2.ll -o fileexe -lm"



# Read the input file
with open(input_file_path, "r") as file:
    lines = file.readlines()
   
for key, value in values.items():
    # Update the specific line in the file
    # print("Updating for {}".format(value))
    print("\n")
    print("{}".format(key), end="\t")
    with open(input_file_path, "w") as file:
        for line in lines:
            if line.strip().startswith("var input = getRangeOfVector("):
            # if line.strip().startswith("var N = "):
                # Replace the second parameter with the current value
                updated_line = f"    var input = getRangeOfVector(0, {value}, 1);\n"
                # updated_line = f"    var N = {value} ;\n"
                file.write(updated_line)
            else:
                file.write(line)
        # print(lines)

    # Iterate through the cases and run the commands
    for case in cases:
        command_mlir = commands_base[0]
        if case["affineOpt"]:
            command_mlir += " -affineOpt"
        if case["canonOpt"]:
            command_mlir += " -canonOpt"
        command_mlir += f" 2> {case['suffix']}"

        commands = [
            command_mlir,
            # f"clang-17 {case['suffix']} -o fileexe -lm",
        ] 
        print(case,end="\n")
        # print("\n")
        # Iterate over each value and perform the necessary operations


        for command in commands:
            # print("running command {}".format(command))
            # os.system(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            