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
input_file_path = "back2backDelay.py"      # " audioCompression.py"
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
    # "100K": 100000,
     "1M": 1000000,
    # "10M": 10000000,
    # "20M": 20000000,
    # "30M": 30000000,
    # "40M": 40000000,
    # "50M": 50000000,
    # "100M": 100000000,
    # "1B": 1000000000
}
NoOfIterations = 100

# OptCase = []

# commands_base = [
#     "./dsp1 lowPassFull.py -emit=llvm",
#     "clang-17 file.ll -o fileexe -lm",
#     # "./fileexe"
# ]

commands_base = [
    # "./dsp1 lowPassFull.py -emit=mlir-affine",
    f"./dsp1 {input_file_path} -emit=llvm",
    "clang-17 -O0 file.ll -o fileexe -lm",
    # "./fileexe"
]

# Define the cases
# cases = [
#     {"affineOpt": False, "canonOpt": False, "suffix": "file2.ll"},
#     {"affineOpt": True, "canonOpt": False, "suffix": "file3.ll"},
#     {"affineOpt": True, "canonOpt": True, "suffix": "file4.ll"},
# ]

cases = [
    # {"affineOpt": True, "canonOpt": True, "suffix": "fileAffineCanonOpt.ll"},
    # {"affineOpt": True, "canonOpt": False, "suffix": "fileAffineOpt.ll"},
    {"affineOpt": False, "canonOpt": False, "suffix": "fileNoOpt.ll"},    
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
                updated_line = f"    var input = getRangeOfVector(0, {value}, 10);\n"
                # updated_line = f"    var N = {value} ;\n"
                file.write(updated_line)
            else:
                file.write(line)
        # print(lines)

    # Iterate through the cases and run the commands
    for case in cases:
        command_llvm = commands_base[0]
        if case["affineOpt"]:
            command_llvm += " -affineOpt"
        if case["canonOpt"]:
            command_llvm += " -canonOpt"
        command_llvm += f" 2> {case['suffix']}"

        commands = [
            command_llvm,
            f"clang-17 {case['suffix']} -o fileexe -lm",
        ] 
        # print(case,end="\n")
        # print("\n")
        # Iterate over each value and perform the necessary operations
        
        for command in commands:
            # Run the commands for the current case
            result = subprocess.run(command, shell=True, capture_output=True, text=True)  
            
        sum_exe_time = 0
        for i in range(0,NoOfIterations):
            # for command in commands:
            #     # print("running command {}".format(command))
            #     # os.system(command)
            #     result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            # Clear the cache to minimize caching effects
            # subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True)
            subprocess.run("sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", shell=True, check=True)

            # The command to be executed
            # command2 = "./fileexe"
            # Limit execution to a single core
            command2 = "taskset -c 0 ./fileexe"  

            # Record the start time
            start_time = time.time()

            # Execute the command
            subprocess.run(command2, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # subprocess.run(command2, shell=True)

            # Record the end time
            end_time = time.time()

            # Calculate the elapsed time
            execution_time = end_time - start_time
            sum_exe_time = sum_exe_time + execution_time
            # print("{}".format(execution_time), end="\t")
        avg_exe_time = sum_exe_time / NoOfIterations
        print("{}".format(avg_exe_time), end="\t")
    # print(f"The command took {execution_time} seconds to execute.")






# Commands to run for each case
    # commands = [
    #     # "./dsp1 audioCompression.py -emit=llvm 2> file2.ll",
    #     # "clang-17 file2.ll -o file2exe -lm",
    #     # "(time ./file2exe > file123) 2> timeOut",
    #     # "time ./file2exe "
    #     "./dsp1 audioCompression.py -emit=llvm -affineOpt 2> file3.ll",
    #     "clang-17 file3.ll -o file3exe -lm",

    # ]