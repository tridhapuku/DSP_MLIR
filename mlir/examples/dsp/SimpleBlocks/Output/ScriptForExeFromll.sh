#!/bin/bash

#Configurable Variables
input="gain10K_simple"

#constant parameter
inputFileExt=".py"
mlir=".mlir"
llvm=".ll"
obj=".o"

#FileNames
inputFile="$input$inputFileExt"
outMlirFile="$input$mlir"
outllvmFile="$input$llvm"
OutputExe="$input$obj"

# Constant paramters
BasePath="/home/abhinav/ForMLIR/SourceCode/llvm-project"
InputFilePath="/mlir/test/Examples/DspExample/gain"

echo "Output-Debug"
echo "  $inputFile , $outMlirFile ,$outllvmFile, $OutputExe "
echo " $BasePath$InputFilePath/$inputFile"

#Steps
#get outMlir with Opt, outllvm , outObj file, run the exe and get the time

echo "Get MLIR-input"
$BasePath/build/bin/dsp1 $BasePath$InputFilePath/$inputFile -emit=mlir -opt 2> $outMlirFile

echo "Get LLVM IR"
# # $BasePath/build/bin/dsp1 $BasePath/mlir/test/Examples/DspExample/$outMlirFile -emit=llvm -opt 2> $inputllvmName
# $BasePath/build/bin/dsp1 $BasePath/mlir/test/Examples/DspExample/$outMlirFile -emit=llvm 2> $outllvmFile
$BasePath/build/bin/dsp1 $outMlirFile -emit=llvm 2> $outllvmFile

echo "use clang-17 to get obj file"
clang-17 $outllvmFile -o $OutputExe

echo "running the exe"
time ./$OutputExe
