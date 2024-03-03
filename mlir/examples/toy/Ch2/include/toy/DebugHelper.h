//===- DebugHelper.h - Helper for printing debug prints----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_AST_H
#define TOY_AST_H

#include <iostream>

#define DEBUG_ENABLED 

// Define a macro for debug prints
#ifdef DEBUG_ENABLED
#define DEBUG_PRINT(msg) std::cout << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

#endif