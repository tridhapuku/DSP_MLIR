
// DebugConfig.h
#pragma once

#include "llvm/Support/raw_ostream.h"  // Include for llvm::errs()

// Enable or disable debug output
#define DEBUG_OUTPUT false

#if DEBUG_OUTPUT
    // Macro for debugging with arguments
    #define DEBUG_PRINT_WITH_ARGS(...) do { \
        llvm::errs() << "Line: " << __LINE__ << " file= " << __FILE__; \
        llvm::errs() << " " << __VA_ARGS__; \
        llvm::errs() << "\n"; \
    } while (0)

    // Macro for debugging without arguments
    #define DEBUG_PRINT_NO_ARGS() do { \
        llvm::errs() << "Line: " << __LINE__ << " file= " << __FILE__ << "\n"; \
    } while (0)
#else
    #define DEBUG_PRINT_WITH_ARGS(...)
    #define DEBUG_PRINT_NO_ARGS(...)
#endif

// #if DEBUG_OUTPUT
//     #define DEBUG_PRINT(x) (llvm::errs() << x << "\n")
// #else
//     #define DEBUG_PRINT(x)
// #endif

// #if DEBUG_OUTPUT
//     #define DEBUG_PRINT(...) do { \
//         llvm::errs() << "Line: " << __LINE__ << " func= " << __func__ << "\n "; \
//         llvm::errs() << __VA_ARGS__ << "\n"; \
//     } while (0)
// #else
//     #define DEBUG_PRINT(...)
// #endif


// #if DEBUG_OUTPUT
//     #define DEBUG_PRINT(...) do { \
//         llvm::errs() << "Line: " << __LINE__ << " file= " << __FILE__; \
//         if constexpr (sizeof(#__VA_ARGS__) > 0) { \
//             llvm::errs() << " "; \
//             llvm::errs() << __VA_ARGS__ << "\n"; \
//         } \
//         llvm::errs() << "\n"; \
//     } while (0)
// #else
//     #define DEBUG_PRINT(...)
// #endif


