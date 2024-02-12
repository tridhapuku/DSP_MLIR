add_custom_target(Dsp)
set_target_properties(Dsp PROPERTIES FOLDER Examples)

macro(add_toy_chapter name)
  add_dependencies(Dsp ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)


add_subdirectory(SimpleBlocks)

