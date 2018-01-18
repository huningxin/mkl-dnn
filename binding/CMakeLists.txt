set(MKLDNN_JS "mkl-dnn")
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(${MKLDNN_JS} src/binding.cpp)
set_property(TARGET ${MKLDNN_JS} PROPERTY CXX_STANDARD 11)
set_target_properties(${MKLDNN_JS} PROPERTIES LINK_FLAGS "-s WASM=1 --bind -Wno-unused-command-line-argument")
target_link_libraries(${MKLDNN_JS} ${LIB_NAME})

set(test_files "")
set(test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test)
file(GLOB_RECURSE files RELATIVE "${test_dir}" "${test_dir}/*")
foreach(f ${files})
  message(STATUS "copy ${test_dir}/${f} ${CMAKE_CURRENT_BINARY_DIR}/${f}")
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${f}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${test_dir}/${f}" "${CMAKE_CURRENT_BINARY_DIR}/${f}"
                     DEPENDS "${test_dir}/${f}"
                     COMMENT "Copying ${f}"
                    )
  list(APPEND test_files "${test_dir}/${f}" "${CMAKE_CURRENT_BINARY_DIR}/${f}")
endforeach()

message(STATUS "${test_files}")

add_custom_target(mkldnn.js ALL
                  DEPENDS ${MKLDNN_JS} ${test_files})