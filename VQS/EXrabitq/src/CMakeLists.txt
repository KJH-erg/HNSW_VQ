set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(create_index create_index.cpp)
target_link_libraries(create_index PRIVATE faiss openblas pthread)
# add_executable(query query.cpp)
# target_link_libraries(query PRIVATE faiss openblas pthread)