add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage
        PRIVATE
        vanilla_options
)

add_executable(advanced_models advanced_models.cpp)
target_link_libraries(advanced_models
        PRIVATE
        vanilla_options
)