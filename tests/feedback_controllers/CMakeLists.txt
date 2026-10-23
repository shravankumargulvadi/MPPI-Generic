add_executable(generic_feedback_controller_test ${PROJECT_SOURCE_DIR}/tests/test_main.cpp generic_feedback_controller_test.cu)
target_link_libraries(generic_feedback_controller_test gtest
        gmock
        gtest_main
        cartpole_mppi)
gtest_discover_tests(generic_feedback_controller_test)
set_target_properties(generic_feedback_controller_test PROPERTIES FOLDER test)

add_executable(ddp_tests ${PROJECT_SOURCE_DIR}/tests/test_main.cpp ddp_test.cu)
target_link_libraries(ddp_tests gtest
        gmock
        gtest_main
        cartpole_mppi)
gtest_discover_tests(ddp_tests)
set_target_properties(ddp_tests PROPERTIES FOLDER test)

add_executable(adaptive_rmppi_reference_test ${PROJECT_SOURCE_DIR}/tests/test_main.cpp adaptive_rmppi_reference_test.cu)
target_link_libraries(adaptive_rmppi_reference_test gtest
        gmock
        gtest_main
        ${MPPI_HEADER_LIBRARY_NAME})
gtest_discover_tests(adaptive_rmppi_reference_test)
set_target_properties(adaptive_rmppi_reference_test PROPERTIES FOLDER test)
