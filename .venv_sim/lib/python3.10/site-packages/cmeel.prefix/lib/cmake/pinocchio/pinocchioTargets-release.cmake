#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pinocchio::pinocchio" for configuration "Release"
set_property(TARGET pinocchio::pinocchio APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pinocchio::pinocchio PROPERTIES
  IMPORTED_LOCATION_RELEASE "${PACKAGE_PREFIX_DIR}/lib/libpinocchio.so"
  IMPORTED_SONAME_RELEASE "libpinocchio.so"
  )

list(APPEND _cmake_import_check_targets pinocchio::pinocchio )
list(APPEND _cmake_import_check_files_for_pinocchio::pinocchio "${PACKAGE_PREFIX_DIR}/lib/libpinocchio.so" )

# Import target "pinocchio::pinocchio_pywrap" for configuration "Release"
set_property(TARGET pinocchio::pinocchio_pywrap APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pinocchio::pinocchio_pywrap PROPERTIES
  IMPORTED_LOCATION_RELEASE "${PACKAGE_PREFIX_DIR}/lib/python3.10/site-packages/pinocchio/pinocchio_pywrap.cpython-310-x86_64-linux-gnu.so"
  IMPORTED_SONAME_RELEASE "pinocchio_pywrap.cpython-310-x86_64-linux-gnu.so"
  )

list(APPEND _cmake_import_check_targets pinocchio::pinocchio_pywrap )
list(APPEND _cmake_import_check_files_for_pinocchio::pinocchio_pywrap "${PACKAGE_PREFIX_DIR}/lib/python3.10/site-packages/pinocchio/pinocchio_pywrap.cpython-310-x86_64-linux-gnu.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
