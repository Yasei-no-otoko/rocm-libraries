# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared CMake support for provider-owned rocKE ahead-of-time artifacts.
#
# rocKE (`rocke`) and the library packages (`kernels`) are consumed from the
# build-local rocke-pyenv (rocke/CMakeLists.txt), so every Python invocation runs
# under ${ROCKE_PYENV_PYTHON} and depends on ${ROCKE_PYENV_STAMP}. The only extra
# import root is this tree's own `rocke_client_aot` tooling package (not part of
# the editable library install).

if(NOT DEFINED ROCKE_PYENV_PYTHON OR NOT DEFINED ROCKE_PYENV_STAMP)
    message(FATAL_ERROR
        "rocke_aot.cmake requires the rocke-pyenv variables (ROCKE_PYENV_PYTHON / "
        "ROCKE_PYENV_STAMP) from rocke/CMakeLists.txt; enable ROCKE_BUILD_PYENV."
    )
endif()

get_filename_component(_ROCKE_CLIENT_AOT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_ROCKE_CLIENT_AOT_BUILD_TOOL "${_ROCKE_CLIENT_AOT_ROOT}/tools/rocke_aot_build.py")
set(_ROCKE_CLIENT_AOT_PYTHON_ROOT "${_ROCKE_CLIENT_AOT_ROOT}/python")
set(_ROCKE_CLIENT_AOT_SCHEMA_ROOT "${_ROCKE_CLIENT_AOT_ROOT}/schemas")

# --- libamd_comgr availability (compile_kernel backend="python") -------------
# rocke.helpers.compile_kernel ctypes-loads libamd_comgr at build time. We locate
# it through comgr's own CMake package (amd_comgr): TheRock's amd-comgr subproject
# provides it, and a standalone build finds it under /opt/rocm. The imported target
# points at comgr in its assembled tree, where comgr's own RUNPATH
# ($ORIGIN, $ORIGIN/llvm/lib, $ORIGIN/rocm_sysdeps/lib) resolves its entire
# dependency closure (libLLVM, libclang-cpp, the vendored rocm_sysdeps libs). So
# once comgr is loaded from that tree nothing about its deps needs forwarding.
# Resolution order:
#   1. explicit ROCKE_COMGR_LIB (cache/env) override,
#   2. find_package(amd_comgr) imported target location,
#   3. otherwise unset, and compile_kernel's own resolver applies at build time.
set(_ROCKE_COMGR_LIB "")
if(DEFINED ROCKE_COMGR_LIB AND NOT "${ROCKE_COMGR_LIB}" STREQUAL "")
    set(_ROCKE_COMGR_LIB "${ROCKE_COMGR_LIB}")
elseif(DEFINED ENV{ROCKE_COMGR_LIB} AND NOT "$ENV{ROCKE_COMGR_LIB}" STREQUAL "")
    set(_ROCKE_COMGR_LIB "$ENV{ROCKE_COMGR_LIB}")
else()
    find_package(amd_comgr CONFIG QUIET)
    if(TARGET amd_comgr)
        get_target_property(_ROCKE_COMGR_CFGS amd_comgr IMPORTED_CONFIGURATIONS)
        foreach(_cfg RELEASE ${_ROCKE_COMGR_CFGS})
            get_target_property(_ROCKE_COMGR_LOC amd_comgr IMPORTED_LOCATION_${_cfg})
            if(_ROCKE_COMGR_LOC)
                set(_ROCKE_COMGR_LIB "${_ROCKE_COMGR_LOC}")
                break()
            endif()
        endforeach()
    endif()
endif()
if("${_ROCKE_COMGR_LIB}" STREQUAL "")
    message(STATUS
        "rocKE AOT build: libamd_comgr not resolved at configure; "
        "compile_kernel's runtime resolver will apply")
elseif(NOT EXISTS "${_ROCKE_COMGR_LIB}")
    message(FATAL_ERROR "ROCKE_COMGR_LIB does not exist: ${_ROCKE_COMGR_LIB}")
else()
    get_filename_component(_ROCKE_COMGR_LIB "${_ROCKE_COMGR_LIB}" ABSOLUTE)
    message(STATUS "rocKE AOT build: libamd_comgr from ${_ROCKE_COMGR_LIB}")
endif()

# Reconfigure when common AOT Python helpers or shared JSON Schemas change.
# Kernel families add their handler + family schemas via the freshness globs in
# rocke_client_add_aot_instances().
file(GLOB_RECURSE _ROCKE_CLIENT_AOT_PACKAGE_MODULES CONFIGURE_DEPENDS
    "${_ROCKE_CLIENT_AOT_PYTHON_ROOT}/rocke_client_aot/*.py"
)
file(GLOB _ROCKE_CLIENT_AOT_COMMON_SCHEMA_DEPENDS CONFIGURE_DEPENDS
    "${_ROCKE_CLIENT_AOT_SCHEMA_ROOT}/*.schema.json"
)

# The .co bytes are determined by the kernels package (builders/signatures) and
# the rocke platform (compile_kernel lowering). Both are editable-installed in
# the pyenv, so edits take effect at runtime but the pyenv stamp never re-fires
# for them. Track their sources so `cmake --build` regenerates the .co + sidecar
# artifacts when the code that produces them changes, rather than shipping stale
# artifacts. Coarse globs: correctness over minimal rebuilds.
get_filename_component(_ROCKE_LIBRARY_DIR "${_ROCKE_CLIENT_AOT_ROOT}/.." ABSOLUTE)
get_filename_component(_ROCKE_ROOT_DIR "${_ROCKE_LIBRARY_DIR}/.." ABSOLUTE)
file(GLOB_RECURSE _ROCKE_CLIENT_AOT_KERNEL_SOURCES CONFIGURE_DEPENDS
    "${_ROCKE_LIBRARY_DIR}/kernels/*.py"
)
file(GLOB_RECURSE _ROCKE_CLIENT_AOT_PLATFORM_SOURCES CONFIGURE_DEPENDS
    "${_ROCKE_ROOT_DIR}/platform/python/rocke/*.py"
)

# Return the PYTHONPATH used by rocKE client AOT tooling in OUT_VAR.
#
# The pyenv already resolves `rocke` and `kernels`; this tree's tooling package
# is prepended, followed by an incoming developer PYTHONPATH.
function(rocke_client_aot_pythonpath OUT_VAR)
    set(_ROCKE_CLIENT_AOT_PYTHONPATH "${_ROCKE_CLIENT_AOT_PYTHON_ROOT}")
    if(DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
        cmake_path(CONVERT "$ENV{PYTHONPATH}" TO_CMAKE_PATH_LIST
                   _ROCKE_CLIENT_AOT_INCOMING_PYTHONPATH)
        list(APPEND _ROCKE_CLIENT_AOT_PYTHONPATH
             ${_ROCKE_CLIENT_AOT_INCOMING_PYTHONPATH})
    endif()
    cmake_path(CONVERT "${_ROCKE_CLIENT_AOT_PYTHONPATH}" TO_NATIVE_PATH_LIST
               _ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE)
    set(${OUT_VAR} "${_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE}" PARENT_SCOPE)
endfunction()

# Return the CMake -E env / CTest ENVIRONMENT entries for AOT Python commands.
#
# PYTHONDONTWRITEBYTECODE keeps configure/build/test runs from writing __pycache__
# into source trees, which matters because generated artifacts live in the build
# tree and the source tree should stay reviewable.
function(rocke_client_aot_pythonpath_environment OUT_VAR)
    rocke_client_aot_pythonpath(_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE)
    string(REPLACE ";" "\\;" _ROCKE_CLIENT_AOT_PYTHONPATH_ESCAPED
           "${_ROCKE_CLIENT_AOT_PYTHONPATH_NATIVE}")
    set(_ROCKE_CLIENT_AOT_ENV
        "PYTHONPATH=${_ROCKE_CLIENT_AOT_PYTHONPATH_ESCAPED}"
        "PYTHONDONTWRITEBYTECODE=1"
    )
    # Forward the resolved libamd_comgr path; compile_kernel honors ROCKE_COMGR_LIB.
    # comgr is loaded from its assembled tree, where comgr's own RUNPATH resolves
    # its full dependency closure, so no dependency directories are forwarded.
    if(NOT "${_ROCKE_COMGR_LIB}" STREQUAL "")
        list(APPEND _ROCKE_CLIENT_AOT_ENV "ROCKE_COMGR_LIB=${_ROCKE_COMGR_LIB}")
    endif()
    set(${OUT_VAR} "${_ROCKE_CLIENT_AOT_ENV}" PARENT_SCOPE)
endfunction()

# Derive the per-instance .co + sidecar output paths from an aot_list.json
# array (one object per instance, keyed by "name"). Returns a single list in
# GEN_VAR (.co + sidecar paths) in the caller scope.
function(_rocke_client_aot_derive_outputs GEN_VAR ARCH_OUTPUT_DIR AOT_LIST)
    file(READ "${AOT_LIST}" _JSON)
    string(JSON _COUNT ERROR_VARIABLE _JSON_ERROR LENGTH "${_JSON}")
    if(_JSON_ERROR)
        message(FATAL_ERROR "Failed to parse ${AOT_LIST}: ${_JSON_ERROR}")
    endif()
    if(_COUNT EQUAL 0)
        message(FATAL_ERROR "No rocKE client AOT instances found in ${AOT_LIST}")
    endif()
    set(_GEN)
    set(_INDEX 0)
    while(_INDEX LESS _COUNT)
        string(JSON _NAME GET "${_JSON}" ${_INDEX} "name")
        list(APPEND _GEN "${ARCH_OUTPUT_DIR}/${_NAME}.co"
             "${ARCH_OUTPUT_DIR}/${_NAME}.sidecar.json")
        math(EXPR _INDEX "${_INDEX} + 1")
    endwhile()
    set(${GEN_VAR} "${_GEN}" PARENT_SCOPE)
endfunction()

# Install one architecture's loose per-kernel .co + .sidecar.json files.
# Per-arch files carry the arch in their name so a multi-arch build -- or
# TheRock's per-family artifact union -- merges them by simple co-location.
# Positional: NAME ARCH; the generated .co + sidecar outputs follow as ARGN.
function(_rocke_client_aot_install NAME ARCH)
    set(_INSTALL_ROOT
        "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/arch_content/rocke/${ARCH}")
    install(FILES ${ARGN} DESTINATION "${_INSTALL_ROOT}")
endfunction()

# Register one AOT kernel architecture instance list and its install rules.
#
# Required arguments:
#   NAME      Target name and per-arch artifact directory component.
#   ARCH      rocKE architecture component, e.g. gfx942 or gfx1151.
#   ARCH_DIR  Directory holding this arch's aot_list.json
#             (kernels/<arch>/<family>/).
#
# Optional arguments:
#   HANDLER        Family AOT handler module (kernels/common/<family>_aot.py).
#   SCHEMA_DIR     Family schema overlay dir (defaults to the shared AOT schemas).
#   PYTHON_DEPENDS Extra Python files whose edits should rebuild the artifacts.
function(rocke_client_add_aot_instances)
    cmake_parse_arguments(ARG "" "NAME;ARCH;ARCH_DIR;HANDLER;SCHEMA_DIR" "PYTHON_DEPENDS" ${ARGN})
    if(NOT ARG_NAME OR NOT ARG_ARCH OR NOT ARG_ARCH_DIR OR NOT ARG_HANDLER)
        message(FATAL_ERROR
            "rocke_client_add_aot_instances requires NAME, ARCH, ARCH_DIR, and HANDLER"
        )
    endif()

    if(NOT TARGET rocke_client_aot_artifacts)
        message(FATAL_ERROR
            "Create rocke_client_aot_artifacts before "
            "calling rocke_client_add_aot_instances"
        )
    endif()

    get_filename_component(_ARCH_DIR "${ARG_ARCH_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT IS_DIRECTORY "${_ARCH_DIR}")
        message(FATAL_ERROR "rocKE client AOT arch directory does not exist: ${_ARCH_DIR}")
    endif()
    get_filename_component(_HANDLER "${ARG_HANDLER}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${_HANDLER}")
        message(FATAL_ERROR "rocKE client AOT handler does not exist: ${_HANDLER}")
    endif()

    set(_AOT_LIST "${_ARCH_DIR}/aot_list.json")
    if(NOT EXISTS "${_AOT_LIST}")
        message(FATAL_ERROR "rocKE client AOT arch directory is missing aot_list.json: ${_ARCH_DIR}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_AOT_LIST}")

    # Family schema overlay (optional) + freshness globs.
    set(_SCHEMA_ARGS)
    set(_SCHEMA_DEPENDS)
    if(ARG_SCHEMA_DIR)
        get_filename_component(_SCHEMA_DIR "${ARG_SCHEMA_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        set(_SCHEMA_ARGS --schema-dir "${_SCHEMA_DIR}")
        file(GLOB _SCHEMA_DEPENDS CONFIGURE_DEPENDS "${_SCHEMA_DIR}/*.schema.json")
    endif()

    rocke_client_aot_pythonpath_environment(_AOT_ENV)

    set(_ARCH_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifacts/${ARG_ARCH}/${ARG_NAME}")
    set(_BUILD_STAMP "${_ARCH_OUTPUT_DIR}/build.stamp")

    # Derive the generated .co and sidecar outputs from the instance names in
    # aot_list.json so Ninja tracks each artifact precisely.
    _rocke_client_aot_derive_outputs(
        _GENERATED_OUTPUTS "${_ARCH_OUTPUT_DIR}" "${_AOT_LIST}")

    # Recreate the artifact directory on every rebuild so removed/renamed
    # instances cannot leave stale .co or sidecar files behind.
    add_custom_command(
        OUTPUT "${_BUILD_STAMP}" ${_GENERATED_OUTPUTS}
        COMMAND "${CMAKE_COMMAND}" -E remove_directory "${_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_AOT_LIST}" "${_ARCH_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E env ${_AOT_ENV}
                "${ROCKE_PYENV_PYTHON}" "${_ROCKE_CLIENT_AOT_BUILD_TOOL}"
                --artifact-dir "${_ARCH_OUTPUT_DIR}"
                --handler "${_HANDLER}"
                ${_SCHEMA_ARGS}
                --arch "${ARG_ARCH}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_BUILD_STAMP}"
        DEPENDS "${ROCKE_PYENV_STAMP}"
                "${_HANDLER}"
                "${_AOT_LIST}"
                "${_ROCKE_CLIENT_AOT_BUILD_TOOL}"
                ${_ROCKE_CLIENT_AOT_PACKAGE_MODULES}
                ${_ROCKE_CLIENT_AOT_KERNEL_SOURCES}
                ${_ROCKE_CLIENT_AOT_PLATFORM_SOURCES}
                ${_ROCKE_CLIENT_AOT_COMMON_SCHEMA_DEPENDS}
                ${_SCHEMA_DEPENDS}
                ${ARG_PYTHON_DEPENDS}
        VERBATIM
        COMMENT "Build rocKE client ${ARG_ARCH} ${ARG_NAME} AOT artifacts"
    )
    add_custom_target("${ARG_NAME}"
        DEPENDS "${_BUILD_STAMP}" ${_GENERATED_OUTPUTS}
        COMMENT "Build ${ARG_NAME} AOT artifacts"
    )
    add_dependencies(rocke_client_aot_artifacts "${ARG_NAME}")

    # Per-arch install: loose .co + sidecar files (split out to keep this
    # registration function within the cmake-lint statement budget).
    _rocke_client_aot_install(
        "${ARG_NAME}" "${ARG_ARCH}" ${_GENERATED_OUTPUTS})
endfunction()
