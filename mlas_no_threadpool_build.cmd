build --skip_tests --skip_submodule_sync --config Release --cmake_extra_defines MLAS_NO_ONNXRUNTIME_THREADPOOL=1 && build\Windows\Release\Release\onnxruntime_mlas_test.exe