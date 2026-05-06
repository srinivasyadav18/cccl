//===----------------------------------------------------------------------===//
//
// reproduce.cu
//
// Minimal libcu++-only reproducer for nvcc #20011-D.
//
// Verified failing under: CUDA 13.1.115, clang++ 18, sm_86, -std=c++17,
// -Xcudafe=--promote_warnings -Xcompiler=-Werror.
//
// --- Root cause ---------------------------------------------------------
//
//   cuda::buffer<T>::~buffer() is compiler-generated with __host__ __device__
//   annotations. It implicitly invokes the destructor of its single data
//   member cuda::__uninitialized_async_buffer<T, ...>, which is declared
//   _CCCL_HOST_API (__host__ only). When cuda::std::unique_ptr<cuda::buffer<T>>
//   is used anywhere in the translation unit -- even as an uninstantiated
//   declaration -- nvcc instantiates unique_ptr's default deleter, which
//   transitively forces the HD check of ~buffer<T>() against the host-only
//   inner dtor, producing:
//
//     error #20011-D: calling a __host__ function
//       ("cuda::__uninitialized_async_buffer<T, ...>::~__uninitialized_async_buffer()")
//       from a __host__ __device__ function ("cuda::buffer<T, ...>::~buffer")
//       is not allowed
//
//   The diagnostic is blamed on
//   cuda/__utility/__basic_any/virtual_tables.h:97 (an HD __query_interface
//   template) -- nvcc's attribution happens to land on the deepest HD
//   function it was analyzing, but the real fix must be in libcu++ aligning
//   annotations between buffer and __uninitialized_async_buffer (either make
//   ~buffer host-only, or make ~__uninitialized_async_buffer HD).
//
// --- Bug trigger --------------------------------------------------------
//
//   Just naming `cuda::std::unique_ptr<cuda::buffer<T, device_accessible>>`
//   as a variable type is sufficient. No construction, no allocation, no
//   member access.
//
// --- Build --------------------------------------------------------------
//
//   nvcc -ccbin=/usr/bin/clang++ -std=c++17 -arch=sm_86 \
//        -D_CCCL_NO_SYSTEM_HEADER \
//        -I/home/coder/cccl/libcudacxx/include \
//        -I/home/coder/cccl/thrust \
//        -I/home/coder/cccl/cub \
//        -Xcudafe=--display_error_number -Xcudafe=--promote_warnings \
//        -Xcompiler=-Werror \
//        -c /home/coder/cccl/reproduce.cu -o /dev/null
//
//   IMPORTANT: the `-D_CCCL_NO_SYSTEM_HEADER` flag is mandatory to see the
//   diagnostic. The CCCL headers carry `#pragma GCC system_header`, so by
//   default nvcc treats the call as a system-header warning and silently
//   suppresses it. The cudax/cub/thrust test suites set this macro (so they
//   DO see the error -- which is exactly how this showed up in the cudax
//   static_map test); end-user code compiling against the installed CCCL
//   headers typically does not, which is why this latent bug has gone
//   unnoticed outside the test infrastructure.
//
//===----------------------------------------------------------------------===//

#include <cuda/__container/buffer.h>
#include <cuda/std/__memory/unique_ptr.h>

int main()
{
  cuda::std::unique_ptr<cuda::buffer<int, cuda::mr::device_accessible>> ptr;
  (void) ptr;
  return 0;
}
