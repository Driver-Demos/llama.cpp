# Purpose
This C header file, `rotate_defs.h`, provides macro definitions for bitwise rotation operations on integers of various sizes (8, 16, 32, and 64 bits). It includes conditional compilation to handle differences between Microsoft Visual C++ (`_MSC_VER`) and other compilers. For MSVC, it uses built-in functions `_rotl` and `_rotr` for 32-bit and 64-bit rotations. For other compilers, it defines the rotation macros using bitwise operations and type casting to ensure correct behavior across different platforms. The file ensures that the operations are performed within the bounds of the specified integer sizes by using masks and type casts, providing a portable way to perform these common bit manipulation tasks.
# Imports and Dependencies

---
- `stdlib.h`
- `stdint.h`


