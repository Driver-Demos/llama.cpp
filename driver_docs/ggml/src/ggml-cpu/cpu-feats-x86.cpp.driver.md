# Purpose
This C++ source code file is designed to detect and evaluate the capabilities of x86 processors, specifically focusing on the presence of various instruction set extensions. The file defines a [`cpuid_x86`](#cpuid_x86cpuid_x86) struct that encapsulates the functionality to query the CPU for supported features using the CPUID instruction, which is a processor supplementary instruction for x86 architecture. The struct provides a series of boolean methods that check for the presence of specific CPU features such as SSE3, AVX, AVX2, and various AVX512 extensions, among others. These methods rely on bitset representations of the CPU's feature flags, which are populated during the struct's initialization by executing the CPUID instruction with different function IDs.

The file also includes a function, [`ggml_backend_cpu_x86_score`](#ggml_backend_cpu_x86_score), which calculates a score based on the presence of certain CPU features. This score is used to determine the suitability of the CPU for specific computational tasks, likely in the context of a larger software system that can leverage these features for performance optimization. The code is conditionally compiled for x86_64 architectures and includes both Microsoft and GCC/Clang specific implementations for executing the CPUID instruction. The file is part of a backend implementation, as indicated by the inclusion of "ggml-backend-impl.h", and it provides a mechanism to assess CPU capabilities, which can be used to optimize software execution paths based on the available hardware features.
# Imports and Dependencies

---
- `ggml-backend-impl.h`
- `intrin.h`
- `cstring`
- `vector`
- `bitset`
- `array`
- `string`


# Data Structures

---
### cpuid\_x86<!-- {{#data_structure:cpuid_x86}} -->
- **Type**: `struct`
- **Members**:
    - `is_intel`: A boolean indicating if the CPU is an Intel processor.
    - `is_amd`: A boolean indicating if the CPU is an AMD processor.
    - `vendor`: A string storing the vendor name of the CPU.
    - `brand`: A string storing the brand name of the CPU.
    - `f_1_ecx`: A 32-bit bitset representing feature flags from the CPUID instruction with EAX=1, ECX register.
    - `f_1_edx`: A 32-bit bitset representing feature flags from the CPUID instruction with EAX=1, EDX register.
    - `f_7_ebx`: A 32-bit bitset representing feature flags from the CPUID instruction with EAX=7, EBX register.
    - `f_7_ecx`: A 32-bit bitset representing feature flags from the CPUID instruction with EAX=7, ECX register.
    - `f_7_edx`: A 32-bit bitset representing feature flags from the CPUID instruction with EAX=7, EDX register.
    - `f_7_1_eax`: A 32-bit bitset representing feature flags from the CPUID instruction with EAX=7, sub-leaf 1, EAX register.
    - `f_81_ecx`: A 32-bit bitset representing feature flags from the CPUID instruction with EAX=0x80000001, ECX register.
    - `f_81_edx`: A 32-bit bitset representing feature flags from the CPUID instruction with EAX=0x80000001, EDX register.
- **Description**: The `cpuid_x86` struct is designed to encapsulate the functionality of querying and interpreting CPU feature flags on x86 architecture processors. It uses the CPUID instruction to gather information about the processor's capabilities, such as supported instruction sets and features, and stores this information in various bitsets. The struct also identifies the CPU vendor and brand, and provides boolean methods to check for specific features like SSE, AVX, and others. This struct is particularly useful for determining the capabilities of the CPU at runtime, allowing software to optimize its execution path based on available hardware features.
- **Member Functions**:
    - [`cpuid_x86::SSE3`](#cpuid_x86SSE3)
    - [`cpuid_x86::PCLMULQDQ`](#cpuid_x86PCLMULQDQ)
    - [`cpuid_x86::MONITOR`](#cpuid_x86MONITOR)
    - [`cpuid_x86::SSSE3`](#cpuid_x86SSSE3)
    - [`cpuid_x86::FMA`](#cpuid_x86FMA)
    - [`cpuid_x86::CMPXCHG16B`](#cpuid_x86CMPXCHG16B)
    - [`cpuid_x86::SSE41`](#cpuid_x86SSE41)
    - [`cpuid_x86::SSE42`](#cpuid_x86SSE42)
    - [`cpuid_x86::MOVBE`](#cpuid_x86MOVBE)
    - [`cpuid_x86::POPCNT`](#cpuid_x86POPCNT)
    - [`cpuid_x86::AES`](#cpuid_x86AES)
    - [`cpuid_x86::XSAVE`](#cpuid_x86XSAVE)
    - [`cpuid_x86::OSXSAVE`](#cpuid_x86OSXSAVE)
    - [`cpuid_x86::AVX`](#cpuid_x86AVX)
    - [`cpuid_x86::F16C`](#cpuid_x86F16C)
    - [`cpuid_x86::RDRAND`](#cpuid_x86RDRAND)
    - [`cpuid_x86::MSR`](#cpuid_x86MSR)
    - [`cpuid_x86::CX8`](#cpuid_x86CX8)
    - [`cpuid_x86::SEP`](#cpuid_x86SEP)
    - [`cpuid_x86::CMOV`](#cpuid_x86CMOV)
    - [`cpuid_x86::CLFSH`](#cpuid_x86CLFSH)
    - [`cpuid_x86::MMX`](#cpuid_x86MMX)
    - [`cpuid_x86::FXSR`](#cpuid_x86FXSR)
    - [`cpuid_x86::SSE`](#cpuid_x86SSE)
    - [`cpuid_x86::SSE2`](#cpuid_x86SSE2)
    - [`cpuid_x86::FSGSBASE`](#cpuid_x86FSGSBASE)
    - [`cpuid_x86::BMI1`](#cpuid_x86BMI1)
    - [`cpuid_x86::HLE`](#cpuid_x86HLE)
    - [`cpuid_x86::AVX2`](#cpuid_x86AVX2)
    - [`cpuid_x86::BMI2`](#cpuid_x86BMI2)
    - [`cpuid_x86::ERMS`](#cpuid_x86ERMS)
    - [`cpuid_x86::INVPCID`](#cpuid_x86INVPCID)
    - [`cpuid_x86::RTM`](#cpuid_x86RTM)
    - [`cpuid_x86::AVX512F`](#cpuid_x86AVX512F)
    - [`cpuid_x86::AVX512DQ`](#cpuid_x86AVX512DQ)
    - [`cpuid_x86::RDSEED`](#cpuid_x86RDSEED)
    - [`cpuid_x86::ADX`](#cpuid_x86ADX)
    - [`cpuid_x86::AVX512PF`](#cpuid_x86AVX512PF)
    - [`cpuid_x86::AVX512ER`](#cpuid_x86AVX512ER)
    - [`cpuid_x86::AVX512CD`](#cpuid_x86AVX512CD)
    - [`cpuid_x86::AVX512BW`](#cpuid_x86AVX512BW)
    - [`cpuid_x86::AVX512VL`](#cpuid_x86AVX512VL)
    - [`cpuid_x86::SHA`](#cpuid_x86SHA)
    - [`cpuid_x86::PREFETCHWT1`](#cpuid_x86PREFETCHWT1)
    - [`cpuid_x86::LAHF`](#cpuid_x86LAHF)
    - [`cpuid_x86::LZCNT`](#cpuid_x86LZCNT)
    - [`cpuid_x86::ABM`](#cpuid_x86ABM)
    - [`cpuid_x86::SSE4a`](#cpuid_x86SSE4a)
    - [`cpuid_x86::XOP`](#cpuid_x86XOP)
    - [`cpuid_x86::TBM`](#cpuid_x86TBM)
    - [`cpuid_x86::SYSCALL`](#cpuid_x86SYSCALL)
    - [`cpuid_x86::MMXEXT`](#cpuid_x86MMXEXT)
    - [`cpuid_x86::RDTSCP`](#cpuid_x86RDTSCP)
    - [`cpuid_x86::_3DNOWEXT`](#cpuid_x86_3DNOWEXT)
    - [`cpuid_x86::_3DNOW`](#cpuid_x86_3DNOW)
    - [`cpuid_x86::AVX512_VBMI`](#cpuid_x86AVX512_VBMI)
    - [`cpuid_x86::AVX512_VNNI`](#cpuid_x86AVX512_VNNI)
    - [`cpuid_x86::AVX512_FP16`](#cpuid_x86AVX512_FP16)
    - [`cpuid_x86::AVX512_BF16`](#cpuid_x86AVX512_BF16)
    - [`cpuid_x86::AVX_VNNI`](#cpuid_x86AVX_VNNI)
    - [`cpuid_x86::AMX_TILE`](#cpuid_x86AMX_TILE)
    - [`cpuid_x86::AMX_INT8`](#cpuid_x86AMX_INT8)
    - [`cpuid_x86::AMX_FP16`](#cpuid_x86AMX_FP16)
    - [`cpuid_x86::AMX_BF16`](#cpuid_x86AMX_BF16)
    - [`cpuid_x86::cpuid`](#cpuid_x86cpuid)
    - [`cpuid_x86::cpuidex`](#cpuid_x86cpuidex)
    - [`cpuid_x86::cpuid`](#cpuid_x86cpuid)
    - [`cpuid_x86::cpuidex`](#cpuid_x86cpuidex)
    - [`cpuid_x86::cpuid_x86`](#cpuid_x86cpuid_x86)

**Methods**

---
#### cpuid\_x86::SSE3<!-- {{#callable:cpuid_x86::SSE3}} -->
The `SSE3` function checks if the SSE3 instruction set is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the first bit of the `f_1_ecx` bitset, which indicates the availability of the SSE3 instruction set.
    - It returns the boolean value of that bit, which is either true (supported) or false (not supported).
- **Output**: The function returns a boolean value indicating whether the SSE3 instruction set is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::PCLMULQDQ<!-- {{#callable:cpuid_x86::PCLMULQDQ}} -->
The `PCLMULQDQ` function checks if the PCLMUL instruction is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the second bit of the `f_1_ecx` bitset, which represents the support for the PCLMULQDQ instruction.
    - It returns the boolean value of that specific bit, indicating whether the instruction is supported.
- **Output**: The function returns a boolean value: `true` if the PCLMULQDQ instruction is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::MONITOR<!-- {{#callable:cpuid_x86::MONITOR}} -->
The `MONITOR` function checks the status of the MONITOR instruction support in the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `f_1_ecx` bitset, specifically the bit at index 3, which indicates the availability of the MONITOR instruction.
    - It returns the boolean value of that specific bit, which is either true (supported) or false (not supported).
- **Output**: The output is a boolean value indicating whether the MONITOR instruction is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SSSE3<!-- {{#callable:cpuid_x86::SSSE3}} -->
The `SSSE3` function checks if the SSSE3 (Supplemental Streaming SIMD Extensions 3) instruction set is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 9th bit of the `f_1_ecx` bitset, which represents the availability of the SSSE3 instruction set.
    - It returns the boolean value of that specific bit, indicating whether SSSE3 is supported (true) or not (false).
- **Output**: The output is a boolean value indicating the support status of the SSSE3 instruction set.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::FMA<!-- {{#callable:cpuid_x86::FMA}} -->
The `FMA` function checks and returns the support status of the Fused Multiply-Add instruction set feature.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 12th bit of the `f_1_ecx` bitset, which represents the availability of the FMA feature.
    - It returns the boolean value of that specific bit, indicating whether FMA is supported or not.
- **Output**: The output is a boolean value: `true` if the FMA feature is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::CMPXCHG16B<!-- {{#callable:cpuid_x86::CMPXCHG16B}} -->
The `CMPXCHG16B` function checks if the CMPXCHG16B instruction is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 13th bit of the `f_1_ecx` bitset to determine support for the CMPXCHG16B instruction.
- **Output**: Returns a boolean value indicating whether the CMPXCHG16B instruction is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SSE41<!-- {{#callable:cpuid_x86::SSE41}} -->
The `SSE41` function checks if the SSE4.1 instruction set is supported by the CPU.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the 19th bit of the `f_1_ecx` bitset, which represents the availability of the SSE4.1 instruction set.
    - It returns the value of that bit as a boolean, indicating support (true) or lack of support (false) for SSE4.1.
- **Output**: The function returns a boolean value: true if SSE4.1 is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SSE42<!-- {{#callable:cpuid_x86::SSE42}} -->
The `SSE42` function checks if the SSE4.2 instruction set is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 20th bit of the `f_1_ecx` bitset, which represents the availability of the SSE4.2 feature.
    - It returns the value of that bit as a boolean, indicating whether SSE4.2 is supported (true) or not (false).
- **Output**: The function returns a boolean value: true if SSE4.2 is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::MOVBE<!-- {{#callable:cpuid_x86::MOVBE}} -->
The `MOVBE` function checks and returns the status of the MOVBE instruction support in the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 22nd bit of the `f_1_ecx` bitset, which represents the MOVBE instruction support.
    - It returns the boolean value of that specific bit, indicating whether MOVBE is supported or not.
- **Output**: The output is a boolean value: true if the MOVBE instruction is supported by the CPU, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::POPCNT<!-- {{#callable:cpuid_x86::POPCNT}} -->
The `POPCNT` function checks if the POPCNT instruction is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 23rd bit of the `f_1_ecx` bitset, which represents the availability of the POPCNT instruction.
    - It returns the boolean value of that specific bit, indicating whether the instruction is supported.
- **Output**: The function returns a boolean value: true if the POPCNT instruction is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AES<!-- {{#callable:cpuid_x86::AES}} -->
Checks if the AES instruction set is supported by the CPU.
- **Inputs**:
    - `void`: This function does not take any input parameters.
- **Control Flow**:
    - The function directly accesses the `f_1_ecx` bitset at index 25 to determine the availability of the AES instruction set.
    - It returns the boolean value stored at that index, which indicates whether AES is supported.
- **Output**: Returns a boolean value: true if the AES instruction set is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::XSAVE<!-- {{#callable:cpuid_x86::XSAVE}} -->
The `XSAVE` function checks if the XSAVE instruction is supported by the CPU.
- **Inputs**:
    - `void`: The function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the 26th bit of the `f_1_ecx` bitset, which indicates the support for the XSAVE instruction.
    - It returns the value of that bit as a boolean, indicating whether XSAVE is supported (true) or not (false).
- **Output**: The function returns a boolean value indicating the support status of the XSAVE instruction.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::OSXSAVE<!-- {{#callable:cpuid_x86::OSXSAVE}} -->
The `OSXSAVE` function checks if the OS supports the OSXSAVE feature of the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 27th bit of the `f_1_ecx` bitset to determine the availability of the OSXSAVE feature.
    - It returns the boolean value of that specific bit, indicating whether OSXSAVE is supported.
- **Output**: The function returns a boolean value: `true` if the OSXSAVE feature is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX<!-- {{#callable:cpuid_x86::AVX}} -->
The `AVX` function checks if the AVX (Advanced Vector Extensions) feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 28th bit of the `f_1_ecx` bitset, which represents the availability of the AVX feature.
    - It returns the boolean value of that specific bit, indicating whether AVX is supported or not.
- **Output**: The function returns a boolean value: `true` if AVX is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::F16C<!-- {{#callable:cpuid_x86::F16C}} -->
The `F16C` function checks if the F16C instruction set feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 29th bit of the `f_1_ecx` bitset, which represents the availability of the F16C instruction set feature.
    - It returns the boolean value of that specific bit, indicating whether the feature is supported.
- **Output**: The function returns a boolean value: `true` if the F16C feature is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::RDRAND<!-- {{#callable:cpuid_x86::RDRAND}} -->
The `RDRAND` function checks if the RDRAND instruction is supported by the CPU.
- **Inputs**:
    - `void`: The function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the 30th bit of the `f_1_ecx` bitset.
    - It returns the value of that bit, which indicates the availability of the RDRAND instruction.
- **Output**: The function returns a boolean value: true if the RDRAND instruction is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::MSR<!-- {{#callable:cpuid_x86::MSR}} -->
The `MSR` function checks and returns the status of the MSR (Model Specific Register) feature in the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 5th bit of the `f_1_edx` bitset, which represents the MSR feature.
    - It returns the boolean value of that specific bit, indicating whether the MSR feature is supported or not.
- **Output**: The output is a boolean value: `true` if the MSR feature is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::CX8<!-- {{#callable:cpuid_x86::CX8}} -->
The `CX8` function checks if the CPU supports the `CX8` instruction set feature.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 8th bit of the `f_1_edx` bitset, which represents the `CX8` feature.
    - It returns the boolean value of that specific bit, indicating the presence or absence of the `CX8` feature.
- **Output**: The function returns a boolean value: `true` if the `CX8` feature is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SEP<!-- {{#callable:cpuid_x86::SEP}} -->
The `SEP` function checks the status of the SEP (Streaming SIMD Extensions) feature in the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 11th bit of the `f_1_edx` bitset to determine if the SEP feature is supported.
- **Output**: The function returns a boolean value indicating whether the SEP feature is enabled (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::CMOV<!-- {{#callable:cpuid_x86::CMOV}} -->
The `CMOV` function checks the status of the conditional move instruction support in the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 15th bit of the `f_1_edx` bitset, which represents the availability of the `CMOV` instruction.
    - It returns the boolean value of that specific bit, indicating whether the `CMOV` instruction is supported.
- **Output**: The function returns a boolean value: `true` if the `CMOV` instruction is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::CLFSH<!-- {{#callable:cpuid_x86::CLFSH}} -->
The `CLFSH` function returns the value of the 19th bit in the `f_1_edx` bitset, indicating support for the CLFLUSH instruction.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 19th bit of the `f_1_edx` bitset.
    - It returns the boolean value of that specific bit.
- **Output**: The output is a boolean value indicating whether the CLFLUSH instruction is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::MMX<!-- {{#callable:cpuid_x86::MMX}} -->
The `MMX` function checks if the MMX instruction set is supported by the CPU.
- **Inputs**:
    - `void`: The function does not take any input parameters.
- **Control Flow**:
    - The function directly accesses the 23rd bit of the `f_1_edx` bitset.
    - It returns the value of that bit, which indicates the support for the MMX instruction set.
- **Output**: The function returns a boolean value: true if MMX is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::FXSR<!-- {{#callable:cpuid_x86::FXSR}} -->
The `FXSR` function checks if the FXSR (Fast Floating Point Save and Restore) feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 24th bit of the `f_1_edx` bitset, which represents the FXSR feature flag.
    - It returns the value of that bit, indicating whether FXSR is supported (true) or not (false).
- **Output**: The output is a boolean value indicating the presence of the FXSR feature in the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SSE<!-- {{#callable:cpuid_x86::SSE}} -->
The `SSE` function checks if the SSE (Streaming SIMD Extensions) feature is supported by returning the value of the 25th bit in the `f_1_edx` bitset.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 25th bit of the `f_1_edx` bitset, which represents the availability of the SSE feature.
    - It returns the boolean value of that specific bit.
- **Output**: The output is a boolean value indicating whether the SSE feature is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SSE2<!-- {{#callable:cpuid_x86::SSE2}} -->
The `SSE2` function checks if the SSE2 instruction set is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 26th bit of the `f_1_edx` bitset, which represents the availability of the SSE2 feature.
    - It returns the boolean value of that specific bit, indicating whether SSE2 is supported (true) or not (false).
- **Output**: The function returns a boolean value: true if SSE2 is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::FSGSBASE<!-- {{#callable:cpuid_x86::FSGSBASE}} -->
The `FSGSBASE` function checks and returns the status of the FSGSBASE CPU feature.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the first bit of the `f_7_ebx` bitset, which indicates the availability of the FSGSBASE feature.
    - It returns the boolean value of that specific bit.
- **Output**: The output is a boolean value indicating whether the FSGSBASE feature is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::BMI1<!-- {{#callable:cpuid_x86::BMI1}} -->
The `BMI1` function checks and returns the status of the BMI1 feature from the CPU's feature set.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `f_7_ebx` bitset to retrieve the value at index 3, which corresponds to the BMI1 feature.
    - It returns the boolean value found at that index, indicating whether the BMI1 feature is supported by the CPU.
- **Output**: The output is a boolean value that indicates the presence (true) or absence (false) of the BMI1 feature in the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::HLE<!-- {{#callable:cpuid_x86::HLE}} -->
Determines if Hardware Lock Elision (HLE) is supported by the CPU.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function checks the `is_intel` boolean to determine if the CPU is from Intel.
    - It then checks the fifth bit of the `f_7_ebx` bitset to see if HLE is supported.
    - The function returns true if both conditions are met, otherwise it returns false.
- **Output**: Returns a boolean value indicating whether HLE is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX2<!-- {{#callable:cpuid_x86::AVX2}} -->
Checks if the AVX2 instruction set is supported by the CPU.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the `f_7_ebx` bitset to retrieve the value at index 5.
    - The value at this index indicates whether the AVX2 feature is supported (true) or not (false).
- **Output**: Returns a boolean value indicating the presence of the AVX2 instruction set support.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::BMI2<!-- {{#callable:cpuid_x86::BMI2}} -->
The `BMI2` function checks if the BMI2 (Bit Manipulation Instruction Set 2) feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 8th bit of the `f_7_ebx` bitset, which represents the availability of the BMI2 feature.
    - It returns the boolean value of that specific bit, indicating whether BMI2 is supported (true) or not (false).
- **Output**: The function returns a boolean value: true if the BMI2 feature is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::ERMS<!-- {{#callable:cpuid_x86::ERMS}} -->
The `ERMS` function returns the value of the 9th bit in the `f_7_ebx` bitset, indicating support for Enhanced REP MOVSB/STOSB.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 9th bit of the `f_7_ebx` bitset.
    - It returns the boolean value of that bit, which indicates whether the feature is supported.
- **Output**: The output is a boolean value that signifies if the Enhanced REP MOVSB/STOSB feature is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::INVPCID<!-- {{#callable:cpuid_x86::INVPCID}} -->
The `INVPCID` function checks the status of the INVPCID CPU feature.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 10th bit of the `f_7_ebx` bitset to determine if the INVPCID feature is supported.
- **Output**: The function returns a boolean value indicating whether the INVPCID feature is enabled (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::RTM<!-- {{#callable:cpuid_x86::RTM}} -->
The `RTM` function checks if the CPU is an Intel processor and if the RTM (Restricted Transactional Memory) feature is supported.
- **Inputs**:
    - `void`: The function does not take any input arguments.
- **Control Flow**:
    - The function evaluates the boolean expression `is_intel && f_7_ebx[11]`.
    - If `is_intel` is true and the 11th bit of `f_7_ebx` is set, the function returns true, indicating RTM support.
    - Otherwise, it returns false.
- **Output**: The function returns a boolean value indicating whether the RTM feature is supported on the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512F<!-- {{#callable:cpuid_x86::AVX512F}} -->
Checks if the AVX512F feature is supported by the CPU.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the 16th bit of the `f_7_ebx` bitset to determine if the AVX512F feature is supported.
    - It returns the boolean value of that specific bit.
- **Output**: Returns a boolean value indicating whether the AVX512F feature is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512DQ<!-- {{#callable:cpuid_x86::AVX512DQ}} -->
The `AVX512DQ` function checks if the AVX512DQ feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 17th bit of the `f_7_ebx` bitset to determine the availability of the AVX512DQ feature.
    - It returns the boolean value of that specific bit, indicating whether the feature is supported.
- **Output**: The function returns a boolean value: `true` if the AVX512DQ feature is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::RDSEED<!-- {{#callable:cpuid_x86::RDSEED}} -->
The `RDSEED` function checks the availability of the RDSEED instruction by returning the value of the 18th bit in the `f_7_ebx` bitset.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 18th bit of the `f_7_ebx` bitset, which is a member of the `cpuid_x86` structure.
    - It returns the boolean value of that specific bit, indicating whether the RDSEED instruction is supported.
- **Output**: The output is a boolean value that indicates the presence (true) or absence (false) of the RDSEED instruction support in the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::ADX<!-- {{#callable:cpuid_x86::ADX}} -->
The `ADX` function checks if the ADX (Multi-Precision Add-Carry Instruction) feature is supported by the CPU.
- **Inputs**:
    - `void`: The function does not take any input parameters.
- **Control Flow**:
    - The function directly accesses the 19th bit of the `f_7_ebx` bitset, which represents the ADX feature flag.
    - It returns the value of this bit, indicating whether ADX is supported (true) or not (false).
- **Output**: The function returns a boolean value: true if the ADX feature is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512PF<!-- {{#callable:cpuid_x86::AVX512PF}} -->
The `AVX512PF` function checks if the AVX512 Prefetch feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 26th bit of the `f_7_ebx` bitset to determine the availability of the AVX512 Prefetch feature.
- **Output**: The function returns a boolean value indicating whether the AVX512 Prefetch feature is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512ER<!-- {{#callable:cpuid_x86::AVX512ER}} -->
The `AVX512ER` function checks and returns the availability of the AVX512ER instruction set feature.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 27th bit of the `f_7_ebx` bitset to determine if the AVX512ER feature is supported.
    - It returns the boolean value of that specific bit.
- **Output**: The output is a boolean value indicating whether the AVX512ER feature is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512CD<!-- {{#callable:cpuid_x86::AVX512CD}} -->
The `AVX512CD` function checks and returns the availability of the AVX512CD CPU feature.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 28th bit of the `f_7_ebx` bitset, which represents the AVX512CD feature flag.
    - It returns the boolean value of that specific bit, indicating whether the feature is supported or not.
- **Output**: The output is a boolean value: `true` if the AVX512CD feature is supported by the CPU, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512BW<!-- {{#callable:cpuid_x86::AVX512BW}} -->
The `AVX512BW` function checks if the AVX512BW instruction set feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 30th bit of the `f_7_ebx` bitset to determine the availability of the AVX512BW feature.
- **Output**: The function returns a boolean value indicating whether the AVX512BW feature is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512VL<!-- {{#callable:cpuid_x86::AVX512VL}} -->
The `AVX512VL` function checks if the AVX512 Vector Length extension is supported by the CPU.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the 31st bit of the `f_7_ebx` bitset.
    - It returns the value of that bit, which indicates the support for AVX512 Vector Length.
- **Output**: The function returns a boolean value: true if AVX512 Vector Length is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SHA<!-- {{#callable:cpuid_x86::SHA}} -->
The `SHA` function checks the availability of the SHA instruction set feature in the CPU.
- **Inputs**:
    - `void`: The function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the 29th bit of the `f_7_ebx` bitset, which represents the availability of the SHA instruction set feature.
    - It returns the boolean value of that specific bit.
- **Output**: The function returns a boolean value indicating whether the SHA instruction set feature is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::PREFETCHWT1<!-- {{#callable:cpuid_x86::PREFETCHWT1}} -->
The `PREFETCHWT1` function checks the availability of the PREFETCHWT1 instruction on the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the first bit in the `f_7_ecx` bitset, which indicates the support for the PREFETCHWT1 instruction.
- **Output**: The output is a boolean value indicating whether the PREFETCHWT1 instruction is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::LAHF<!-- {{#callable:cpuid_x86::LAHF}} -->
The `LAHF` function checks the availability of the LAHF instruction by returning the value of the first bit in the `f_81_ecx` bitset.
- **Inputs**:
    - `void`: The function takes no input arguments.
- **Control Flow**:
    - The function directly accesses the first bit of the `f_81_ecx` bitset, which represents the availability of the LAHF instruction.
    - It returns the boolean value of that bit.
- **Output**: The function returns a boolean value indicating whether the LAHF instruction is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::LZCNT<!-- {{#callable:cpuid_x86::LZCNT}} -->
The `LZCNT` function checks if the CPU supports the LZCNT instruction for Intel processors.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function evaluates the boolean expression `is_intel && f_81_ecx[5]`.
    - It checks if the CPU is an Intel processor and if the LZCNT feature is enabled in the `f_81_ecx` bitset.
- **Output**: The function returns a boolean value indicating whether the LZCNT instruction is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::ABM<!-- {{#callable:cpuid_x86::ABM}} -->
The `ABM` function checks if the CPU is an AMD processor and if the ABM feature is supported.
- **Inputs**:
    - `void`: The function takes no input arguments.
- **Control Flow**:
    - The function evaluates the boolean expression `is_amd && f_81_ecx[5]`.
    - It returns `true` if both conditions are satisfied, indicating that the CPU is an AMD processor and the ABM feature is supported; otherwise, it returns `false`.
- **Output**: The function returns a boolean value indicating whether the ABM feature is supported on an AMD CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SSE4a<!-- {{#callable:cpuid_x86::SSE4a}} -->
Checks if the SSE4a instruction set is supported on AMD processors.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function evaluates the boolean variable `is_amd` to check if the CPU is an AMD processor.
    - It then checks the 6th bit of the `f_81_ecx` bitset to determine if SSE4a is supported.
- **Output**: Returns a boolean value indicating whether the SSE4a instruction set is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::XOP<!-- {{#callable:cpuid_x86::XOP}} -->
The `XOP` function checks if the CPU is an AMD processor and if the XOP instruction set is supported.
- **Inputs**:
    - `void`: The function does not take any input arguments.
- **Control Flow**:
    - The function evaluates the boolean expression `is_amd && f_81_ecx[11]`.
    - It returns `true` if both conditions are satisfied, indicating that the CPU is an AMD processor and supports the XOP instruction set; otherwise, it returns `false`.
- **Output**: The function returns a boolean value indicating whether the XOP instruction set is supported by the AMD CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::TBM<!-- {{#callable:cpuid_x86::TBM}} -->
Checks if the CPU supports the `TBM` (Trailing Bit Manipulation) instruction set for AMD processors.
- **Inputs**:
    - `void`: This function does not take any input parameters.
- **Control Flow**:
    - The function evaluates two conditions: whether the CPU is an AMD processor (`is_amd`) and whether the 21st bit of the `f_81_ecx` bitset is set.
    - If both conditions are true, the function returns `true`; otherwise, it returns `false`.
- **Output**: Returns a boolean value indicating the support for the `TBM` instruction set.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::SYSCALL<!-- {{#callable:cpuid_x86::SYSCALL}} -->
The `SYSCALL` function checks if the CPU is an Intel processor and if the SYSCALL feature is supported.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the boolean variable `is_intel` to determine if the CPU is from Intel.
    - It then checks the 11th bit of the `f_81_edx` bitset to see if the SYSCALL feature is supported.
- **Output**: The function returns a boolean value indicating whether the SYSCALL feature is supported on an Intel CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::MMXEXT<!-- {{#callable:cpuid_x86::MMXEXT}} -->
Checks if the MMX Extended instruction set is supported on AMD processors.
- **Inputs**:
    - `void`: This function does not take any input parameters.
- **Control Flow**:
    - The function evaluates the boolean variable `is_amd` to check if the CPU is an AMD processor.
    - It then checks the 22nd bit of the `f_81_edx` bitset to determine if the MMX Extended feature is supported.
- **Output**: Returns a boolean value indicating whether the MMX Extended instruction set is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::RDTSCP<!-- {{#callable:cpuid_x86::RDTSCP}} -->
The `RDTSCP` function checks if the CPU supports the RDTSCP instruction for Intel processors.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function evaluates the boolean variable `is_intel` to determine if the CPU is an Intel processor.
    - It then checks the 27th bit of the `f_81_edx` bitset to see if the RDTSCP instruction is supported.
- **Output**: The function returns a boolean value indicating whether the RDTSCP instruction is supported on the Intel CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::\_3DNOWEXT<!-- {{#callable:cpuid_x86::_3DNOWEXT}} -->
Checks if the CPU supports the 3DNow! Extended instruction set.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function evaluates the boolean expression 'is_amd && f_81_edx[30]'.
    - It checks if the CPU is an AMD processor and if the 3DNow! Extended feature is supported based on the corresponding bit in the feature flags.
- **Output**: Returns a boolean value indicating whether the 3DNow! Extended instruction set is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::\_3DNOW<!-- {{#callable:cpuid_x86::_3DNOW}} -->
The `_3DNOW` function checks if the CPU supports the 3DNow! instruction set, specifically for AMD processors.
- **Inputs**: None
- **Control Flow**:
    - The function evaluates the boolean variable `is_amd` to determine if the CPU is an AMD processor.
    - It then checks the 31st bit of the `f_81_edx` bitset to see if the 3DNow! feature is supported.
- **Output**: The function returns a boolean value indicating whether the 3DNow! instruction set is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512\_VBMI<!-- {{#callable:cpuid_x86::AVX512_VBMI}} -->
Checks if the AVX512 Vector Bit Manipulation Instructions (VBMI) feature is supported by the CPU.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the `f_7_ecx` bitset, specifically the second bit (index 1).
    - It returns the value of that bit, which indicates the availability of the AVX512 VBMI feature.
- **Output**: Returns a boolean value: true if the AVX512 VBMI feature is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512\_VNNI<!-- {{#callable:cpuid_x86::AVX512_VNNI}} -->
Checks if the AVX512 VNNI instruction set is supported by the CPU.
- **Inputs**:
    - `void`: This function does not take any input arguments.
- **Control Flow**:
    - The function directly accesses the 11th bit of the `f_7_ecx` bitset.
    - It returns the boolean value of that specific bit, indicating support for AVX512 VNNI.
- **Output**: Returns a boolean value: true if AVX512 VNNI is supported, false otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512\_FP16<!-- {{#callable:cpuid_x86::AVX512_FP16}} -->
The `AVX512_FP16` function checks if the AVX512 FP16 feature is supported by returning the value of the 23rd bit in the `f_7_edx` bitset.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 23rd bit of the `f_7_edx` bitset, which represents the availability of the AVX512 FP16 feature.
    - It returns the boolean value of that specific bit.
- **Output**: The output is a boolean value indicating whether the AVX512 FP16 feature is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX512\_BF16<!-- {{#callable:cpuid_x86::AVX512_BF16}} -->
The `AVX512_BF16` function checks if the AVX512 BF16 instruction set feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the fifth element of the `f_7_1_eax` bitset, which represents the availability of the AVX512 BF16 feature.
    - It returns the boolean value of that specific bit, indicating whether the feature is supported.
- **Output**: The function returns a boolean value: `true` if the AVX512 BF16 feature is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AVX\_VNNI<!-- {{#callable:cpuid_x86::AVX_VNNI}} -->
The `AVX_VNNI` function checks if the AVX Vector Neural Network Instructions (VNNI) feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the fifth bit of the `f_7_1_eax` bitset, which represents the availability of the AVX VNNI feature.
    - It returns the boolean value of that specific bit.
- **Output**: The function returns a boolean value indicating whether the AVX VNNI feature is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AMX\_TILE<!-- {{#callable:cpuid_x86::AMX_TILE}} -->
The `AMX_TILE` function checks the availability of the AMX (Advanced Matrix Extensions) tile feature on the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 24th bit of the `f_7_edx` bitset, which represents the AMX tile feature availability.
    - It returns the boolean value of that specific bit, indicating whether the feature is supported or not.
- **Output**: The function returns a boolean value: `true` if the AMX tile feature is supported, and `false` otherwise.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AMX\_INT8<!-- {{#callable:cpuid_x86::AMX_INT8}} -->
The `AMX_INT8` function checks if the AMX (Advanced Matrix Extensions) INT8 feature is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 25th bit of the `f_7_edx` bitset, which represents the AMX INT8 feature flag.
    - It returns the boolean value of that specific bit, indicating whether the feature is supported (true) or not (false).
- **Output**: The output is a boolean value indicating the presence of the AMX INT8 feature in the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AMX\_FP16<!-- {{#callable:cpuid_x86::AMX_FP16}} -->
The `AMX_FP16` function returns the value of the 21st bit in the `f_7_1_eax` bitset, indicating support for AMX FP16 instructions.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 21st bit of the `f_7_1_eax` bitset.
    - It returns the boolean value of that specific bit.
- **Output**: The output is a boolean value indicating whether the AMX FP16 feature is supported by the CPU.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::AMX\_BF16<!-- {{#callable:cpuid_x86::AMX_BF16}} -->
The `AMX_BF16` function checks the availability of the AMX BF16 feature in the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the 22nd bit of the `f_7_edx` bitset to determine if the AMX BF16 feature is supported.
- **Output**: The function returns a boolean value indicating whether the AMX BF16 feature is supported (true) or not (false).
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::cpuid<!-- {{#callable:cpuid_x86::cpuid}} -->
The `cpuid` function retrieves CPU information based on the specified `eax` value and stores it in the provided `cpu_info` array.
- **Inputs**:
    - `cpu_info`: An array of four integers where the CPU information will be stored.
    - `eax`: An integer representing the function identifier for the `cpuid` instruction.
- **Control Flow**:
    - The function calls the intrinsic `__cpuid` which directly accesses the CPU's CPUID instruction.
    - The `cpu_info` array is populated with the results of the CPUID instruction based on the value of `eax`.
- **Output**: The function does not return a value; instead, it populates the `cpu_info` array with CPU feature information.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::cpuidex<!-- {{#callable:cpuid_x86::cpuidex}} -->
The `cpuidex` function retrieves CPU information based on the specified `eax` and `ecx` values.
- **Inputs**:
    - `cpu_info`: An array of four integers that will be filled with CPU information.
    - `eax`: An integer that specifies the function ID for the CPUID instruction.
    - `ecx`: An integer that specifies the sub-function ID for the CPUID instruction.
- **Control Flow**:
    - The function calls the intrinsic `__cpuidex`, which is a compiler-specific function that executes the CPUID instruction.
    - The `cpu_info` array is populated with the results of the CPUID instruction based on the provided `eax` and `ecx` values.
- **Output**: The function does not return a value; instead, it fills the `cpu_info` array with CPU feature information.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::cpuid<!-- {{#callable:cpuid_x86::cpuid}} -->
The `cpuid` function retrieves CPU information by executing the CPUID instruction and stores the results in the provided array.
- **Inputs**:
    - `cpu_info`: An array of four integers where the CPU information will be stored after executing the CPUID instruction.
    - `eax`: An integer representing the function identifier for the CPUID instruction, which specifies what information to retrieve.
- **Control Flow**:
    - The function uses inline assembly to execute the CPUID instruction.
    - The results of the CPUID instruction are stored in the `cpu_info` array, with each register's value being assigned to a specific index in the array.
    - The `eax` input argument is used to specify which information to retrieve from the CPU.
- **Output**: The function does not return a value; instead, it populates the `cpu_info` array with the results of the CPUID instruction.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::cpuidex<!-- {{#callable:cpuid_x86::cpuidex}} -->
The `cpuidex` function retrieves CPU information based on the specified `eax` and `ecx` values using the `cpuid` instruction.
- **Inputs**:
    - `cpu_info`: An array of four integers where the CPU information will be stored.
    - `eax`: The value to be placed in the `eax` register, which specifies the function to be executed by the `cpuid` instruction.
    - `ecx`: The value to be placed in the `ecx` register, which can modify the behavior of the `cpuid` instruction.
- **Control Flow**:
    - The function uses inline assembly to execute the `cpuid` instruction.
    - The output registers are specified to store the results in the `cpu_info` array.
    - The `eax` and `ecx` inputs are passed to the `cpuid` instruction to determine the specific information to retrieve.
- **Output**: The function does not return a value; instead, it populates the `cpu_info` array with the results of the `cpuid` instruction, which includes various CPU features and information.
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)


---
#### cpuid\_x86::cpuid\_x86<!-- {{#callable:cpuid_x86::cpuid_x86}} -->
The `cpuid_x86` constructor retrieves and stores CPU information such as vendor, brand, and feature flags using the [`cpuid`](#cpuid_x86cpuid) and [`cpuidex`](#cpuid_x86cpuidex) assembly instructions.
- **Inputs**: None
- **Control Flow**:
    - Calls [`cpuid`](#cpuid_x86cpuid) with 0 to determine the highest valid function ID and stores the result in `n_ids`.
    - Iterates from 0 to `n_ids`, calling [`cpuidex`](#cpuid_x86cpuidex) to gather CPU information and stores it in the `data` vector.
    - Extracts the vendor string from the first entry of `data` and checks if it corresponds to Intel or AMD.
    - If valid, loads feature flags from `data` for functions 0x00000001 and 0x00000007 into respective bitsets.
    - Calls [`cpuid`](#cpuid_x86cpuid) with 0x80000000 to find the highest extended function ID and stores it in `n_ex_ids`.
    - Iterates from 0x80000000 to `n_ex_ids`, calling [`cpuidex`](#cpuid_x86cpuidex) to gather extended CPU information into `ext_data`.
    - Loads additional feature flags from `ext_data` for function 0x80000001 and interprets the CPU brand string if available.
- **Output**: The constructor initializes the `vendor`, `brand`, and various feature flags as bitsets, indicating the capabilities of the CPU.
- **Functions called**:
    - [`cpuid_x86::cpuid`](#cpuid_x86cpuid)
    - [`cpuid_x86::cpuidex`](#cpuid_x86cpuidex)
- **See also**: [`cpuid_x86`](#cpuid_x86)  (Data Structure)



# Functions

---
### test\_x86\_is<!-- {{#callable:test_x86_is}} -->
The `test_x86_is` function prints the CPU vendor, brand, and various feature support flags for an x86 CPU using the `cpuid_x86` class.
- **Inputs**: None
- **Control Flow**:
    - Instantiate a `cpuid_x86` object named `is`.
    - Print the CPU vendor and brand using `is.vendor` and `is.brand`.
    - Print whether the CPU is Intel or AMD using `is.is_intel` and `is.is_amd`.
    - Print the support status of various CPU features by calling corresponding methods on the `is` object, such as `is.SSE3()`, `is.PCLMULQDQ()`, etc.
- **Output**: The function outputs a series of printed statements to the console, detailing the CPU vendor, brand, and support for various CPU features.


---
### ggml\_backend\_cpu\_x86\_score<!-- {{#callable:ggml_backend_cpu_x86_score}} -->
The `ggml_backend_cpu_x86_score` function calculates a score based on the presence of specific CPU instruction set extensions on x86 architecture.
- **Inputs**: None
- **Control Flow**:
    - Initialize the score to 1.
    - Create an instance of `cpuid_x86` to check CPU features.
    - For each defined macro (e.g., `GGML_FMA`, `GGML_F16C`), check if the corresponding CPU feature is supported using the `cpuid_x86` instance.
    - If a required feature is not supported, return 0 immediately.
    - If a feature is supported, increment the score by a specific value (usually a power of 2).
    - Continue checking all defined features and updating the score accordingly.
    - Return the final score.
- **Output**: An integer score representing the presence of specific CPU instruction set extensions, or 0 if any required feature is missing.


