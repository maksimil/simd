workspace "simd"
   configurations { "Debug", "Release" }

project "simd"
   kind "ConsoleApp"
   language "C++"
   vectorextensions "AVX"
   files {"*.h", "*.cpp"}
   filter { "configurations:Debug" }
      defines { "DEBUG" }
      symbols "On"

   filter { "configurations:Release" }
      defines { "NDEBUG" }
      optimize "On"
