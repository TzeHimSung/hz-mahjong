"""Build the exact One-Ply Expectation leaf kernel with the platform compiler (Zig on Windows)."""
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OnePlyBuildExt(build_ext):
    def build_extension(self, extension):
        if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
            return super().build_extension(extension)
        target = Path(self.get_ext_fullpath(extension.name)).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        library = Path(sys.base_prefix) / "libs" / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
        command = [sys.executable, "-m", "ziglang", "cc", "-shared", "-O3",
                   "-target", "x86_64-windows-gnu", "-I", sysconfig.get_path("include"),
                   *extension.sources, str(library), "-o", str(target)]
        subprocess.run(command, check=True)


setup(
    ext_modules=[Extension("heuristic_ai.strategy.native._one_ply_native", ["src/heuristic_ai/strategy/native/_one_ply_native.c"],
                           extra_compile_args=["-O3"] if sys.platform != "win32" else [])],
    cmdclass={"build_ext": OnePlyBuildExt},
)
