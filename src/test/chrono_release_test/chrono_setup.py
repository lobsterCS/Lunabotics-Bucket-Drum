"""
Solves error: ImportError: DLL load failed while importing _irrlicht: A dynamic link library (DLL)
initialization routine failed.
Solves issue where anaconda/Library/bin does not get added to PATH when importing irrlicht
"""
import os, sys

env = os.path.dirname(sys.executable)
dll_dirs = [
    os.path.join(env, "Library", "bin"),
    os.path.join(env, "Library", "mingw-w64", "bin"),
    os.path.join(env, "Library", "usr", "bin"),
    os.path.join(env, "bin"),
]
for d in dll_dirs:
    if os.path.isdir(d):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]