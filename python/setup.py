import os
import re
import sys
import platform
import subprocess
import shutil
import glob
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        # Check for MinGW
        mingw_path = self._find_mingw()
        if not mingw_path:
            print("WARNING: MinGW not found in PATH. Build might fail.")
        else:
            print(f"Found MinGW at: {mingw_path}")
            # Add MinGW to the PATH if not already there
            mingw_bin = os.path.join(mingw_path, 'bin')
            if mingw_bin not in os.environ['PATH']:
                os.environ['PATH'] = mingw_bin + os.pathsep + os.environ['PATH']

        for ext in self.extensions:
            self.build_extension(ext)

    def _find_mingw(self):
        """Find MinGW in the PATH."""
        # Check common MinGW installation paths
        mingw_paths = []
        
        # Check PATH for mingw32-make.exe or make.exe
        for path in os.environ['PATH'].split(os.pathsep):
            mingw32_make = os.path.join(path, 'mingw32-make.exe')
            make = os.path.join(path, 'make.exe')
            gcc = os.path.join(path, 'gcc.exe')
            if os.path.exists(mingw32_make) or os.path.exists(make) or os.path.exists(gcc):
                # Found MinGW in PATH
                # Go up one directory to get the MinGW root (bin parent directory)
                if os.path.basename(path).lower() == 'bin':
                    return os.path.dirname(path)
                return path

        # Check common installation directories
        common_dirs = [
            'C:\\MinGW',
            'C:\\msys64\\mingw64',
            'C:\\msys64\\mingw32',
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'mingw-w64'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'CLion', 'bin', 'mingw'),
            'C:\\Program Files\\mingw-w64',
            'C:\\Program Files (x86)\\mingw-w64',
        ]
        
        for dir in common_dirs:
            if os.path.exists(dir):
                bin_dir = os.path.join(dir, 'bin')
                if os.path.exists(bin_dir):
                    return dir
        
        return None

    def _find_eigen(self, ext):
        """Find the Eigen library or download it if not present."""
        # Check for Eigen in standard locations
        eigen_dirs = [
            os.path.join(os.path.dirname(ext.sourcedir), 'packages', 'eigen-3.4.0'),
            os.path.join(os.path.dirname(ext.sourcedir), 'packages', 'eigen'),
            os.path.join(os.path.dirname(ext.sourcedir), 'eigen-3.4.0'),
            os.path.join(os.path.dirname(ext.sourcedir), 'eigen'),
            os.path.join(ext.sourcedir, 'eigen'),
            os.path.join(self.build_temp, 'eigen'),
        ]
        
        for eigen_dir in eigen_dirs:
            eigen_core = os.path.join(eigen_dir, 'Eigen', 'Core')
            if os.path.exists(eigen_core):
                print(f"Found Eigen at: {eigen_dir}")
                return eigen_dir
        
        # If Eigen is not found, download and extract it
        print("Downloading Eigen...")
        import urllib.request
        import zipfile
        
        # Create temp directory for download
        eigen_temp_dir = os.path.join(self.build_temp, 'eigen_download')
        if not os.path.exists(eigen_temp_dir):
            os.makedirs(eigen_temp_dir)
        
        # Download Eigen
        url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
        zip_path = os.path.join(eigen_temp_dir, "eigen.zip")
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(eigen_temp_dir)
            
            # Find the extracted directory
            import glob
            extracted_dirs = glob.glob(os.path.join(eigen_temp_dir, 'eigen-*'))
            if extracted_dirs:
                eigen_dir = os.path.join(self.build_temp, 'eigen')
                if os.path.exists(eigen_dir):
                    shutil.rmtree(eigen_dir)
                shutil.copytree(extracted_dirs[0], eigen_dir)
                print(f"Eigen downloaded and installed to {eigen_dir}")
                return eigen_dir
            else:
                print("Failed to find extracted Eigen directory")
        except Exception as e:
            print(f"Error downloading Eigen: {e}")
        
        print("WARNING: Eigen not found and download failed")
        return None

    def _download_pybind11(self, ext):
        """Download pybind11 if not present."""
        pybind11_dir = os.path.join(ext.sourcedir, 'pybind11')
        if os.path.exists(pybind11_dir):
            return pybind11_dir
            
        print("Downloading pybind11...")
        import urllib.request
        import zipfile
        
        # Create temp directory for download
        temp_dir = os.path.join(self.build_temp, 'pybind11_download')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Download pybind11
        url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.4.zip"
        zip_path = os.path.join(temp_dir, "pybind11.zip")
        try:
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the extracted directory
            import glob
            extracted_dirs = glob.glob(os.path.join(temp_dir, 'pybind11-*'))
            if extracted_dirs:
                # Copy to the destination
                if os.path.exists(pybind11_dir):
                    shutil.rmtree(pybind11_dir)
                shutil.copytree(extracted_dirs[0], pybind11_dir)
                print(f"pybind11 downloaded and installed to {pybind11_dir}")
                return pybind11_dir
            else:
                print("Failed to find extracted pybind11 directory")
        except Exception as e:
            print(f"Error downloading pybind11: {e}")
            print("Continuing without pybind11 download...")
        
        return None

    def _find_model_sources(self):
        """Search for the model source files in the project directory."""
        # Start at the project root (parent of python directory)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Check for direct path to src/models
        src_dir = os.path.join(project_root, "src")
        models_dir = os.path.join(src_dir, "models")
        
        if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
            print(f"WARNING: Models directory not found at {models_dir}")
            return None
            
        print(f"Found models directory at: {models_dir}")
        
        # Find the model source files
        model_files = {
            "black_scholes": None,
            "normal": None,
            "stochastic_vol_model": None,
            "log_normal_sv": None,
            "heston_sv": None
        }
        
        # Also look for pricers
        pricers_dir = os.path.join(src_dir, "pricers")
        pricer_files = {}
        if os.path.exists(pricers_dir) and os.path.isdir(pricers_dir):
            print(f"Found pricers directory at: {pricers_dir}")
            pricer_files = {
                "black_scholes_pricer": None
            }
        
        for model_name in model_files.keys():
            model_path = os.path.join(models_dir, f"{model_name}.cpp")
            if os.path.exists(model_path):
                model_files[model_name] = model_path.replace("\\", "/")
                print(f"Found {model_name} at: {model_files[model_name]}")
        
        # Look for pricer implementations
        for pricer_name in pricer_files.keys():
            pricer_path = os.path.join(pricers_dir, f"{pricer_name}.cpp")
            if os.path.exists(pricer_path):
                pricer_files[pricer_name] = pricer_path.replace("\\", "/")
                print(f"Found {pricer_name} at: {pricer_files[pricer_name]}")
        
        # Check if we found all model files
        missing_models = [name for name, path in model_files.items() if path is None]
        if missing_models:
            print(f"WARNING: Could not find the following model source files: {', '.join(missing_models)}")
        
        # Return both models and pricers
        return model_files, pricer_files, src_dir.replace("\\", "/")

    def _create_cmakelists(self, ext, eigen_dir):
        """Create a clean CMakeLists.txt file with proper paths."""
        # Convert paths to use forward slashes for CMake
        ext_dir = ext.sourcedir.replace('\\', '/')
        eigen_dir = eigen_dir.replace('\\', '/') if eigen_dir else ""
        
        # Get the project root (parent of python directory)
        project_root = os.path.abspath(os.path.join(os.path.dirname(ext_dir), '..')).replace('\\', '/')
        
        # Find the model source files
        source_info = self._find_model_sources()
        if not source_info:
            print("WARNING: Could not find model source files, using default paths.")
            src_dir = os.path.join(project_root, "src").replace('\\', '/')
            include_dir = os.path.join(project_root, "include").replace('\\', '/')
            
            model_sources = f"""
    ${{CMAKE_CURRENT_SOURCE_DIR}}/src/bindings.cpp
    {src_dir}/models/black_scholes.cpp
    {src_dir}/models/normal.cpp
    {src_dir}/models/stochastic_vol_model.cpp
    {src_dir}/models/log_normal_sv.cpp
    {src_dir}/models/heston_sv.cpp
    {src_dir}/pricers/black_scholes_pricer.cpp
"""
            include_paths = f"""
    {src_dir}
    {include_dir}
    {ext_dir}/../include
"""
        else:
            model_files, pricer_files, src_dir = source_info
            include_dir = os.path.join(os.path.dirname(src_dir), "include").replace('\\', '/')
            
            # Create the list of source files
            model_sources = """
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp
"""
            for model_path in model_files.values():
                if model_path:
                    model_sources += f"    {model_path}\n"
            
            # Add pricer source files
            for pricer_path in pricer_files.values():
                if pricer_path:
                    model_sources += f"    {pricer_path}\n"
            
            # Find the include directories
            include_paths = f"""
    {src_dir}
    {include_dir}
    {ext_dir}/../include
"""
        
        # Create a new CMakeLists.txt
        cmake_content = f"""cmake_minimum_required(VERSION 3.10)
project(option_pricer_python)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add pybind11
add_subdirectory(pybind11)

# Include directories
include_directories(
{include_paths}
)

# Add Eigen include path
include_directories({eigen_dir})

# Add the extension module
pybind11_add_module(_core
{model_sources}
)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install the extension module
install(TARGETS _core DESTINATION option_pricer)
"""
        
        # Write the new CMakeLists.txt
        cmake_lists_path = os.path.join(ext.sourcedir, 'CMakeLists.txt')
        with open(cmake_lists_path, 'w') as f:
            f.write(cmake_content)
        
        print(f"Created CMakeLists.txt at {cmake_lists_path}")
        
        return cmake_lists_path

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Convert to forward slashes for CMake
        extdir = extdir.replace('\\', '/')

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Download or find Eigen
        eigen_dir = self._find_eigen(ext)

        # Download pybind11 if necessary
        pybind11_dir = self._download_pybind11(ext)

        # Create a fresh CMakeLists.txt
        cmake_lists_path = self._create_cmakelists(ext, eigen_dir)
        print(f"Created CMakeLists.txt at {cmake_lists_path}")

        if platform.system() == "Windows":
            cmake_args.append(f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}')
            
            # Force MinGW Makefiles
            cmake_args += ['-G', 'MinGW Makefiles']
            
            # Specify the C and C++ compilers
            if shutil.which('gcc'):
                cmake_args += ['-DCMAKE_C_COMPILER=gcc']
            if shutil.which('g++'):
                cmake_args += ['-DCMAKE_CXX_COMPILER=g++']
            
            build_args = ['--', '-j4']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        print("CMake args:", cmake_args)
        print("Build args:", build_args)
        print("Source dir:", ext.sourcedir)
        print("Build temp:", self.build_temp)
        
        try:
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            
            # On Windows with MinGW, use mingw32-make instead of the default
            if platform.system() == "Windows":
                make_cmd = 'mingw32-make' if shutil.which('mingw32-make') else 'make'
                subprocess.check_call([make_cmd], cwd=self.build_temp)
            else:
                subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        except subprocess.CalledProcessError as e:
            print(f"Error during build: {e}")
            print("Detailed error information:")
            # Try to find CMake error log
            error_log = os.path.join(self.build_temp, 'CMakeFiles', 'CMakeError.log')
            if os.path.exists(error_log):
                with open(error_log, 'r') as f:
                    print(f.read())
            raise


# Read the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='option-pricer',
    version='0.1.0',
    author='Marwin Steiner',
    author_email='marwin.steiner@gmail.com',
    description='Option pricing library with various models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/marwinsteiner/option-pricer-cpp',
    packages=find_packages(),
    ext_modules=[CMakeExtension('option_pricer._core')],
    cmdclass=dict(build_ext=CMakeBuild),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    license='MIT',
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16.0',
    ],
    zip_safe=False,
)
