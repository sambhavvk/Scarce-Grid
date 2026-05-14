import sys
import platform

def check_environment():
    print("System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    try:
        import numpy
        print(f"NumPy Version: {numpy.__version__}")
    except ImportError:
        print("NumPy is not installed correctly")
    
    try:
        import matplotlib
        print(f"Matplotlib Version: {matplotlib.__version__}")
    except ImportError:
        print("Matplotlib is not installed correctly")
    
    try:
        import seaborn
        print(f"Seaborn Version: {seaborn.__version__}")
    except ImportError:
        print("Seaborn is not installed correctly")

if __name__ == "__main__":
    check_environment()