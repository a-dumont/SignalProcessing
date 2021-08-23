NAME = correlations
PYLIB_EXT = $(if $(filter $(OS),Windows_NT),.pyd,.so)
TARGET_STATIC = lib$(NAME).a
TARGET_PYLIB = obj/lib$(NAME)$(PYLIB_EXT)

IDIR = includes
ODIR = obj
LDIR = lib
SDIR = src

SRC  = $(wildcard $(SDIR)/*.cpp)
OBJ  = $(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(SRC))
OBJ_PY  = $(filter %_py.o,$(OBJ)) 
ASS  = $(patsubst $(SDIR)/%.cpp,$(ODIR)/%.s,$(SRC))
DEPS = $(OBJ:.o=.d)

EXTERNAL_OBJ = 
EXTERNAL_INCLUDES = 
CXX = $(OS:Windows_NT=x86_64-w64-mingw32-)g++
OPTIMIZATION = -O3 -march=native
CPP_STD = -std=c++14
WARNINGS = -Wall
MINGW_COMPATIBLE = $(OS:Windows_NT=-DMS_WIN64 -D_hypot=hypot)
DEPS_FLAG = -MMD -MP

POSITION_INDEP = -fPIC
SHARED = -shared

FFTW = -lfftw3 -lm
OMP = -fopenmp -fopenmp-simd
MATH = -lm

PY = $(OS:Windows_NT=/c/Anaconda2/)python

PY_INCL := $(shell $(PY) -m pybind11 --includes)

PY_LINKS 	= 	$(OS:Windows_NT=-L /c/Anaconda2/ -lpython27)

LINKS 		= 	$(OMP) $(PY_LINKS) $(FFTW)
LINKING 	= 	$(CXX) $(OPTIMIZATION) $(POSITION_INDEP) $(SHARED) -o $(TARGET_PYLIB) \
				$(OBJ_PY) $(LINKS) $(EXTERNAL_OBJ) $(DEPS_FLAG) $(MINGW_COMPATIBLE)
STATIC_LIB 	= 	ar cr $(TARGET_STATIC) $(OBJ) 

INCLUDES 	= 	$(OMP) $(PY_INCL) $(EXTERNAL_INCLUDES)
COMPILE  	= 	$(CXX) $(CPP_STD) $(OPTIMIZATION) $(POSITION_INDEP) $(WARNINGS) \
				-c -o $@ $< $(INCLUDES) $(DEPS_FLAG) $(MINGW_COMPATIBLE)
ASSEMBLY 	= 	$(CXX) $(CPP_STD) $(OPTIMIZATION) $(POSITION_INDEP) $(WARNINGS) \
				-S -o $@ $< $(INCLUDES) $(DEPS_FLAG) $(MINGW_COMPATIBLE)

python_library : $(TARGET_PYLIB)

compile_objects : $(OBJ)

assembly : $(ASS)

all : $(TARGET_PYLIB) $(TARGET_STATIC) $(OBJ) $(ASS)

static_library : $(TARGET_STATIC)

$(TARGET_PYLIB): $(OBJ_PY)
	@ echo " $(OBJ_PY) "
	@ echo "---------Compile library $(TARGET_PYLIB)---------"
	$(LINKING)

$(TARGET_STATIC) : $(OBJ)
	@ echo " "
	@ echo "---------Compiling static library $(TARGET_STATIC)---------"
	$(STATIC_LIB)
	
$(ODIR)/%.o : $(SDIR)/%.cpp
	@ echo " "
	@ echo "---------Compile object $@ from $<--------"
	$(COMPILE)
	
$(ODIR)/%.s : $(SDIR)/%.cpp
	@ echo " "
	@ echo "---------Assembly $@ from $<--------"
	$(ASSEMBLY)
	
-include $(DEPS)

clean:
	@rm -f $(TARGET_PYLIB) $(TARGET_STATIC) $(OBJ)  $(OBJ_PY) $(ASS) $(DEPS)
	 	 
.PHONY: all , clean , python_library , compile_objects , static_library , assembly
