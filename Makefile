EXEC = string-art
SRC = $(notdir $(wildcard src/*.cc))
OBJECTS = $(addprefix build/,${SRC:.cc=.o})
DEPENDS = ${OBJECTS:.o=.d}

CXX = g++
CXXFLAGS = -O3 -std=c++20 -Wall -Werror=vla -MMD
DBG_CXXFLAGS = -g -std=c++20 -Wall -Werror=vla -MMD
LIBS = $(shell Magick++-config --cppflags --cxxflags)
LDFLAGS = $(shell Magick++-config --ldflags --libs) -fopenmp
#-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment

${EXEC}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} ${LDFLAGS} -o ${EXEC}

build/%.o: src/%.cc
	@mkdir -p build
	${CXX} ${CXXFLAGS} ${LIBS} -c $< -o $@

.PHONY: debug

debug: ${OBJECTS}
	${CXX} ${DBG_CXXFLAGS} ${LIBS} ${LDFLAGS} $(addprefix src/, ${SRC}) -o ${EXEC}

-include ${DEPENDS}

.PHONY: clean

clean:
	rm ${OBJECTS} ${EXEC} ${DEPENDS}
