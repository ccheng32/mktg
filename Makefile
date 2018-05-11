DEBUG=n
CPPFLAGS=-fopenmp --std=c++14 -Wall

ifeq ($(DEBUG),n)
	CPPFLAGS+= -O3
else
	CPPFLAGS+= -O0 -g -DDEBUG
endif

OBJ=main.o graph.o tera.o

mktg: $(OBJ)
	g++ $(CPPFLAGS) $(OBJ) -o mktg

main.o: main.cpp graph.h
	g++ $(CPPFLAGS) -c main.cpp

tera.o: tera.cpp graph.h
	g++ $(CPPFLAGS) -c tera.cpp

graph.o: graph.cpp graph.h
	g++ $(CPPFLAGS) -c graph.cpp
clean:
	rm -f *.o mktg
