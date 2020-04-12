include Makefile.config
.PHONY: all
all: simple
simple:
	g++ $(FLAGS) $(INCLUDE) -c src/main.cpp 
	g++ $(FLAGS) $(INCLUDE) -c src/canny.cpp
	g++ -o $(OUT)/main main.o canny.o $(LIB) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
	rm *.o