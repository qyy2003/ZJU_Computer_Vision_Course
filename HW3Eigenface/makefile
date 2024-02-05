INCLUDE = $(shell pkg-config --cflags opencv4)  # 获得 opencv 相关头文件路径
LIBS = $(shell pkg-config --libs opencv4)  # 获得 opencv 扩展库
SOURCES = $(sort $(wildcard *.cpp))
# 目标文件
OBJECTS = $(SOURCES:.cpp=.o)
# 可执行文件
TARGET = show
$(TARGET):$(OBJECTS)
	g++ -o $(TARGET) $(OBJECTS) -I $(INCLUDE) $(LIBS) -std=c++17
$(OBJECTS):$(SOURCES)
	g++ -c $(SOURCES) $(INCLUDE) $(LIBS) -std=c++17
clean:
	rm $(OBJECTS)
# 编译规则 $@代表目标文件 $< 代表第一个依赖文件
%.o:%.cpp
	g++ -I $(INCLUDE) -o $@ -c $<
