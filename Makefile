default: build/ep2

build:
	mkdir -p $@
build/ep2: src/ep2.c lib/jlib/jmatrix.h lib/jlib/jnetwork.h | build
	gcc -o $@ -I lib/jlib $< -lm

clean:
	rm -rf build

.PHONY: clean default