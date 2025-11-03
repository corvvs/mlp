.PHONY: build clean

build:
	npx tsc

exec:
	node dist/index.js

clean:
	rm -rf dist
