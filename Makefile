.PHONY: build clean

build:
	npx tsc

exec:
	npm run run

run: build exec

clean:
	rm -rf dist
