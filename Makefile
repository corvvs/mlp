.PHONY: build clean 

depends:
	npm install

build: depends
	npx tsc

clean:
	rm -rf dist
