.PHONY: build depends clean

build: depends
	npx tsc

depends:
	npm install

clean:
	rm -rf dist
