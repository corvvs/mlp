.PHONY: build clean

build:
	npx tsc

exec:
# 	node dist/index.js preproc --data=data.csv
	node dist/index.js split

run: build exec

clean:
	rm -rf dist
