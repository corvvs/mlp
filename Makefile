.PHONY: build clean

build:
	npx tsc

exec:
# 	node dist/index.js preproc --data=data.csv
# 	node dist/index.js split
	node dist/index.js predict --model=trained.json

run: build exec

train: build
	node dist/index.js train

predict: build
	node dist/index.js predict --model=trained.json

clean:
	rm -rf dist
