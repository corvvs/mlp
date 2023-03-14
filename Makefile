NAMES	:= train predict

all:	install $(NAMES)

install:
	npm i

p:	predict
	node predict.js data.csv

t:	train
	node train.js data.csv

train:
	tsc $@.ts

predict:
	tsc $@.ts

fclean:
	rm -rf *.js libs/*.js
