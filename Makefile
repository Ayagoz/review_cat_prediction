NAME?=product


GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all build stop run logs

all: stop build run logs

build:
	docker build \
	-t $(NAME) .

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

logs:
	docker logs -f $(NAME)

run:
	docker run --rm -it \
		$(GPUS_OPTION) \
		--net=host \
		--ipc=host \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		bash

run-x11:
	xhost local:root
	docker run --rm -it \
		$(GPUS_OPTION) \
		--net=host \
		--ipc=host \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(HOME)/.Xauthority:/root/.Xauthority \
		-e DISPLAY=$(shell echo ${DISPLAY}) \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		bash