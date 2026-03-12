IMAGE_NAME := thousand-tasks

.PHONY: help build deploy_mt3 preprocess_demos create_alignment_dataset create_interaction_dataset create_mtact_dataset train_bc_alignment train_bc_interaction train_mtact run debug stop

help:				## Show this help
	@echo 'Usage: make [target]'
	@echo
	@echo 'Targets:'
	@echo
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
	@echo

build:				## Build Docker image
	docker build -t ${IMAGE_NAME} .

.start_if_not_running:
	@if ! docker ps -a | grep -w ${IMAGE_NAME}; then $(MAKE) run; fi

run:				## Starts Docker container in detached mode
	@xhost +si:localuser:root >> /dev/null
	@docker run \
			--detach \
			--rm \
			--net=host \
			--gpus all \
			--privileged \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			--name ${IMAGE_NAME} \
			${IMAGE_NAME} bash -c \
	  		"tail -f /dev/null"

deploy_mt3:			## Run MT3 inference example
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python deployment/deploy_mt3.py

deploy_ret_bc:			## Run Ret-BC inference example (retrieval + BC interaction)
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python deployment/deploy_ret_bc.py

deploy_bc_ret:			## Run BC-Ret inference example (BC alignment + retrieval interaction)
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python deployment/deploy_bc_ret.py

deploy_bc_bc:			## Run BC-BC inference example (BC alignment + BC interaction)
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python deployment/deploy_bc_bc.py

deploy_mtact:			## Run MT-ACT+ inference example (end-to-end multi-task transformer)
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python deployment/deploy_mtact.py

preprocess_demos:		## Preprocess demonstrations for BC training (generates masks and videos)
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python thousand_tasks/demo_preprocessing/preprocess_demos.py

create_alignment_dataset:	## Create processed dataset for BC alignment training
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python thousand_tasks/training/act_bn_reaching/create_dataset.py

train_bc_alignment:		## Train BC alignment (bottleneck reaching) policy
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python thousand_tasks/training/act_bn_reaching/train.py

create_interaction_dataset:	## Create processed dataset for BC interaction training
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python thousand_tasks/training/act_interaction/create_dataset.py

train_bc_interaction:		## Train BC interaction policy
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python thousand_tasks/training/act_interaction/train.py

create_mtact_dataset:		## Create processed dataset for MT-ACT+ end-to-end training
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python thousand_tasks/training/act_end_to_end/create_dataset.py

train_mtact:			## Train MT-ACT+ end-to-end policy
	@xhost +si:localuser:root >> /dev/null
	@docker run --rm \
			--gpus all \
			--net=host \
			-e DISPLAY=${DISPLAY} \
			-v /tmp/.X11-unix/:/tmp/.X11-unix \
			-v ${PWD}:/workspace:rw \
			${IMAGE_NAME} \
			python thousand_tasks/training/act_end_to_end/train.py

debug: 				## Starts a cli inside the container
debug:.start_if_not_running
	@docker exec -it ${IMAGE_NAME} bash



stop:				## Stop Docker container
	@docker stop -t 1 ${IMAGE_NAME} >> /dev/null



