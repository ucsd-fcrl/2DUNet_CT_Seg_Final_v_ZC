PORT=8891   #This is the port alloted to you to view Jupyter notebook
HOST_EXP_DIR=$(pwd) # This is the directory on your computer which becomes visible inside the docker
DOCKER_EXP_DIR=/Experiment # This is the name(path) of the directory inside the docker
HOST_DATA_DIR=/media    # This is the directory where permanent data is located
DOCKER_DATA_DIR=/Data  # Data directory visible inside docker
GPUS=2  # Number of GPUs you need in the docker
IMAGE='zhennongchen/cuda100_all_collection_new'  # This is the docker image used to create the container.
NOTEBOOK=false

while getopts ":hi:g:n" opt; do
	case $opt in
	h)
		echo "** Available options **" >&2
		echo " -h for help " >&2
		echo " -g NUM to request NUM GPU units " >&2
		echo " -n to request for Jupyter Notebook access, located at http://localhost:$PORT/tree? " >&2
		echo " -i IMAGE to request creating docker using IMAGE instead of default $IMAGE" >&2		
	        exit 1
		;;
	g) 
	       echo " ** $OPTARG GPUS are requested **" >&2
	       GPUS=$OPTARG
       		;;
	n)
		echo " ** Jupyter Notebook requested and is located at http://localhost:$PORT/tree?  " >&2
		echo "In a separate terminal window, please do: ssh -L $PORT:localhost:$PORT $USER@contijoch-bayern.ucsd.edu  **" >&2
		NOTEBOOK=true
		;;
	i)
		echo " ** Docker image  $OPTARG  is requested **" >&2
		IMAGE=$OPTARG
		;;
	
 	\?)
	echo "Invalid option: -$OPTARG" >&2
	exit 1
	;;
	:)
	echo "Option -$OPTARG requires an argument." >&2
	exit 1
	;;
esac	
done
echo " ** Current directory would be reflected inside the docker with as $DOCKER_DIR ** "

if $NOTEBOOK
then
	if (("$GPUS" > 0 ))
	then
		sudo docker run -it -p $PORT:$PORT -v $HOST_EXP_DIR:$DOCKER_EXP_DIR -v $HOST_DATA_DIR:$DOCKER_DATA_DIR --gpus $GPUS $IMAGE jupyter notebook --port=$PORT --no-browser --ip='0.0.0.0' --allow-root
	else
		sudo docker run -it -p $PORT:$PORT -v $HOST_EXP_DIR:$DOCKER_EXP_DIR -v $HOST_DATA_DIR:$DOCKER_DATA_DIR $IMAGE jupyter notebook --port=$PORT --no-browser --ip='0.0.0.0' --allow-root
		
	fi	
else
	if (("$GPUS" > 0 ))
	then

		sudo docker run -it -v $HOST_EXP_DIR:$DOCKER_EXP_DIR -v $HOST_DATA_DIR:$DOCKER_DATA_DIR --gpus $GPUS $IMAGE bash
	else
		 sudo docker run -it -v $HOST_EXP_DIR:$DOCKER_EXP_DIR  -v $HOST_DATA_DIR:$DOCKER_DATA_DIR  $IMAGE bash
	fi
fi
