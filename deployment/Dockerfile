FROM pytorch/pytorch:latest

# Build the container as root
USER root

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

WORKDIR /home/user

# Set the cuda version. This depends on the docker base image!
ENV CUDA=cu110

# Install the pip requirements for the nv
RUN pip install --upgrade pip setuptools && \
    pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html && \
    pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html && \
    pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html && \
    pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html && \
    pip install torch-geometric 

# Copy the deployment code and package
COPY deployment/files ./files 
#COPY packages/model_package ./model_package 

# Copy Agora and install requirements
COPY agora ./agora
RUN pip install -r ./agora/requirements.txt --no-cache-dir

# Prepare the start script and use it as an entrypoint
COPY deployment/start.sh ./start.sh 

# Make our start script executable
RUN ["chmod", "+x", "start.sh"]

# Make sure user has access to all
RUN chown -R user:user .
USER user

# Start the API
ENTRYPOINT [ "./start.sh" ]