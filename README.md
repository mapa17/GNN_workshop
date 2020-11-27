# Agora - working remotely with ease 
Analyzing companies online communication patterns to ensure employees happiness and cooperation.

## Build and run a docker image
**Note**: You have to put a trained model.pkl file into `deployment/files/model.pkl'

Build the docker image
```
docker build -t agora -f deployment/Dockerfile .
```

Run the docker image
```
docker run --rm -it --name agora_inst -p 5000:5000 agora:latest
```
