#pull the pythin image form the docker hub
# use a lightweight python image
FROM python:3.14-slim
#set the working directory inside container
WORKDIR /app

#copy requirements
COPY requirements.txt .
# installing the dependecies

RUN pip install -r requirements.txt

#Copy applocation code and saved model inside the app directory
COPY . .

# Document the port used by the fast API application

EXPOSE 8000


# starting hte api server when the container is ready
# 0.0.0.0 - allow  connections throught the container's network interface

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]

#Docker Commands:
    #docker ps   #chech  ruuning containers
    # docker ps -a   #for chaching all the 
    # docker build -t mlops-iris-image-cs .
    #docker run -d -p 8000:8000 --name mlops-cs-container mlops-iris-image-cs